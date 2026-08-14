# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー版の左右対称変換 (rsl-rl の mirror loss / data augmentation 用)。

policy 観測が ``59 + (短期 + 長期) × 7`` 次元に伸びるため、既存の
:func:`~..mdp.symmetry.compute_symmetric_states_high_level` (59 次元固定でチェックして
例外を投げる) がそのままでは使えない。先頭 59 の反転は既存実装に委譲し、末尾の履歴ブロック
だけここで反転する。

critic 観測には履歴を足していない (64 次元のまま) ので、そちらは既存実装をそのまま使う。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..mdp.symmetry import (
    _POLICY_OBS_DIM,
    _mirror_gk_critic_obs,
    _mirror_gk_policy_obs,
)
from .observations import GK_HIST_FRAME_DIM

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states_dualhist"]


_HIST_FRAME_MIRROR_SIGN = [
    1.0, -1.0,   # ボール (x, y) フィールド座標系 → y 反転
    1.0,         # 検出マスク (左右に無関係)
    1.0, -1.0,   # 自機 (x, y) → y 反転
    -1.0, 1.0,   # sin(heading) 反転 / cos(heading) そのまま
]
assert len(_HIST_FRAME_MIRROR_SIGN) == GK_HIST_FRAME_DIM

# 上位 action = 歩行コマンド (vx, vy, wz)。矢状面の鏡像では横速度と旋回だけ符号反転。
_HIGH_ACTION_MIRROR_SIGN = [1.0, -1.0, -1.0]

_CACHE: dict = {}


def _sign(key, values, n_repeat: int, device, dtype) -> torch.Tensor:
    ck = (key, n_repeat, device, dtype)
    t = _CACHE.get(ck)
    if t is None:
        t = torch.tensor(list(values) * n_repeat, device=device, dtype=dtype)
        _CACHE[ck] = t
    return t


def _mirror_policy_obs_dh(obs: torch.Tensor) -> torch.Tensor:
    """policy 観測 (N, 59 + T×7) を矢状面に対して左右反転する。"""
    extra = obs.shape[-1] - _POLICY_OBS_DIM
    if extra < 0 or extra % GK_HIST_FRAME_DIM != 0:
        raise ValueError(
            f"symmetry(dualhist): policy 観測の次元 {obs.shape[-1]} が"
            f" {_POLICY_OBS_DIM} + N×{GK_HIST_FRAME_DIM} の形になっていません。"
            " 履歴フレームの構成を変えたなら _HIST_FRAME_MIRROR_SIGN も更新すること。"
        )

    out = torch.empty_like(obs)
    # 先頭 59 (歩行 49 + タスク 10) は既存の反転をそのまま使う
    out[:, :_POLICY_OBS_DIM] = _mirror_gk_policy_obs(obs[:, :_POLICY_OBS_DIM])
    if extra > 0:
        n_frames = extra // GK_HIST_FRAME_DIM
        sign = _sign("hist", _HIST_FRAME_MIRROR_SIGN, n_frames, obs.device, obs.dtype)
        out[:, _POLICY_OBS_DIM:] = obs[:, _POLICY_OBS_DIM:] * sign
    return out


@torch.no_grad()
def compute_symmetric_states_dualhist(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: torch.Tensor | None = None,
):
    """観測・行動に左右対称変換を適用して拡張する (rsl-rl 用)。

    返すバッチは ``[元のサンプル, 左右反転したサンプル]`` の順に連結され、バッチサイズが
    2 倍になる (既存の goalkeeper 実装と同じ規約)。
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs_dh(obs["policy"])
        if "critic" in obs_aug.keys():
            # critic には履歴を足していないので既存の 64 次元用の反転をそのまま使う
            obs_aug["critic"][:batch_size] = obs["critic"][:]
            obs_aug["critic"][batch_size:] = _mirror_gk_critic_obs(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        if actions.shape[-1] != len(_HIGH_ACTION_MIRROR_SIGN):
            raise ValueError(
                f"symmetry(dualhist): 上位 action の次元が想定"
                f" ({len(_HIGH_ACTION_MIRROR_SIGN)}) と異なります: {actions.shape[-1]}"
            )
        batch_size = actions.shape[0]
        sign = _sign("high_action", _HIGH_ACTION_MIRROR_SIGN, 1, actions.device, actions.dtype)
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = actions * sign
    else:
        actions_aug = None

    return obs_aug, actions_aug
