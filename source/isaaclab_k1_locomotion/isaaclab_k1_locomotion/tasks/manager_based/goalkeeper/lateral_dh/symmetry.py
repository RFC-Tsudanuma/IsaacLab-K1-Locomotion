# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""履歴つき横移動ポリシーの左右対称性 (sagittal plane mirror)。

★ 2026-08-23: ``feat/inoue_walk_double_encoder`` の ``locomotion/mdp/symmetry.py`` を
  横移動タスク向けに移植したもの。「1 ステップ内の置換 + 符号反転を全ステップに繰り返す」
  添字テンソルを一度だけ構築してキャッシュする方式と、mirror loss 時に履歴全体を複製
  しないメモリ最適化をそのまま引き継いでいる。

mirror loss が有効な場合、PPO は以下を最小化する::

    || policy(mirror(obs)) - mirror(policy(obs)) ||^2

☠ **メモリ最適化の要点** (あちらのコメントより): 履歴観測は巨大 (100 ステップ × 数十次元
  × ミニバッチ) なので、``TensorDict.repeat(2)`` で全グループを複製すると OOM する。
  mirror loss だけなら対称性の評価は **直近 1 ステップの生観測** で足りるので、最新
  フレームだけを取り出して反転し :data:`~.history_layout.LATEST_FRAME_GROUP` として返す。
  ネット側 (:class:`~.networks.LateralHistoryActorCritic.get_actor_obs`) がそれを履歴長ぶん
  タイル展開する (= エピソード開始直後のバックフィル状態と同じ、実在しうる観測)。
  data augmentation を有効にしたときだけ履歴全体を反転する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

from .history_layout import (
    CRITIC_TERM_SPECS,
    DIRECT_TERM_SPECS,
    HISTORY_LENGTH,
    LATEST_FRAME_GROUP,
    POLICY_TERM_SPECS,
)

__all__ = ["compute_symmetric_states_lateral_history"]

# JOINT_NAMES_K1 の並び (12 関節):
#    0 Left_Hip_Pitch     6 Right_Hip_Pitch
#    1 Left_Hip_Roll      7 Right_Hip_Roll
#    2 Left_Hip_Yaw       8 Right_Hip_Yaw
#    3 Left_Knee_Pitch    9 Right_Knee_Pitch
#    4 Left_Ankle_Pitch  10 Right_Ankle_Pitch
#    5 Left_Ankle_Roll   11 Right_Ankle_Roll
_LEFT_JOINT_IDX = [0, 1, 2, 3, 4, 5]
_RIGHT_JOINT_IDX = [6, 7, 8, 9, 10, 11]
# 左右を入れ替えた後に掛ける符号。矢状面内で動く pitch / knee はそのまま、
# 面外成分を持つ roll / yaw は符号を反転する。
_JOINT_MIRROR_SIGN = [
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
]

# 反転種別 → (項内の添字置換 (None は恒等), 符号 (None は全て +1))
_KIND_TRANSFORMS = {
    "ang_vel": (None, (-1.0, 1.0, -1.0)),      # roll(x), yaw(z) を反転
    "gravity": (None, (1.0, -1.0, 1.0)),       # 横方向(y) を反転
    "lin_vel": (None, (1.0, -1.0, 1.0)),       # 横方向(y) を反転
    "vel_cmd": (None, (1.0, -1.0, -1.0)),      # lin_vel_y, ang_vel_z を反転
    "joint": (tuple(_RIGHT_JOINT_IDX + _LEFT_JOINT_IDX), tuple(_JOINT_MIRROR_SIGN)),
    "phase": ((2, 3, 0, 1), None),             # [sinL, cosL, sinR, cosR] の左右入れ替え
    "zmp": (None, (1.0, -1.0)),                # 横方向(y) を反転
    # --- direct グループ (GK タスクスロット) 用 ---
    "xy": (None, (1.0, -1.0)),                 # base yaw frame の (x, y)
    "scalar": (None, None),                    # 左右に無関係なフラグ
    "flip": (None, (-1.0,)),                   # 横方向の目標値 (target_y)
    "self_state": (None, (1.0, -1.0, -1.0, 1.0)),  # (x, y, sin(yaw), cos(yaw))
}

# (specs_key, history_length, device, dtype) → (perm, sign)
_MIRROR_CACHE: dict = {}


def _mirror_consts(
    specs_key: str,
    term_specs,
    device: torch.device,
    dtype: torch.dtype,
    history_length: int = HISTORY_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """「項ごとに (履歴長 × 次元) ブロック」レイアウト全体に対する添字置換と符号を作る。

    ``history_length=1`` を渡せば履歴なしのグループ (direct / 最新1フレーム) にも使える。
    """
    key = (specs_key, history_length, device, dtype)
    consts = _MIRROR_CACHE.get(key)
    if consts is None:
        perm: list[int] = []
        sign: list[float] = []
        offset = 0
        for _, dim, kind in term_specs:
            term_perm, term_sign = _KIND_TRANSFORMS[kind]
            term_perm = tuple(range(dim)) if term_perm is None else term_perm
            term_sign = (1.0,) * dim if term_sign is None else term_sign
            if len(term_perm) != dim or len(term_sign) != dim:
                raise ValueError(
                    f"symmetry: 反転種別 '{kind}' の定義が項次元 {dim} と一致しません。"
                    " history_layout.py の項リストと _KIND_TRANSFORMS を突き合わせてください。"
                )
            for step in range(history_length):
                base = offset + step * dim
                perm.extend(base + p for p in term_perm)
                sign.extend(term_sign)
            offset += history_length * dim
        consts = (
            torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=dtype, device=device),
        )
        _MIRROR_CACHE[key] = consts
    return consts


def _mirror_joints(joint_data: torch.Tensor) -> torch.Tensor:
    """関節ごとの量 (..., 12) を左右反転する (行動の反転に使う)。"""
    out = torch.empty_like(joint_data)
    out[..., _LEFT_JOINT_IDX] = joint_data[..., _RIGHT_JOINT_IDX]
    out[..., _RIGHT_JOINT_IDX] = joint_data[..., _LEFT_JOINT_IDX]
    sign = torch.tensor(_JOINT_MIRROR_SIGN, device=joint_data.device, dtype=joint_data.dtype)
    return out * sign


_LATEST_FRAME_IDX_CACHE: dict = {}


def _latest_frame_indices(term_specs, device: torch.device) -> torch.Tensor:
    """履歴グループから「各項の最新 1 ステップ分」を取り出す添字。"""
    key = (term_specs, device)
    idx = _LATEST_FRAME_IDX_CACHE.get(key)
    if idx is None:
        indices: list[int] = []
        offset = 0
        for _, dim, _ in term_specs:
            start = offset + (HISTORY_LENGTH - 1) * dim
            indices.extend(range(start, start + dim))
            offset += HISTORY_LENGTH * dim
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        _LATEST_FRAME_IDX_CACHE[key] = idx
    return idx


def _check_dim(obs: torch.Tensor, name: str, term_specs, history_length: int) -> None:
    expected = sum(dim for _, dim, _ in term_specs) * history_length
    if obs.shape[-1] != expected:
        raise ValueError(
            f"symmetry: 観測グループ '{name}' の次元が想定 ({expected}) と異なります:"
            f" {obs.shape[-1]}。history_layout.py の項リストを env_cfg の観測定義に"
            " 合わせて更新してください。"
        )


@torch.no_grad()
def compute_symmetric_states_lateral_history(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: torch.Tensor | None = None,
):
    """観測・行動に左右対称変換を適用して拡張する (rsl-rl 用)。

    返すバッチは ``[元のサンプル, 左右反転したサンプル]`` の順に連結され、
    バッチサイズが 2 倍になる。
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        from tensordict import TensorDict

        aug_groups: dict = {}

        # --- direct グループ (履歴なし) ---
        direct = obs["direct"]
        _check_dim(direct, "direct", DIRECT_TERM_SPECS, 1)
        perm, sign = _mirror_consts("direct", DIRECT_TERM_SPECS, direct.device, direct.dtype, 1)
        direct_aug = torch.empty(
            2 * batch_size, direct.shape[-1], device=direct.device, dtype=direct.dtype
        )
        direct_aug[:batch_size] = direct
        torch.index_select(direct, -1, perm, out=direct_aug[batch_size:])
        direct_aug[batch_size:].mul_(sign)
        aug_groups["direct"] = direct_aug

        policy = obs["policy"]
        _check_dim(policy, "policy", POLICY_TERM_SPECS, HISTORY_LENGTH)

        if actions is None:
            # mirror loss 用途: 最新 1 フレームだけを反転して返す (履歴を複製しない)。
            frame_idx = _latest_frame_indices(POLICY_TERM_SPECS, policy.device)
            frame = policy.index_select(-1, frame_idx)
            perm, sign = _mirror_consts(
                "policy_frame", POLICY_TERM_SPECS, policy.device, policy.dtype, 1
            )
            frame_aug = torch.empty(
                2 * batch_size, frame.shape[-1], device=frame.device, dtype=frame.dtype
            )
            frame_aug[:batch_size] = frame
            torch.index_select(frame, -1, perm, out=frame_aug[batch_size:])
            frame_aug[batch_size:].mul_(sign)
            aug_groups[LATEST_FRAME_GROUP] = frame_aug
        else:
            # data augmentation 用途: 拡張バッチがそのまま学習に使われるので履歴全体を反転する。
            mirror_targets = [("policy", POLICY_TERM_SPECS)]
            if "critic" in obs.keys():
                _check_dim(obs["critic"], "critic", CRITIC_TERM_SPECS, HISTORY_LENGTH)
                mirror_targets.append(("critic", CRITIC_TERM_SPECS))
            for group_name, term_specs in mirror_targets:
                group = obs[group_name]
                perm, sign = _mirror_consts(group_name, term_specs, group.device, group.dtype)
                group_aug = torch.empty(
                    2 * batch_size, group.shape[-1], device=group.device, dtype=group.dtype
                )
                group_aug[:batch_size] = group
                torch.index_select(group, -1, perm, out=group_aug[batch_size:])
                group_aug[batch_size:].mul_(sign)
                aug_groups[group_name] = group_aug

        obs_aug = TensorDict(aug_groups, batch_size=[2 * batch_size])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
