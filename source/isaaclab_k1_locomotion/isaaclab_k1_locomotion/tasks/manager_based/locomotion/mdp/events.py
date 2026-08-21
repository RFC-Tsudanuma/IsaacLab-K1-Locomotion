# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 専用のイベント関数。"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_PHASE_FREQ_ATTR = "_phase_freq_per_env"
_PHASE_OFFSET_ATTR = "_phase_offset_per_env"


def randomize_phase_freq(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    base_phase_freq: float,
    offset_range: tuple[float, float] = (-0.1, 0.1),
):
    """環境毎の歩行周波数を ``base_phase_freq + uniform(offset_range)`` でランダム化する。

    結果は ``env._phase_freq_per_env`` (shape ``[num_envs]``) に保持し、
    位相を扱う観測/報酬関数 (``phase_obs``, ``feet_phase``, ``foot_clearance_ji_pen`` 等)
    から :func:`get_phase_freq` 経由で参照する。
    """
    base = float(base_phase_freq)

    buf: torch.Tensor | None = getattr(env, _PHASE_FREQ_ATTR, None)
    if buf is None:
        buf = torch.full((env.num_envs,), base, device=env.device)
        setattr(env, _PHASE_FREQ_ATTR, buf)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    low, high = float(offset_range[0]), float(offset_range[1])
    offsets = torch.empty(env_ids.numel(), device=env.device).uniform_(low, high)
    buf[env_ids] = base + offsets


def get_phase_freq(env: "ManagerBasedEnv", default: float) -> "float | torch.Tensor":
    """環境毎にランダム化された位相周波数があればそれを、無ければスカラー ``default`` を返す。"""
    val = getattr(env, _PHASE_FREQ_ATTR, None)
    if val is None:
        return default
    return val


def randomize_phase_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    mode: str = "binary",
):
    """歩行位相の **初期オフセット** を env ごとに振る (``mode="reset"`` で使う)。

    位相は ``2π * pf * t`` (t = エピソード開始からの経過時間) で計算されるので、この項が
    無いと **全 env・全エピソードで必ず位相 0 から歩き始める**。stance_ratio=0.5 の
    ``feet_phase`` では位相 0 の直後は「左が支持脚・右が遊脚」なので、**歩き出しの遊脚が
    常に右足**に固定される。キックタスクではこれが「右足でしか蹴らない」方策に直結する
    (蹴り足は接触時に遊脚だった方の足なので、位相が時刻の決定論的な関数だと片側に固定される)。

    オフセットを振ると同じ経過時間でも遊脚の左右が env ごとに入れ替わるので、鏡像の解が
    どちらも学習データに現れる。方策から見た位相は ``gait_phase`` 観測 (sin/cos) に
    そのまま出るので、オフセットを別途観測に足す必要はない。

    Args:
        mode: ``"binary"`` は {0, π} を等確率 (歩き出しの遊脚が左右ちょうど半々になる。
            歩容そのものは今までと同じ位相構造のまま鏡像になるだけなので、既定はこちら)。
            ``"uniform"`` は [0, 2π) の一様乱数 (エピソード開始位置が周期の途中になる
            ぶん難しくなるが、位相と経過時間の相関が完全に切れる)。

    結果は ``env._phase_offset_per_env`` (shape ``[num_envs]``) に保持し、位相を扱う
    観測/報酬関数から :func:`get_phase_offset` 経由で参照する。この関数を event に
    登録していない環境では :func:`get_phase_offset` が 0.0 を返すので、既存タスクの
    挙動は 1 ビットも変わらない (:func:`randomize_phase_freq` と同じ流儀)。
    """
    buf: torch.Tensor | None = getattr(env, _PHASE_OFFSET_ATTR, None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _PHASE_OFFSET_ATTR, buf)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    n = env_ids.numel()
    if mode == "binary":
        pick = torch.randint(0, 2, (n,), device=env.device, dtype=torch.float32)
        buf[env_ids] = pick * math.pi
    elif mode == "uniform":
        buf[env_ids] = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * math.pi)
    else:
        raise ValueError(f"randomize_phase_offset: 未知の mode '{mode}' (binary / uniform)")


def get_phase_offset(env: "ManagerBasedEnv") -> "float | torch.Tensor":
    """env ごとの位相オフセットがあればそれを、無ければスカラー 0.0 を返す。

    位相を計算する全ての関数はこれを足すこと::

        phase_left = (2π * pf * t + get_phase_offset(env)) % (2π)

    :func:`randomize_phase_offset` が登録されていない環境では 0.0 が返るので、
    足しても既存の値は変わらない。
    """
    val = getattr(env, _PHASE_OFFSET_ATTR, None)
    if val is None:
        return 0.0
    return val


def reset_prev_high_action(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
):
    """リセットされた env の ``_prev_high_action`` バッファを 0 にする。

    バッファ実体は ``HierarchicalVecEnvWrapper`` が用意するので、本関数は無ければ
    no-op で返す。Observation 計算は ``_reset_idx`` の後に走るので、ここで 0 化
    しておけば新エピソード最初の観測 ``last_high_action`` も 0 になる。
    """
    buf = getattr(env, "_prev_high_action", None)
    if buf is None:
        return
    if env_ids is None:
        buf.zero_()
        return
    buf[env_ids] = 0.0


__all__ = [
    "randomize_phase_freq",
    "get_phase_freq",
    "randomize_phase_offset",
    "get_phase_offset",
    "reset_prev_high_action",
]
