# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 専用のイベント関数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_PHASE_FREQ_ATTR = "_phase_freq_per_env"


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


__all__ = ["randomize_phase_freq", "get_phase_freq", "reset_prev_high_action"]
