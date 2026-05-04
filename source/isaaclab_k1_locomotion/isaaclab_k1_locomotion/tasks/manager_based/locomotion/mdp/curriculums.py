# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the K1 locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_command_resampling_time_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    resampling_time_range: tuple[float, float],
    num_steps: int,
):
    """指定ステップ数を超えたら、コマンドのリサンプリング時間範囲を変更する。"""
    if env.common_step_counter > num_steps:
        term = env.command_manager.get_term(command_name)
        term.cfg.resampling_time_range = resampling_time_range
