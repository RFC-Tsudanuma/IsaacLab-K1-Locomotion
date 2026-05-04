# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# observations.py

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Helper Functions
##

def phase_obs(
    env: ManagerBasedRLEnv,
    phase_freq: float = 1.5,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """現在の歩行位相を sin/cos で返す (左足, 右足の計4次元)。

    コマンド速度が ``cmd_threshold`` 以下のときは位相をゼロで埋め、
    停止すべき状況であることをポリシーに明示する。
    """
    t = env.episode_length_buf * env.step_dt
    phase_left = 2.0 * math.pi * phase_freq * t
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed <= cmd_threshold
    phase = torch.where(is_stopped, torch.zeros_like(phase), phase)

    return phase
