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

def phase_obs(env: ManagerBasedRLEnv, phase_freq: float = 1.5) -> torch.Tensor:
    """現在の歩行位相を sin/cos で返す (左足, 右足の計4次元)"""
    t = env.episode_length_buf * env.step_dt
    phase_left = 2.0 * math.pi * phase_freq * t
    phase_right = phase_left + math.pi

    return torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)
