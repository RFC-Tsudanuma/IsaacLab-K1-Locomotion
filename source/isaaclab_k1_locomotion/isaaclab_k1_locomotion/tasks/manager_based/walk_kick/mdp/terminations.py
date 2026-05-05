# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reached_ball(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """ロボットがボールに十分近づいたら True を返す終了条件。shape: (N,) bool"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    dist = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return dist < threshold
