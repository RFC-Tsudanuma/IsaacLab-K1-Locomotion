# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos_rel(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットベースフレーム（yaw aligned）でのボール相対位置。shape: (N, 3)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    rel_pos_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)
    return rel_pos_b


def ball_vel_rel(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットベースフレーム（yaw aligned）でのボール速度。shape: (N, 3)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    vel_w = ball.data.root_lin_vel_w[:, :3]
    vel_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), vel_w)
    return vel_b


def kick_dir_sincos(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """蹴り方向コマンドを (sin θ, cos θ) で返す。shape: (N, 2)"""
    command = env.command_manager.get_command(command_name)  # (N, 3) [sin θ, cos θ, 0]
    return command[:, :2]
