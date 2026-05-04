# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def touch_ball(
    env: ManagerBasedRLEnv,
    sensor_cfg_right: SceneEntityCfg = SceneEntityCfg("contact_balls_right"),
    sensor_cfg_left: SceneEntityCfg = SceneEntityCfg("contact_balls_left"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """足がボールに接触したときの報酬。どちらの足でも可。shape: (N,)"""
    sensor_right: ContactSensor = env.scene[sensor_cfg_right.name]
    sensor_left: ContactSensor = env.scene[sensor_cfg_left.name]

    # net_forces_w: (N, n_bodies, 3) — フィルタ済みなのでボールからの力のみ
    force_right = sensor_right.data.net_forces_w[:, 0, :].norm(dim=-1)
    force_left = sensor_left.data.net_forces_w[:, 0, :].norm(dim=-1)

    return ((force_right > threshold) | (force_left > threshold)).float()


def ball_distance(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールに近いほど高い報酬（接近を促す）。shape: (N,)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    dist = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return torch.exp(-dist)


def kick_ball_velocity(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """ボールが速度コマンド方向に動いているときの報酬。shape: (N,)"""
    ball = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w[:, :2]  # (N, 2)

    command = env.command_manager.get_command(command_name)  # (N, 3) [vx, vy, wz]
    cmd_dir = command[:, :2]
    cmd_dir_normalized = cmd_dir / (cmd_dir.norm(dim=-1, keepdim=True) + 1e-6)

    ball_speed_in_cmd_dir = (ball_vel * cmd_dir_normalized).sum(dim=-1)
    return torch.clamp(ball_speed_in_cmd_dir, min=0.0)


def kick_ball_forward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールがロボットの前方に動いているときの報酬（コマンドマネージャー不要）。shape: (N,)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    ball_vel_w = ball.data.root_lin_vel_w[:, :2]

    # ロボットのヨー角から前方ベクトルを計算
    quat = robot.data.root_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (N, 2)

    speed_forward = (ball_vel_w * forward).sum(dim=-1)
    return torch.clamp(speed_forward, min=0.0)
