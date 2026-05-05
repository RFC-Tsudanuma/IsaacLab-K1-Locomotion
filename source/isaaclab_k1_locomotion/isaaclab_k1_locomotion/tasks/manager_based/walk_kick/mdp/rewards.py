# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

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


def ball_in_front(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    fov_half_angle: float = 0.524,
) -> torch.Tensor:
    """ボールがロボットのカメラ視野内にあるときの報酬。shape: (N,)

    ロボット前方を基準に水平角が fov_half_angle [rad] 以内なら 1.0，
    それ以外は指数的に減衰。デフォルト 0.524 rad ≈ 30°（片側）。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]

    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    rel_pos_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)

    x_rel = rel_pos_b[:, 0]
    y_rel = rel_pos_b[:, 1]

    # ロボット前方からボールへの水平角（絶対値）
    horiz_angle = torch.atan2(torch.abs(y_rel), x_rel.clamp(min=1e-6))

    in_front = (x_rel > 0.0).float()
    # FOV 内は 1.0，外側は指数減衰
    in_fov_score = torch.exp(-((horiz_angle - fov_half_angle).clamp(min=0.0) ** 2) / (fov_half_angle ** 2))
    return in_front * in_fov_score


def robot_xy_speed(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """ロボットの水平面（XY）移動速度ノルム。shape: (N,)"""
    robot = env.scene["robot"]
    return robot.data.root_lin_vel_w[:, :2].norm(dim=-1)


def reach_ball_bonus(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """ボールに到達したとき（かつ time_out でないとき）に 1 を返す成功ボーナス。shape: (N,)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    dist = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    reached = dist < threshold
    not_timeout = ~env.termination_manager.time_outs
    return (reached & not_timeout).float()


def approach_ball(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    sigma: float = 1.0,
) -> torch.Tensor:
    """ボールに近いほど高い報酬（蹴り位置への接近を促す）。shape: (N,)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    dist = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return torch.exp(-dist / sigma)


def align_to_kick_direction(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """ロボットのヨー方向が kick_direction コマンドと一致しているときの報酬。shape: (N,)

    command = [sin θ, cos θ, 0]（KickDirectionCommand の形式）。
    前方向ベクトルとの cos 類似度を [0, 1] にクリップして返す。
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)  # (N, 3) [sin θ, cos θ, 0]

    quat = robot.data.root_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # ロボット前方ベクトル (cos yaw, sin yaw)
    forward_x = torch.cos(yaw)
    forward_y = torch.sin(yaw)

    # kick_direction ベクトル: command[0]=sin θ, command[1]=cos θ → (cos θ, sin θ)
    kick_x = command[:, 1]  # cos θ
    kick_y = command[:, 0]  # sin θ

    cos_sim = forward_x * kick_x + forward_y * kick_y
    return torch.clamp(cos_sim, min=0.0)


def kick_direction_exp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    command_name: str = "kick_direction",
    sigma: float = 0.5,
) -> torch.Tensor:
    """ボール速度方向が kick_direction コマンドと一致しているときの指数報酬。shape: (N,)

    ボールが動いていないときは報酬なし。
    sigma: 方向誤差の許容幅（小さいほど厳密な方向一致を要求）。
    """
    ball = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w[:, :2]  # (N, 2)
    ball_speed = ball_vel.norm(dim=-1)

    moving = (ball_speed > 0.01).float()
    ball_dir = ball_vel / (ball_speed.unsqueeze(-1) + 1e-6)

    command = env.command_manager.get_command(command_name)  # (N, 3) [sin θ, cos θ, 0]
    kick_dir = torch.stack([command[:, 1], command[:, 0]], dim=-1)  # (cos θ, sin θ)

    cos_sim = (ball_dir * kick_dir).sum(dim=-1)
    return moving * torch.exp(-(1.0 - cos_sim) / sigma)


def kick_velocity_exp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    command_name: str = "kick_direction",
    sigma: float = 1.0,
) -> torch.Tensor:
    """kick_direction 方向のボール速度が大きいときの指数報酬。shape: (N,)

    1 - exp(-v / sigma) の形式で，速度が増すほど 1 に近づく。
    sigma: 報酬が飽和する速度スケール [m/s]。
    """
    ball = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w[:, :2]  # (N, 2)

    command = env.command_manager.get_command(command_name)  # (N, 3) [sin θ, cos θ, 0]
    kick_dir = torch.stack([command[:, 1], command[:, 0]], dim=-1)  # (cos θ, sin θ)

    speed_in_kick_dir = (ball_vel * kick_dir).sum(dim=-1)
    return 1.0 - torch.exp(-torch.clamp(speed_in_kick_dir, min=0.0) / sigma)


def single_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg_right: SceneEntityCfg = SceneEntityCfg("contact_balls_right"),
    sensor_cfg_left: SceneEntityCfg = SceneEntityCfg("contact_balls_left"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """両足同時接触ペナルティ（片足接触を推奨）。負の重みで使用。shape: (N,)

    両足が同時にボールに接触しているとき 1.0 を返す。
    負の重みを設定することで同時接触を抑制し，片足キックを促す。
    """
    sensor_right: ContactSensor = env.scene[sensor_cfg_right.name]
    sensor_left: ContactSensor = env.scene[sensor_cfg_left.name]

    force_right = sensor_right.data.net_forces_w[:, 0, :].norm(dim=-1)
    force_left = sensor_left.data.net_forces_w[:, 0, :].norm(dim=-1)

    both_contact = (force_right > threshold) & (force_left > threshold)
    return both_contact.float()
