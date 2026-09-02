# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# observations.py

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .events import get_gait_phase, get_phase_freq
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
    adaptive: bool = False,
    l_fwd: float = 0.31,
    l_lat: float = 0.11,
    l_back: float = 0.0,
    f_min: float = 1.2,
    f_max: float = 5.0,
    dr_base: float = 1.6,
    use_actual_speed: bool = False,
    cmd_gain: float = 1.0,
    vel_lag_s: float = 0.0,
    vel_noise_std: float = 0.0,
    lateral_phase_flip: bool = False,
) -> torch.Tensor:
    """現在の歩行位相を sin/cos で返す (左足, 右足の計4次元)。

    コマンド速度が ``cmd_threshold`` 未満のときは位相をゼロで埋め、
    停止すべき状況であることをポリシーに明示する。

    ``adaptive=True`` にすると位相の周波数を **速度指令とその向き** から決める
    (:func:`~.events.adaptive_phase_freq`)。横歩きは歩幅が前進の 60% しか出ないので、
    固定周波数では前進と横を両立できないため。既定 False で従来どおり。
    ☠ ``adaptive=True`` のときは ``reset_gait_phase`` を EventTerm に登録すること。
    """
    phase_left = get_gait_phase(
        env, phase_freq, adaptive=adaptive, command_name=command_name,
        l_fwd=l_fwd, l_lat=l_lat, l_back=l_back, f_min=f_min, f_max=f_max, dr_base=dr_base,
        use_actual_speed=use_actual_speed, cmd_gain=cmd_gain,
        vel_lag_s=vel_lag_s, vel_noise_std=vel_noise_std,
        lateral_phase_flip=lateral_phase_flip,
    )
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed < cmd_threshold
    phase = torch.where(is_stopped, torch.zeros_like(phase), phase)

    return phase

def ball_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ボールの線速度を、ロボットの base yaw frame で返す (x: 前後, y: 左右)。"""
    ball: Articulation = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    vel_w = ball.data.root_com_vel_w[:, :3]
    vel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), vel_w)
    return vel_b[:, :2]

def ball_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ボールとロボットの相対位置を、ロボットの base yaw frame で返す (x: 前後, y: 左右)。"""
    ball: Articulation = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    offset_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    offset_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), offset_w)
    return offset_b[:, :2]


def last_high_action(
    env: ManagerBasedRLEnv,
    action_dim: int = 3,
) -> torch.Tensor:
    """前回ステップで上位ポリシーが出力した行動 (歩行コマンド) を返す。

    ``HierarchicalVecEnvWrapper`` が毎ステップ ``env._prev_high_action``
    (shape ``(num_envs, action_dim)``) を更新する。リセット時は
    :func:`mdp.events.reset_prev_high_action` で対象 env のスロットが 0 にされる。
    未初期化のときは 0 を返す (起動時など)。
    """
    buf = getattr(env, "_prev_high_action", None)
    if buf is None or buf.shape != (env.num_envs, action_dim):
        return torch.zeros((env.num_envs, action_dim), device=env.device)
    return buf


def kick_direction_b(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ワールド座標系のキック方向コマンドをロボットの base yaw frame に回転して返す (2D 単位ベクトル)。"""
    robot: Articulation = env.scene[asset_cfg.name]
    kick_dir_w_xy = env.command_manager.get_term(command_name).command  # (N, 2)
    # 3D に拡張 (z=0) してから base yaw frame に回転
    z = torch.zeros_like(kick_dir_w_xy[:, :1])
    kick_dir_w = torch.cat([kick_dir_w_xy, z], dim=1)
    kick_dir_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), kick_dir_w)
    return kick_dir_b[:, :2]
