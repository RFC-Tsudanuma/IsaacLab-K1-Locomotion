# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_position_world(env: ManagerBasedRLEnv, goal_pos: tuple[float, float, float]) -> torch.Tensor:
    goal = torch.tensor(goal_pos, device=env.device, dtype=torch.float32)
    return env.scene.env_origins + goal.unsqueeze(0)


def ball_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    return ball.data.root_pos_w - env.scene.env_origins


def ball_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[asset_cfg.name]
    return ball.data.root_lin_vel_w


def ball_pos_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)


def goal_direction(
    env: ManagerBasedRLEnv,
    goal_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    goal_pos_w = _goal_position_world(env, goal_pos)
    goal_vec_w = goal_pos_w - ball.data.root_pos_w
    goal_dir_w = goal_vec_w / torch.clamp(torch.linalg.norm(goal_vec_w, dim=1, keepdim=True), min=1.0e-6)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_dir_w)
