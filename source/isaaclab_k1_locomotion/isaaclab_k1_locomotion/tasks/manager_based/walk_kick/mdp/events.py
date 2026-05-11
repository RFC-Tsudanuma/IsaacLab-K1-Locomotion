# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sample_max_walking_speed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.5, 1.5),
    y_range: tuple[float, float] = (0.5, 1.0),
    yaw_range: tuple[float, float] = (1.0, 1.5),
):
    """エピソードリセット時に最大歩行速度 [vx, vy, wz] をランダムサンプリングして env に保持する。"""
    if not hasattr(env, "_max_walking_speed"):
        env._max_walking_speed = torch.zeros(env.num_envs, 3, device=env.device)

    n = len(env_ids)
    env._max_walking_speed[env_ids, 0] = torch.empty(n, device=env.device).uniform_(*x_range)
    env._max_walking_speed[env_ids, 1] = torch.empty(n, device=env.device).uniform_(*y_range)
    env._max_walking_speed[env_ids, 2] = torch.empty(n, device=env.device).uniform_(*yaw_range)


def sample_kick_range(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    speed_range: tuple[float, float] = (1.0, 2.0),
):
    """エピソードリセット時に目標ボール速度（kick range）をランダムサンプリングして env に保持する。"""
    if not hasattr(env, "_kick_range"):
        env._kick_range = torch.zeros(env.num_envs, 1, device=env.device)

    n = len(env_ids)
    env._kick_range[env_ids, 0] = torch.empty(n, device=env.device).uniform_(*speed_range)


def reset_ball_polar(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    r_min: float = 0.5,
    r_max: float = 1.0,
):
    """エピソードリセット時にボールをロボット初期位置から [r_min, r_max] m の距離にランダム配置。"""
    ball = env.scene[ball_cfg.name]
    n = len(env_ids)

    r = torch.empty(n, device=env.device).uniform_(r_min, r_max)
    theta = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)

    origins = env.scene.env_origins[env_ids]
    default_z = ball.data.default_root_state[env_ids, 2]

    ball_state = ball.data.default_root_state[env_ids].clone()
    ball_state[:, 0] = origins[:, 0] + r * torch.cos(theta)
    ball_state[:, 1] = origins[:, 1] + r * torch.sin(theta)
    ball_state[:, 2] = origins[:, 2] + default_z
    ball_state[:, 7:] = 0.0

    ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
