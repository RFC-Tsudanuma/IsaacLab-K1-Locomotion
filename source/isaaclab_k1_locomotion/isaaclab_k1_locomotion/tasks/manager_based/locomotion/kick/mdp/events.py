# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_ball_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    ball: RigidObject = env.scene[asset_cfg.name]
    root_states = ball.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=ball.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in vel_keys], device=ball.device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    ball.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocities, env_ids=env_ids)
