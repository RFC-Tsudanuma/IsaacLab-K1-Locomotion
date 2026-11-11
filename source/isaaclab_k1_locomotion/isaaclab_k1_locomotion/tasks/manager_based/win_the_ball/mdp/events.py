# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_in_front_rectangle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    forward_range: tuple[float, float] = (0.4, 1.0),
    lateral_range: tuple[float, float] = (-0.5, 0.5),
    ball_radius: float = 0.11,
) -> None:
    """ボールをロボット基準の前方矩形内へ静止配置する。

    ``forward_range`` と ``lateral_range`` はロボットの yaw 座標系で指定する。
    前方下限は、リセット直後に足とボールが接触しない値にすること。
    """
    if forward_range[0] < 0.0 or forward_range[0] > forward_range[1]:
        raise ValueError(f"不正な forward_range: {forward_range}")
    if lateral_range[0] > lateral_range[1]:
        raise ValueError(f"不正な lateral_range: {lateral_range}")

    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    count = len(env_ids)

    robot_pos_w = robot.data.root_pos_w[env_ids]
    quat = robot.data.root_quat_w[env_ids]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    forward = torch.empty(count, device=env.device).uniform_(*forward_range)
    lateral = torch.empty(count, device=env.device).uniform_(*lateral_range)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    state = ball.data.default_root_state[env_ids].clone()
    state[:, 0] = robot_pos_w[:, 0] + forward * cos_yaw - lateral * sin_yaw
    state[:, 1] = robot_pos_w[:, 1] + forward * sin_yaw + lateral * cos_yaw
    state[:, 2] = env.scene.env_origins[env_ids, 2] + ball_radius
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0

    ball.write_root_state_to_sim(state, env_ids=env_ids)