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


def ball_kicked_after_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg_right: SceneEntityCfg = SceneEntityCfg("contact_balls_right"),
    sensor_cfg_left: SceneEntityCfg = SceneEntityCfg("contact_balls_left"),
    delay_steps: int = 20,
    contact_threshold: float = 0.5,
) -> torch.Tensor:
    """足がボールに触れてから delay_steps ステップ後に True を返す終了条件。shape: (N,) bool

    delay_steps=20 は制御周波数 50Hz で約 0.4 秒に相当。
    """
    sensor_right: ContactSensor = env.scene[sensor_cfg_right.name]
    sensor_left: ContactSensor = env.scene[sensor_cfg_left.name]

    force_right = sensor_right.data.net_forces_w[:, 0, :].norm(dim=-1)
    force_left = sensor_left.data.net_forces_w[:, 0, :].norm(dim=-1)
    contacted = (force_right > contact_threshold) | (force_left > contact_threshold)

    if not hasattr(env, "_kick_contact_counter"):
        env._kick_contact_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # エピソードリセット時にカウンタをリセット
    just_reset = env.episode_length_buf <= 1
    env._kick_contact_counter[just_reset] = 0

    # 接触開始またはカウント中 → まだ delay_steps に達していなければインクリメント
    counting = env._kick_contact_counter > 0
    should_increment = (counting | contacted) & (env._kick_contact_counter < delay_steps)
    env._kick_contact_counter[should_increment] += 1

    return env._kick_contact_counter >= delay_steps
