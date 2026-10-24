# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bhuman_walk_kick の観測項。

1 ステップの観測 = 土台の walk の 49 項 (そのままの並び) + キック用の項
(``locomotion.history_layout.KICK_EXTRA_TERM_SPECS``)。キック用の項の中身は
B-Human ``t1_ball.py::_compute_observations`` のボール・キック関係と同じ:

    ball_position (2)          両足裏の中点から見たボール位置 (yaw フレーム, 1-3 ステップ遅れ) * 0.1
    kick_direction (1)         キック方向 / pi (胴体の向き基準)
    kick_direction_sincos (2)  (sin, cos)
    kick_flags (3)             強キック / 精度不問 / 逸脱可
    ball_velocity (2)          ボール速度 (yaw フレーム) * 0.1
    expected_ball_velocity (1) 目標距離に必要なボール初速 * 0.1
    gait_frequency_action (1)  前ステップの歩行周波数の action

B-Human の「1 つ前のボール位置」は観測履歴 (100 ステップ) に含まれるので入れない。
ボール関係のノイズは今と過去で同じノイズを共有するなどの癖があるので、ObsTerm の noise では
なく :meth:`BHumanKickCommand.obs_cache` で引く。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .kick_command import BHumanKickCommand


def _kick(env: ManagerBasedRLEnv, command_name: str) -> BHumanKickCommand:
    term = env.command_manager.get_term(command_name)
    term.obs_cache()
    return term


def last_joint_action(env: ManagerBasedRLEnv, num_joints: int = 12) -> torch.Tensor:
    """前ステップの関節 action (脚 12)。walk の ``actions`` 項と同じ幅に揃えるため周波数は除く。"""
    return env.action_manager.action[:, :num_joints]


def gait_frequency_action(env: ManagerBasedRLEnv, clip: float = 2.0) -> torch.Tensor:
    """前ステップの歩行周波数の action (B-Human と同じく ±2 で clip した生値)。"""
    return env.action_manager.action[:, -1:].clamp(-clip, clip)


def ball_position(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    return _kick(env, command_name).obs_ball_pos


def kick_direction(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    return (_kick(env, command_name).obs_direction_angle / torch.pi).unsqueeze(-1)


def kick_direction_sincos(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    a = _kick(env, command_name).obs_direction_angle
    return torch.stack((torch.sin(a), torch.cos(a)), dim=-1)


def kick_flags(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    k = _kick(env, command_name)
    return torch.stack(
        (k.flag_is_strong_kick.float(), k.flag_is_inaccurate_kick, k.flag_allow_deviation.float()), dim=-1
    )


def ball_velocity(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    return _kick(env, command_name).obs_ball_vel


def expected_ball_velocity(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    return _kick(env, command_name).obs_expected_vel.unsqueeze(-1)


def kick_pose_state(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    """critic 専用: B-Human の inside_kick_pose_counter と ball_close_to_kick_pose。"""
    k = _kick(env, command_name)
    return torch.stack((k.inside_kick_pose_counter, k.ball_close_to_kick_pose), dim=-1)
