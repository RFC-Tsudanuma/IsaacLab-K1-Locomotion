# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""1対1ボール奪取へ向けた段階学習環境。

この段階では相手ロボットを置かず、ロボット正面の1 m幅の矩形領域にある静止球を
指定方向へ蹴る。観測・報酬・コマンド・行動空間は ``walk_kick`` と同一に保つため、
既存の ``k1_walk_kick_360`` checkpointをそのまま初期値にできる。
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp as walk_kick_mdp
from ..walk_kick.walk_kick_env_cfg import (
    K1WalkKickEnvCfg,
    K1WalkKickEnvCfg_PLAY,
    _KICK_STATE_PARAMS,
)
from ..walk_kick_both_feet.walk_kick_both_feet_env_cfg import _apply_phase_offset
from . import mdp


def _apply_front_rectangle_ball_spawn(cfg: K1WalkKickEnvCfg) -> None:
    """既存の扇形スポーンを第1段階用の前方矩形スポーンへ置き換える。"""
    cfg.events.reset_ball.func = mdp.reset_ball_in_front_rectangle
    cfg.events.reset_ball.params = {
        "forward_range": (0.4, 1.0),
        "lateral_range": (-0.5, 0.5),
        "ball_radius": 0.11,
    }


def _apply_directional_kick_setup(cfg: K1WalkKickEnvCfg) -> None:
    """側方通常キックと後方ヒールキックに共通する設定を適用する。"""
    _apply_phase_offset(cfg)
    cfg.commands.kick_direction.class_type = mdp.WinTheBallKickDirectionCommand
    cfg.commands.kick_direction.target_speed_range = (0.8, 2.0)
    cfg.commands.base_velocity.class_type = mdp.WinTheBallBallFollowVelocityCommand
    cfg.episode_length_s = 15.0

    cfg.rewards.walk_speed = None
    cfg.curriculum.walk_speed_weight = None
    cfg.rewards.approach_penalty = None
    cfg.curriculum.approach_penalty_weight = None
    cfg.rewards.kick_pose_overshoot = None
    cfg.curriculum.kick_pose_overshoot_weight = None

    cfg.terminations.kick_finished.params["v_thresh_target_frac"] = 0.6
    cfg.terminations.kick_finished.params["v_thresh_floor"] = 0.2
    cfg.commands.base_velocity.v_thresh_target_frac = 0.6
    cfg.commands.base_velocity.v_thresh_floor = 0.2

    cfg.rewards.kick_velocity_strong.weight = 0.0
    cfg.curriculum.kick_velocity_strong_weight = None
    cfg.rewards.kick_velocity_scaled.params["sigma_velocity"] = 0.35
    cfg.rewards.kick_velocity_overshoot = RewTerm(
        func=walk_kick_mdp.kick_velocity_overshoot,
        weight=-0.6,
        params={**_KICK_STATE_PARAMS, "margin": 0.15, "overshoot_sat": 1.0},
    )
    cfg.rewards.quick_accurate_kick = RewTerm(
        func=mdp.quick_accurate_kick,
        weight=8.0,
        params={
            **_KICK_STATE_PARAMS,
            "time_scale": 4.0,
            "sigma_direction": 0.35,
            "sigma_velocity": 0.35,
        },
    )
    cfg.rewards.non_forward_facing_error = RewTerm(
        func=mdp.non_forward_facing_error,
        weight=-0.5,
        params={"command_name": "kick_direction"},
    )
    cfg.rewards.side_foot_kick = RewTerm(
        func=mdp.side_foot_kick,
        weight=4.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": 0.35,
            "sigma_velocity": 0.35,
            "side_saturation": 0.34,
            "command_name": "kick_direction",
        },
    )
    cfg.rewards.side_ball_orthogonality = RewTerm(
        func=mdp.side_ball_orthogonality,
        weight=2.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_dot": 0.25,
            "command_name": "kick_direction",
        },
    )
    cfg.rewards.rear_foot_over_ball_motion = RewTerm(
        func=mdp.rear_foot_over_ball_motion,
        weight=2.0,
        params={
            **_KICK_STATE_PARAMS,
            "min_clearance": 0.15,
            "cross_distance": 0.13,
            "lateral_tolerance": 0.18,
            "straight_rear_halfwidth_deg": 20.0,
            "straight_rear_gain": 3.0,
            "command_name": "kick_direction",
        },
    )
    cfg.rewards.rear_low_crossing_penalty = RewTerm(
        func=mdp.rear_low_crossing_penalty,
        weight=-1.0,
        params={
            "min_clearance": 0.15,
            "danger_longitudinal": 0.16,
            "lateral_tolerance": 0.18,
            "command_name": "kick_direction",
        },
    )
    cfg.rewards.rear_heel_kick = RewTerm(
        func=mdp.rear_heel_kick,
        weight=4.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": 0.35,
            "sigma_velocity": 0.35,
            "command_name": "kick_direction",
        },
    )
    cfg.rewards.direction_matched_foot_kick = RewTerm(
        func=mdp.direction_matched_foot_kick,
        weight=1.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": 0.35,
            "sigma_velocity": 0.35,
            "center_deadzone_deg": 5.0,
            "command_name": "kick_direction",
        },
    )


@configclass
class K1WinTheBallKickEnvCfg(K1WalkKickEnvCfg):
    """前方キック習得後、成功率に応じて側方・後方を解禁する統合タスク。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_front_rectangle_ball_spawn(self)
        _apply_directional_kick_setup(self)
        self.commands.kick_direction.ranges.heading = (-math.pi / 4, math.pi / 4)
        self.curriculum.direction_expansion = CurrTerm(
            func=mdp.kick_success_direction_expansion,
            params={
                "command_name": "kick_direction",
                "metric_window_episodes": 1000,
                "hold_iterations": 1000.0,
                "steps_per_iteration": 48,
                "qualification_rate": 0.9,
                "max_direction_error_deg": 15.0,
                "initial_stage": 0,
                "front_halfwidth": math.pi / 4,
                "side_halfwidth": 3.0 * math.pi / 4,
                "rear_halfwidth": math.pi,
            },
        )


@configclass
class K1WinTheBallKickEnvCfg_PLAY(K1WinTheBallKickEnvCfg):
    """統合タスクを全方向で可視化・評価する設定。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.kick_direction.ranges.heading = (-math.pi, math.pi)
        self.curriculum.direction_expansion = None


@configclass
class K1WinTheBallDirectionalEnvCfg(K1WalkKickEnvCfg):
    """第2段階: 側方キックと、正面姿勢を保つ後方ヒールキックを学習する。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_front_rectangle_ball_spawn(self)
        _apply_directional_kick_setup(self)

        # 旧タスクIDとの互換用。カリキュラムを待たず最初から全方向をサンプルする。
        self.commands.kick_direction.ranges.heading = (-math.pi, math.pi)


@configclass
class K1WinTheBallDirectionalEnvCfg_PLAY(K1WinTheBallDirectionalEnvCfg):
    """第2段階タスクの可視化・評価用設定。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None