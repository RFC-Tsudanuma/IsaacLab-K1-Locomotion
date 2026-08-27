# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inside-kick training with the canonical CVKF horizon observation.

The stationary phase preserves the 223-dimensional ``walk_long_pass_history``
inside-kick observation/reward contract and appends the 83-dimensional CVKF
belief.  The moving phase keeps the same 306-dimensional actor contract and
introduces only incoming trajectories plus a coupled speed/closest-point
curriculum.
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..walk_long_pass_history.walk_long_pass_history_env_cfg import (
    K1WalkLongPassHistoryEnvCfg,
    K1WalkLongPassHistoryObservationsCfg,
    K1WalkLongPassHistoryPolicyCfg,
    _BALL_RADIUS,
    _STEPS_PER_ITERATION,
)
from . import mdp


_MOVING_BALL_SPEED_STAGES_MPS = (0.0, 0.5, 1.0, 1.5, 2.0)
_CLOSEST_APPROACH_RADIUS_STAGES_M = (0.0, 0.0625, 0.125, 0.1875, 0.25)
_PATH_LENGTH_MIN_M = 1.5
_PATH_LENGTH_MAX_STAGES_M = (3.0, 3.0, 3.0, 3.0, 3.0)
_NOMINAL_TTC_RANGE_S = (1.5, 5.0)
_SPEED_CURRICULUM_WARMUP_STEPS = 1000 * _STEPS_PER_ITERATION
_LOCALIZATION_DR_START_STEPS = 500 * _STEPS_PER_ITERATION
_LOCALIZATION_DR_END_STEPS = 1500 * _STEPS_PER_ITERATION
_KICK_DETECTION_WARMUP_STEPS = 5


@configclass
class K1WalkKickInsideCVKFPolicyCfg(K1WalkLongPassHistoryPolicyCfg):
    """The 223D inside observation followed by the canonical 83D CVKF belief."""

    belief = ObsTerm(
        func=mdp.CVKFBeliefObservation,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("soccer_ball"),
        },
    )


@configclass
class K1WalkKickInsideCVKFObservationsCfg(K1WalkLongPassHistoryObservationsCfg):
    policy: K1WalkKickInsideCVKFPolicyCfg = K1WalkKickInsideCVKFPolicyCfg()


@configclass
class K1WalkKickInsideCVKFStationaryEnvCfg(K1WalkLongPassHistoryEnvCfg):
    """Phase 1: the proven inside-kick task with an appended CVKF belief."""

    observations: K1WalkKickInsideCVKFObservationsCfg = (
        K1WalkKickInsideCVKFObservationsCfg()
    )
    localization_dr_start_step: int = _LOCALIZATION_DR_START_STEPS
    localization_dr_end_step: int = _LOCALIZATION_DR_END_STEPS
    localization_dr_force_full: bool = False


@configclass
class K1WalkKickInsideCVKFStationaryEnvCfg_PLAY(
    K1WalkKickInsideCVKFStationaryEnvCfg
):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKickInsideCVKFMovingEnvCfg(K1WalkKickInsideCVKFStationaryEnvCfg):
    """Phase 2: incoming balls with coupled speed and XY-near-point stages."""

    localization_dr_force_full: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_ball.func = mdp.reset_incoming_ball_near_robot
        self.events.reset_ball.params = {
            "ball_cfg": SceneEntityCfg("soccer_ball"),
            "ball_radius": _BALL_RADIUS,
            "speed_range_mps": (0.0, _MOVING_BALL_SPEED_STAGES_MPS[0]),
            "path_length_range_m": (
                _PATH_LENGTH_MIN_M,
                _PATH_LENGTH_MAX_STAGES_M[0],
            ),
            "closest_approach_radius_range_m": (
                0.0,
                _CLOSEST_APPROACH_RADIUS_STAGES_M[0],
            ),
            "approach_heading_range_rad": (-math.pi, math.pi),
            "nominal_ttc_range_s": _NOMINAL_TTC_RANGE_S,
        }
        self.curriculum.moving_ball_speed = CurrTerm(
            func=mdp.MovingBallSpeedCurriculum,
            params={
                "stages_mps": _MOVING_BALL_SPEED_STAGES_MPS,
                "spawn_distance_max_stages_m": _PATH_LENGTH_MAX_STAGES_M,
                "spawn_distance_min_m": _PATH_LENGTH_MIN_M,
                "closest_approach_radius_max_stages_m": (
                    _CLOSEST_APPROACH_RADIUS_STAGES_M
                ),
                "success_threshold": 0.80,
                "frontier_fraction": 0.75,
                "min_episodes_per_direction": 1000,
                "required_consecutive_windows": 2,
                "warmup_steps": _SPEED_CURRICULUM_WARMUP_STEPS,
                "reset_event_name": "reset_ball",
            },
        )

        kick_detection = {
            "track_ball": True,
            "physical_kick_detection": True,
            "kick_detection_foot_distance_threshold": 0.23,
            "kick_detection_min_foot_speed_towards_ball": 0.2,
            "kick_detection_velocity_change_threshold": 0.5,
            "kick_detection_warmup_steps": _KICK_DETECTION_WARMUP_STEPS,
        }
        self.terminations.kick_finished.params.update(kick_detection)
        for name, value in kick_detection.items():
            setattr(self.commands.base_velocity, name, value)


@configclass
class K1WalkKickInsideCVKFMovingEnvCfg_PLAY(K1WalkKickInsideCVKFMovingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.moving_ball_speed = None
        self.events.reset_ball.params["speed_range_mps"] = (
            0.0,
            _MOVING_BALL_SPEED_STAGES_MPS[-1],
        )
        self.events.reset_ball.params["path_length_range_m"] = (
            _PATH_LENGTH_MIN_M,
            _PATH_LENGTH_MAX_STAGES_M[-1],
        )
        self.events.reset_ball.params["closest_approach_radius_range_m"] = (
            0.0,
            _CLOSEST_APPROACH_RADIUS_STAGES_M[-1],
        )
