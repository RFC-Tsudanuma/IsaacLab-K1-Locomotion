# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WalkKick variant with the DirectKicking CVKF/LSTM observation contract.

This task leaves the existing 55-dimensional WalkKick task untouched and uses a
separate staged environment family.  Its policy observation follows the
132-dimensional ``direct_kicking_horizon_lstm_global_target_direction_v3`` schema:

* 47 locomotion features;
* 13 six-dimensional CVKF horizon tokens, relative velocity, and filter status
  (83 features in total);
* the two-dimensional, estimated ball-to-global-target direction.

The LSTM encodes the horizon tokens in the policy model.  It is not an
environment reward and it is not recurrent across control steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from ..locomotion.rough_env_cfg import _COMMAND_THRESHOLD, _PHASE_FREQ
from ..locomotion.velocity_env_cfg import JOINT_NAMES_K1, ObservationsCfg
from ..walk_kick import mdp as walk_kick_mdp
from ..walk_kick.walk_kick_env_cfg import (
    K1WalkKickEnvCfg,
    _BALL_RADIUS,
    _CTRL_DT,
    _KICK_STATE_PARAMS,
    _SIGMA_DIRECTION,
)
from . import mdp


_MOVING_BALL_SPEED_STAGES = (0.0, 1.0, 2.0, 4.0)
_MOVING_BALL_SPEED_RANGE = (0.0, _MOVING_BALL_SPEED_STAGES[-1])
_MOVING_BALL_SPAWN_DISTANCE_MIN_M = 1.5
_MOVING_BALL_SPAWN_DISTANCE_MAX_STAGES_M = (3.0, 3.0, 5.0, 8.0)
_MOVING_BALL_SPAWN_BEARING_RANGE = (-0.87266463, 0.87266463)
_MOVING_BALL_CLOSEST_APPROACH_RANGE = (-0.25, 0.25)
_MOVING_BALL_NOMINAL_TTC_RANGE_S = (1.5, 5.0)
_BALL_FRICTION_RANGE = (0.9, 1.3)
_BALL_MASS_KG = 0.45
_BALL_ROLLING_DECAY_RATE_RANGE_S_INV = (0.12, 0.18)
_BALL_GROUND_CONTACT_TOLERANCE_M = 0.01
_KICK_TARGET_HEADING_RANGE = (-math.pi / 2.0, math.pi / 2.0)
_KICK_DETECTION_WARMUP_STEPS = 5
_STEPS_PER_ITERATION = 48
_BALL_DIRECTION_PENALTY_PER_KICK = 1.5
_BALL_DIRECTION_PENALTY_RAMP_START_ITERATION = 500
_BALL_DIRECTION_PENALTY_RAMP_END_ITERATION = 1000
# Stage 2 first acquires a stationary-ball kick with measured nominal sensor
# timing and zero Localization bias.  Once the inherited 500-iteration
# kick-reward ramp has finished, episode-sampled timing/bias bounds widen to
# their full values over 1,000 more iterations.  This deterministic boundary is
# intentionally a named tuning choice.
_LOCALIZATION_DR_START_STEPS = 500 * _STEPS_PER_ITERATION
_LOCALIZATION_DR_END_STEPS = 1500 * _STEPS_PER_ITERATION
# Keep the ball stationary for the first 1,000 learning iterations.  The
# likelihood runner collects 48 control steps per learning iteration.
_BALL_SPEED_CURRICULUM_WARMUP_STEPS = 1000 * _STEPS_PER_ITERATION


def _leg_joints() -> SceneEntityCfg:
    """Return the 12 policy joints in the action/checkpoint order."""
    return SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)


def _trunk() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names="Trunk")


def _feet() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        body_names=["left_foot_link", "right_foot_link"],
        preserve_order=True,
    )


def gait_phase_cos_sin(
    env,
    phase_freq: float = 1.6,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
) -> torch.Tensor:
    """Return the current WalkKick gait phase in the source model's (cos, sin) order."""
    phase_sin_cos = walk_kick_mdp.gait_phase_sincos(
        env,
        phase_freq=phase_freq,
        command_name=command_name,
        cmd_threshold=cmd_threshold,
    )
    return phase_sin_cos[:, [1, 0]]


def estimated_base_velocity_command(
    env,
    command_name: str = "kick_direction",
    max_vel: float = 1.0,
    max_ang_vel: float = 1.0,
    r_stance: float = 0.25,
    alpha: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """Late-bind the likelihood estimator command for config-only importers."""
    return mdp.observed_base_velocity_command(
        env,
        command_name=command_name,
        max_vel=max_vel,
        max_ang_vel=max_ang_vel,
        r_stance=r_stance,
        alpha=alpha,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
    )


def estimated_gait_phase_cos_sin(
    env,
    phase_freq: float = 1.6,
    cmd_threshold: float = 0.05,
    command_name: str = "kick_direction",
    max_vel: float = 1.0,
    max_ang_vel: float = 1.0,
    r_stance: float = 0.25,
    alpha: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """Gate gait phase with the same estimated command visible to the actor."""
    visible_command = estimated_base_velocity_command(
        env,
        command_name=command_name,
        max_vel=max_vel,
        max_ang_vel=max_ang_vel,
        r_stance=r_stance,
        alpha=alpha,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
    )
    phase_sin_cos = walk_kick_mdp.gait_phase_sincos(
        env,
        phase_freq=phase_freq,
        cmd_threshold=cmd_threshold,
        gate_command=visible_command,
    )
    return phase_sin_cos[:, [1, 0]]


def privileged_true_kick_geometry(
    env,
    command_name: str = "kick_direction",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """Late-bind privileged geometry while keeping lightweight cfg imports valid."""
    return mdp.true_kick_geometry(
        env,
        command_name=command_name,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
    )


def _compute_domain_randomization_latent(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> torch.Tensor:
    """Expose the target task's sampled trunk COM/mass values as four unit latents.

    The source checkpoint observes the random draw rather than the resulting SI
    value.  The target task uses different physical ranges, so this reconstructs
    the same unit-uniform contract without changing its existing randomization.
    """
    robot = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]

    # K1_locomotion.urdf Trunk inertial origin.  Isaac Lab does not retain a
    # public default-COM tensor, while the default mass is retained explicitly.
    default_com = torch.tensor(
        (-0.0043392, -0.00065534, 0.065686),
        device=env.device,
        dtype=robot.data.body_com_pos_b.dtype,
    )
    com = robot.data.body_com_pos_b[:, body_id, :]
    com_offset = com - default_com
    com_low = torch.tensor((-0.05, -0.05, -0.01), device=env.device, dtype=com.dtype)
    com_span = torch.tensor((0.10, 0.10, 0.02), device=env.device, dtype=com.dtype)
    com_latent = (com_offset - com_low) / com_span

    masses = robot.root_physx_view.get_masses().to(device=env.device)
    mass = masses[:, body_id]
    default_mass = robot.data.default_mass[:, body_id].to(
        device=mass.device,
        dtype=mass.dtype,
    )
    mass_latent = ((mass - default_mass) + 1.5) / 3.0
    return torch.cat((com_latent, mass_latent.unsqueeze(-1)), dim=-1)


class DomainRandomizationLatent(ManagerTermBase):
    """Cache startup COM/mass draws after startup events have run."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._cached: torch.Tensor | None = None

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
    ) -> torch.Tensor:
        if self._cached is None:
            self._cached = _compute_domain_randomization_latent(env, asset_cfg)
        return self._cached

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Startup events run after ObservationManager's shape probe.  Clearing
        # here prevents that pre-startup value from becoming the policy latent.
        del env_ids
        self._cached = None


def base_height(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root-link height above the local flat-terrain origin."""
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def true_ball_velocity(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """Return privileged world-frame horizontal ball velocity."""
    return env.scene[ball_cfg.name].data.root_lin_vel_w[:, :2]


def feet_position_xy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["left_foot_link", "right_foot_link"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Return privileged left/right foot XY positions in the local world frame."""
    robot = env.scene[asset_cfg.name]
    positions = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    positions = positions - env.scene.env_origins[:, None, :2]
    return positions.reshape(env.num_envs, 4)


@configclass
class K1WalkKickLikelihoodPolicyCfg(ObsGroup):
    """Actor observation in the exact 132-dimensional checkpoint order."""

    # 47-feature locomotion prefix.
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, noise=Gnoise(std=0.01))
    base_ang_vel = ObsTerm(
        func=mdp.observed_base_ang_vel,
        params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("soccer_ball")},
    )
    velocity_commands = ObsTerm(
        func=estimated_base_velocity_command,
        params={
            "command_name": "kick_direction",
            "max_vel": 1.0,
            "max_ang_vel": 1.0,
            "r_stance": 0.25,
            "alpha": 0.5,
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("soccer_ball"),
        },
    )
    gait_phase = ObsTerm(
        func=gait_phase_cos_sin,
        params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD},
    )
    joint_pos = ObsTerm(
        func=loco_mdp.joint_pos_rel,
        noise=Gnoise(std=0.01),
        params={"asset_cfg": _leg_joints()},
    )
    joint_vel = ObsTerm(
        func=loco_mdp.joint_vel_rel,
        noise=Gnoise(std=0.1),
        scale=0.1,
        params={"asset_cfg": _leg_joints()},
    )
    previous_action = ObsTerm(func=loco_mdp.last_action)

    # 13 * 6 horizon tokens + relative velocity (2) + filter status (3).
    belief = ObsTerm(
        func=mdp.CVKFBeliefObservation,
        params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("soccer_ball")},
    )
    target_direction = ObsTerm(
        func=mdp.observed_kick_direction,
        params={
            "command_name": "kick_direction",
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("soccer_ball"),
        },
    )

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1WalkKickLikelihoodCriticCfg(ObsGroup):
    """Twenty privileged truth features appended to the actor observation."""

    domain_randomization = ObsTerm(
        func=DomainRandomizationLatent,
        params={"asset_cfg": _trunk()},
    )
    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
    base_height = ObsTerm(func=base_height)
    kick_geometry = ObsTerm(
        func=privileged_true_kick_geometry,
        params={
            "command_name": "kick_direction",
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("soccer_ball"),
        },
    )
    ball_velocity = ObsTerm(func=true_ball_velocity)
    feet_position = ObsTerm(func=feet_position_xy, params={"asset_cfg": _feet()})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1WalkKickLikelihoodObservationsCfg(ObservationsCfg):
    policy: K1WalkKickLikelihoodPolicyCfg = K1WalkKickLikelihoodPolicyCfg()
    critic: K1WalkKickLikelihoodCriticCfg = K1WalkKickLikelihoodCriticCfg()


@configclass
class _K1WalkKickLikelihoodBaseEnvCfg(K1WalkKickEnvCfg):
    """Shared 132D model and global-target command contract for all stages."""

    observations: K1WalkKickLikelihoodObservationsCfg = K1WalkKickLikelihoodObservationsCfg()
    localization_dr_start_step: int = _LOCALIZATION_DR_START_STEPS
    localization_dr_end_step: int = _LOCALIZATION_DR_END_STEPS
    localization_dr_force_full: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        previous = self.commands.kick_direction
        self.commands.kick_direction = mdp.GlobalTargetKickCommandCfg(
            asset_name=previous.asset_name,
            resampling_time_range=previous.resampling_time_range,
            rel_standing_envs=previous.rel_standing_envs,
            rel_heading_envs=previous.rel_heading_envs,
            heading_command=previous.heading_command,
            heading_control_stiffness=previous.heading_control_stiffness,
            debug_vis=previous.debug_vis,
            ranges=previous.ranges,
            target_speed_range=previous.target_speed_range,
            low_speed_threshold=previous.low_speed_threshold,
            target_distance_range=(5.0, 12.0),
            ball_asset_name="soccer_ball",
        )
        self.commands.kick_direction.ranges.heading = _KICK_TARGET_HEADING_RANGE


def _freeze_kick_reward_curricula_at_final(cfg) -> None:
    """Keep Stage 2's converged reward contract when Stage 3 starts at step 0."""
    curriculum_names = {
        "track_lin_vel_xy_exp": "track_lin_vel_weight",
        "track_ang_vel_z_exp": "track_ang_vel_weight",
        "kick_direction": "kick_direction_weight",
        "ball_direction_penalty": "ball_direction_penalty_weight",
        "kick_velocity_scaled": "kick_velocity_scaled_weight",
        "kick_velocity_strong": "kick_velocity_strong_weight",
        "walk_speed": "walk_speed_weight",
        "approach_penalty": "approach_penalty_weight",
        "kick_pose_overshoot": "kick_pose_overshoot_weight",
    }
    for reward_name, curriculum_name in curriculum_names.items():
        curriculum = getattr(cfg.curriculum, curriculum_name, None)
        if curriculum is None:
            continue
        getattr(cfg.rewards, reward_name).weight = float(curriculum.params["end_weight"])
        setattr(cfg.curriculum, curriculum_name, None)


def _align_kick_curricula_with_runner(cfg) -> None:
    """Use the likelihood runner's actual rollout length for Stage 2 ramps."""
    for curriculum_name in (
        "track_lin_vel_weight",
        "track_ang_vel_weight",
        "kick_direction_weight",
        "ball_direction_penalty_weight",
        "kick_velocity_scaled_weight",
        "kick_velocity_strong_weight",
        "walk_speed_weight",
        "approach_penalty_weight",
        "kick_pose_overshoot_weight",
    ):
        curriculum = getattr(cfg.curriculum, curriculum_name, None)
        if curriculum is not None:
            curriculum.params["steps_per_iteration"] = _STEPS_PER_ITERATION


@configclass
class K1WalkKickLikelihoodWalkPhaseEnvCfg(_K1WalkKickLikelihoodBaseEnvCfg):
    """Stage 1: DirectKicking-compatible locomotion without a ball or kick."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.follow_ball = False
        self.commands.base_velocity.kick_direction_command_name = None
        self.commands.kick_direction = None

        # Preserve exactly 132 actor + 20 privileged values.  The explicit
        # velocity-command prefix remains observable; ball/target slots are
        # neutral because no estimator or target exists in this stage.
        self.observations.policy.base_ang_vel.func = loco_mdp.base_ang_vel
        self.observations.policy.base_ang_vel.noise = Gnoise(std=0.15)
        self.observations.policy.base_ang_vel.params = {}
        self.observations.policy.velocity_commands.func = loco_mdp.generated_commands
        self.observations.policy.velocity_commands.params = {"command_name": "base_velocity"}
        self.observations.policy.belief.func = walk_kick_mdp.zero_obs
        self.observations.policy.belief.params = {"dim": 83}
        self.observations.policy.target_direction.func = walk_kick_mdp.zero_obs
        self.observations.policy.target_direction.params = {"dim": 2}
        self.observations.critic.kick_geometry.func = walk_kick_mdp.zero_obs
        self.observations.critic.kick_geometry.params = {"dim": 6}
        self.observations.critic.ball_velocity.func = walk_kick_mdp.zero_obs
        self.observations.critic.ball_velocity.params = {"dim": 2}

        self.scene.soccer_ball = None
        self.scene.contact_balls_left = None
        self.scene.contact_balls_right = None
        self.events.reset_ball = None

        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        for term_name in (
            "kick_direction",
            "kick_velocity_scaled",
            "kick_velocity_strong",
            "walk_speed",
            "approach_penalty",
            "kick_pose_overshoot",
        ):
            setattr(self.rewards, term_name, None)
            setattr(self.curriculum, f"{term_name}_weight", None)
        self.curriculum.track_lin_vel_weight = None
        self.curriculum.track_ang_vel_weight = None
        self.terminations.kick_finished = None
        self.episode_length_s = 20.0


@configclass
class K1WalkKickLikelihoodWalkPhaseEnvCfg_PLAY(K1WalkKickLikelihoodWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKickLikelihoodStationaryEnvCfg(_K1WalkKickLikelihoodBaseEnvCfg):
    """Stage 2: stationary-ball kick with nominal-to-full sensor DR."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.ball_rolling_resistance = EventTerm(
            func=mdp.ApplyBallRollingResistance,
            mode="interval",
            interval_range_s=(_CTRL_DT, _CTRL_DT),
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "ball_radius": _BALL_RADIUS,
                "ball_mass": _BALL_MASS_KG,
                "decay_rate_range_s_inv": _BALL_ROLLING_DECAY_RATE_RANGE_S_INV,
                "contact_tolerance_m": _BALL_GROUND_CONTACT_TOLERANCE_M,
            },
        )
        self.rewards.ball_direction_penalty = RewTerm(
            func=walk_kick_mdp.ball_direction_penalty,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION},
        )
        self.curriculum.ball_direction_penalty_weight = CurrTerm(
            func=walk_kick_mdp.linear_reward_weight,
            params={
                "term_name": "ball_direction_penalty",
                "start_weight": 0.0,
                "end_weight": -_BALL_DIRECTION_PENALTY_PER_KICK / _CTRL_DT,
                "start_step": _BALL_DIRECTION_PENALTY_RAMP_START_ITERATION,
                "end_step": _BALL_DIRECTION_PENALTY_RAMP_END_ITERATION,
                "steps_per_iteration": _STEPS_PER_ITERATION,
            },
        )
        _align_kick_curricula_with_runner(self)
        base_velocity = self.commands.base_velocity
        self.observations.policy.gait_phase.func = estimated_gait_phase_cos_sin
        self.observations.policy.gait_phase.params = {
            "phase_freq": _PHASE_FREQ,
            "cmd_threshold": _COMMAND_THRESHOLD,
            "command_name": base_velocity.kick_direction_command_name,
            "max_vel": base_velocity.max_vel,
            "max_ang_vel": base_velocity.max_ang_vel,
            "r_stance": base_velocity.r_stance,
            "alpha": base_velocity.alpha,
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("soccer_ball"),
        }


@configclass
class K1WalkKickLikelihoodStationaryEnvCfg_PLAY(K1WalkKickLikelihoodStationaryEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKickLikelihoodEnvCfg(K1WalkKickLikelihoodStationaryEnvCfg):
    """Stage 3: full estimator DR plus an incoming-ball TTC curriculum.

    The existing WalkKick task and its ball asset stay unchanged.  This task
    changes the reset trajectory, enables pre-kick ball tracking/physical-kick
    detection, preserves the widened ordinary Coulomb-friction range, and
    promotes paired spawn-distance/speed caps from stationary to 8 m / 4 m/s.
    TTC controls the nominal approach timing; the inherited pseudo rolling
    resistance remains independent of the CVKF.
    """

    localization_dr_force_full: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _freeze_kick_reward_curricula_at_final(self)

        self.events.reset_ball.func = mdp.reset_moving_ball_trajectory
        self.events.reset_ball.params = {
            "ball_cfg": SceneEntityCfg("soccer_ball"),
            "ball_radius": _BALL_RADIUS,
            "speed_range_mps": (0.0, _MOVING_BALL_SPEED_STAGES[0]),
            "incoming_probability": 1.0,
            "incoming_spawn_distance_range_m": (
                _MOVING_BALL_SPAWN_DISTANCE_MIN_M,
                _MOVING_BALL_SPAWN_DISTANCE_MAX_STAGES_M[0],
            ),
            "outgoing_spawn_distance_range_m": (
                _MOVING_BALL_SPAWN_DISTANCE_MIN_M,
                _MOVING_BALL_SPAWN_DISTANCE_MAX_STAGES_M[0],
            ),
            "closest_approach_offset_range_m": _MOVING_BALL_CLOSEST_APPROACH_RANGE,
            "spawn_bearing_range_rad": _MOVING_BALL_SPAWN_BEARING_RANGE,
            "nominal_ttc_range_s": _MOVING_BALL_NOMINAL_TTC_RANGE_S,
        }
        self.events.ball_friction = EventTerm(
            func=mdp.RandomizeBallFriction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "friction_range": _BALL_FRICTION_RANGE,
            },
        )
        self.curriculum.moving_ball_speed = CurrTerm(
            func=mdp.MovingBallSpeedCurriculum,
            params={
                "stages_mps": _MOVING_BALL_SPEED_STAGES,
                "spawn_distance_max_stages_m": _MOVING_BALL_SPAWN_DISTANCE_MAX_STAGES_M,
                "spawn_distance_min_m": _MOVING_BALL_SPAWN_DISTANCE_MIN_M,
                "success_threshold": 0.80,
                "frontier_fraction": 0.75,
                "min_episodes_per_direction": 1000,
                "required_consecutive_windows": 2,
                "warmup_steps": _BALL_SPEED_CURRICULUM_WARMUP_STEPS,
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
class K1WalkKickLikelihoodEnvCfg_PLAY(K1WalkKickLikelihoodEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.moving_ball_speed = None
        self.events.reset_ball.params["speed_range_mps"] = _MOVING_BALL_SPEED_RANGE
        final_distance_range = (
            _MOVING_BALL_SPAWN_DISTANCE_MIN_M,
            _MOVING_BALL_SPAWN_DISTANCE_MAX_STAGES_M[-1],
        )
        self.events.reset_ball.params[
            "incoming_spawn_distance_range_m"
        ] = final_distance_range
        self.events.reset_ball.params[
            "outgoing_spawn_distance_range_m"
        ] = final_distance_range
