# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WalkKick variant with the DirectKicking CVKF/LSTM observation contract.

This task leaves the existing 55-dimensional WalkKick task untouched and uses a
separate moving-ball environment.  Its policy observation follows
the 132-dimensional ``direct_kicking_horizon_lstm_direction_only_v2`` schema:

* 47 locomotion features;
* 13 six-dimensional CVKF horizon tokens, relative velocity, and filter status
  (83 features in total);
* the two-dimensional kick target direction.

The LSTM encodes the horizon tokens in the policy model.  It is not an
environment reward and it is not recurrent across control steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from ..locomotion.rough_env_cfg import _COMMAND_THRESHOLD, _PHASE_FREQ
from ..locomotion.velocity_env_cfg import JOINT_NAMES_K1, ObservationsCfg
from ..walk_kick import mdp as walk_kick_mdp
from ..walk_kick.walk_kick_env_cfg import K1WalkKickEnvCfg, _BALL_RADIUS
from . import mdp


_MOVING_BALL_SPEED_RANGE = (0.0, 1.0)
_MOVING_BALL_SPAWN_DISTANCE_RANGE = (1.5, 3.0)
_MOVING_BALL_SPAWN_BEARING_RANGE = (-0.87266463, 0.87266463)
_MOVING_BALL_CLOSEST_APPROACH_RANGE = (-0.25, 0.25)
_BALL_FRICTION_RANGE = (0.9, 1.3)
_KICK_DETECTION_WARMUP_STEPS = 5


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
    default_mass = robot.data.default_mass[:, body_id]
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


def zero_external_wrench(env) -> torch.Tensor:
    """Represent the current task's zero continuous force/torque contract."""
    return torch.zeros((env.num_envs, 6), device=env.device)


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
        func=loco_mdp.generated_commands,
        params={"command_name": "base_velocity"},
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
    """Twenty privileged features appended to policy observations by the runner."""

    domain_randomization = ObsTerm(
        func=DomainRandomizationLatent,
        params={"asset_cfg": _trunk()},
    )
    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel, noise=Gnoise(std=0.10))
    base_height = ObsTerm(func=base_height, noise=Gnoise(std=0.02))
    external_wrench = ObsTerm(func=zero_external_wrench)
    ball_velocity = ObsTerm(func=true_ball_velocity)
    feet_position = ObsTerm(func=feet_position_xy, params={"asset_cfg": _feet()})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1WalkKickLikelihoodObservationsCfg(ObservationsCfg):
    policy: K1WalkKickLikelihoodPolicyCfg = K1WalkKickLikelihoodPolicyCfg()
    critic: K1WalkKickLikelihoodCriticCfg = K1WalkKickLikelihoodCriticCfg()


@configclass
class _K1WalkKickLikelihoodBaseEnvCfg(K1WalkKickEnvCfg):
    """Unregistered observation/model base used by the moving-ball task."""

    observations: K1WalkKickLikelihoodObservationsCfg = K1WalkKickLikelihoodObservationsCfg()


@configclass
class K1WalkKickLikelihoodEnvCfg(_K1WalkKickLikelihoodBaseEnvCfg):
    """Likelihood task with the current DirectKicking moving-ball distribution.

    The existing WalkKick task and its ball asset stay unchanged.  This task
    changes only the reset trajectory, enables pre-kick ball
    tracking/physical-kick detection, and widens the ball's ordinary Coulomb
    friction.  Ground material and restitution remain the target Isaac Lab
    task's native contract; no rolling-friction model or CVKF coupling is added.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.reset_ball.func = mdp.reset_moving_ball_trajectory
        self.events.reset_ball.params = {
            "ball_cfg": SceneEntityCfg("soccer_ball"),
            "ball_radius": _BALL_RADIUS,
            "speed_range_mps": _MOVING_BALL_SPEED_RANGE,
            "incoming_probability": 0.5,
            "incoming_spawn_distance_range_m": _MOVING_BALL_SPAWN_DISTANCE_RANGE,
            "outgoing_spawn_distance_range_m": _MOVING_BALL_SPAWN_DISTANCE_RANGE,
            "closest_approach_offset_range_m": _MOVING_BALL_CLOSEST_APPROACH_RANGE,
            "spawn_bearing_range_rad": _MOVING_BALL_SPAWN_BEARING_RANGE,
        }
        self.events.ball_friction = EventTerm(
            func=mdp.RandomizeBallFriction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "friction_range": _BALL_FRICTION_RANGE,
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
