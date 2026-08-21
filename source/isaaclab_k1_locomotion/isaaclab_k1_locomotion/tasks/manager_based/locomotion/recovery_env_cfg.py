# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

"""K1 RECOVERY task — regain balance from a near-fall state instead of falling.

Motivation (FDM fall-head project, 2026-07-26): with the learned fall head driving a
"predict -> stop" intervention, 59-62 % of the REMAINING falls happen while the intervention is
already active — the robot was warned, was stopped, and fell anyway (PRISM's "unstoppable state").
Stopping is not a strong enough response, so the alarm needs a policy that actively recovers.

Design constraints that make this policy a DROP-IN replacement for the walk policy at deployment
(the FDM env loads a single TorchScript file into one action term, see
``fdm/env_cfg/robot_cfg.py: cfg.actions.velocity_cmd.low_level_policy_file``):

* the observation group is inherited from :class:`K1FlatEnvCfg` UNCHANGED (49-dim, incl. the
  gait-phase and velocity-command entries) — otherwise the two policies cannot be swapped at runtime;
* the action term is unchanged (12 leg joint position offsets, scale 0.5, use_default_offset);
* the velocity command is pinned to ZERO: the recovery policy's job is to stay upright and come to
  rest, and at deployment the alarm hands it a zero command, so training must match that.

What differs from walking: the episode starts in a near-fall state (tilted base + a hard push),
episodes are short, and the reward pays for staying upright / coming to rest rather than for
tracking a velocity.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import K1FlatEnvCfg


@configclass
class K1RecoveryEnvCfg(K1FlatEnvCfg):
    """Recover from a near-fall state (flat ground)."""

    def __post_init__(self):
        super().__post_init__()

        # -- episode: recovery is a short-horizon task (a fall resolves in ~1-2 s)
        self.episode_length_s = 4.0

        # -- command: always stand still. The obs keeps the 3 command entries (shape must match the
        # walk policy) but they are pinned to zero, which is what the alarm feeds at deployment.
        # The inherited flat curriculum WIDENS the command ranges as training progresses (and the
        # smoke run showed it doing exactly that: lin_vel_x_max 0.6), which would undo the pinning —
        # so the command curricula are removed here.
        self.curriculum.lin_vel_command = None
        self.curriculum.command_resampling_time_range = None
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # -- initial state: the NEAR-FALL distribution this policy exists for. A tilted base with
        # angular rate and a shove, so the episode starts where the walk policy is already losing it.
        # Ranges are deliberately wide: the alarm fires ~1.7 s before the fall, so the states handed
        # over span "slightly off balance" to "almost gone".
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (0.0, 0.02),
                    # calibrated on the smoke run: at +-0.45 rad tilt with +-1.5 rad/s rates, 97 % of
                    # episodes ended in a fall within ~0.5 s — an unrecoverable distribution teaches
                    # nothing. These ranges leave a meaningful fraction of survivable starts, which
                    # is what PPO needs to find the recovery behaviour at all.
                    "roll": (-0.10, 0.10),
                    "pitch": (-0.10, 0.10),
                    "yaw": (-math.pi, math.pi),
                },
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.3, 0.3),
                    "z": (-0.1, 0.1),
                    "roll": (-0.4, 0.4),
                    "pitch": (-0.4, 0.4),
                    "yaw": (-0.4, 0.4),
                },
            },
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.9, 1.1), "velocity_range": (-0.5, 0.5)},
        )
        # The near-fall states this policy exists for are created DURING the episode by hard pushes,
        # not by the reset: starting every episode already-falling gave 98 % unrecoverable episodes
        # (smoke run) and no learning signal. Starting near-nominal and shoving hard means the policy
        # meets the same states with a survivable path behind them.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.8, 1.6),
            params={
                "velocity_range": {
                    "x": (-1.4, 1.4),
                    "y": (-1.1, 1.1),
                    "roll": (-1.6, 1.6),
                    "pitch": (-1.6, 1.6),
                }
            },
        )

        # -- rewards: stay upright, come to rest, keep the base at standing height.
        # The velocity-tracking terms are KEPT: with the command pinned to zero they already are the
        # "come to rest" reward (exp(-|v|^2/std^2) is maximal at v = 0), so no extra penalty term is
        # needed — and isaaclab has no lin_vel_xy_l2 / ang_vel_z_l2 to add anyway.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        if getattr(self.rewards, "track_lin_vel_xy_coarse", None) is not None:
            # the coarse (wide-std) companion keeps a gradient while the robot is still moving fast
            self.rewards.track_lin_vel_xy_coarse.weight = 1.0
        if getattr(self.rewards, "feet_air_time", None) is not None:
            self.rewards.feet_air_time.weight = 0.0  # stepping is a means here, not the goal

        # survival is the task: every step upright pays
        self.rewards.is_alive = RewTerm(func=mdp.is_alive, weight=2.0)
        # keep the torso level (this is the term the fall label keys on)
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
        # do not sink: K1 stands at ~0.5 m, the fall label latches at 0.28 m
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.50}
        )
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.3
        # smoothness / actuation limits (a recovery that destroys the gearbox is not a recovery)
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

        # -- terminations: torso contact = the fall we are trying to avoid
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="Trunk"
        )
        # The inherited base_height termination (root below 0.35 m) fires on EVERY episode of the
        # near-fall distribution — a crouch deep enough to catch yourself is not a fall, and killing
        # the episode there removes exactly the states this policy must learn to survive. Lower it to
        # the height the FDM fall label itself latches at (0.28 m), so "terminated" means "fell".
        self.terminations.base_height.params["minimum_height"] = 0.28

        # -- no terrain curriculum on flat ground (inherited), and no phase-freq randomisation need
        self.curriculum.terrain_levels = None


@configclass
class K1RecoveryEnvCfgDiagNoPush(K1RecoveryEnvCfg):
    """Diagnostic only: identical but with the push event removed (isolates reset vs push)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}


class K1RecoveryEnvCfg_PLAY(K1RecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class K1RecoveryRoughEnvCfg(K1RecoveryEnvCfg):
    """Recovery trained on the SAME rough terrain families the FDM is evaluated on.

    Why: the flat-ground recovery policy cut falls no more than a plain stop did, and 70 % of the
    remaining falls happened while it was already driving the joints (stairs_ramp A/B, 2026-07-26).
    A balance-recovery reflex learned on a plane cannot fix a foot that landed on a stair edge — the
    failures in deployment are terrain-induced, so the recovery policy has to see terrain.

    The observation group stays untouched (no height scan, 49-dim, identical to the walk policy) so
    the policy remains hot-swappable; it must learn terrain-robust reflexes from proprioception alone.
    """

    def __post_init__(self):
        super().__post_init__()

        # the FDM's mixed training terrain (eval families, K1-scaled). fdm and this package live in
        # the same python env; the import is deliberately local so the flat task never needs fdm.
        import fdm.env_cfg.terrain_cfg as fdm_terrain_cfg

        gen = fdm_terrain_cfg.build_fall_train_mix_cfg()
        gen = fdm_terrain_cfg.scale_eval_terrain(gen, 0.15)  # K1 scale, as in every FDM run
        gen.curriculum = False  # sample difficulty uniformly (no terrain-level curriculum term here)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = gen
        self.scene.terrain.max_init_terrain_level = None
