# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draft env cfg for hierarchical dribble training.

Design:
- Robot / terrain / actions are inherited from ``K1FlatEnvCfg``.
- The scene adds a soccer ball + two foot-vs-ball contact sensors, copied from
  the kick env (``kick_env_cfg.py``).
- Commands: a ``kick_direction`` command holds a world-frame xy unit vector
  representing where the high-level policy should drive the ball. The original
  ``base_velocity`` command from FlatEnv is kept so the ``low_level`` obs group
  retains its ``velocity_commands`` slot (whose value is overwritten by the
  hierarchical wrapper before reaching the frozen policy anyway).
- Observations:

    * ``policy`` group  = FlatEnv K1PolicyCfg - ``velocity_commands`` + ``ball_pos_rel`` + ``kick_direction_b``
    * ``critic`` group  = FlatEnv K1CriticCfg - ``velocity_commands`` + ``ball_pos_rel`` + ``ball_vel`` + ``kick_direction_b``
    * ``low_level`` group = FlatEnv K1PolicyCfg as-is                   (for frozen)

  The ``low_level`` group is kept structurally identical to what the frozen
  walking policy was trained on so it can be consumed directly by the frozen
  net. ``scripts/rsl_rl/train_dribble.py`` should be launched with
  ``--low_level_obs_group low_level``.

- Rewards (in addition to inherited FlatEnv rewards):
    * ``ball_velocity_along_kick`` — ball velocity projected onto the world-frame kick direction
    * ``ball_speed`` — magnitude of the ball's world-frame xy velocity
    * ``robot_velocity_toward_ball`` — small shaping reward: robot velocity component
      toward the ball (zeroed out when very close)
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import K1FlatEnvCfg
from .rough_env_cfg import K1CriticCfg, K1ObservationsCfg, K1PolicyCfg
from .velocity_env_cfg import CommandsCfg, MySceneCfg
from .mdp.commands import KickDirectionCommandCfg
from .mdp.observations import ball_pos_rel, ball_vel, kick_direction_b
from .mdp.rewards import (
    action_smoothness_l2,
    ball_speed,
    ball_velocity_along_kick,
    com_jerk_l2,
    robot_velocity_toward_ball,
)


@configclass
class K1DribbleSceneCfg(MySceneCfg):
    """Scene: FlatEnv scene + soccer ball + foot-vs-ball contact sensors."""

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,  # 11cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),  # 450g
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.7,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.11)),
    )

    contact_balls_right: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    contact_balls_left: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        update_period=0.0,
        history_length=1,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )


@configclass
class K1DribblePolicyCfg(K1PolicyCfg):
    """High-level policy obs: FlatEnv K1PolicyCfg + ball position + kick direction (in base frame)."""

    ball_pos_rel = ObsTerm(func=ball_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
    kick_direction_b = ObsTerm(func=kick_direction_b, params={"command_name": "kick_direction"})


@configclass
class K1DribbleCriticCfg(K1CriticCfg):
    """High-level critic obs: FlatEnv K1CriticCfg + ball pos/vel (privileged) + kick direction."""

    ball_pos_rel = ObsTerm(func=ball_pos_rel)
    ball_vel = ObsTerm(func=ball_vel)
    kick_direction_b = ObsTerm(func=kick_direction_b, params={"command_name": "kick_direction"})


@configclass
class K1DribbleObservationsCfg(K1ObservationsCfg):
    """Dribble obs groups.

    ``low_level`` is structurally identical to the FlatEnv ``K1PolicyCfg`` that
    the frozen walking policy was trained on. The hierarchical wrapper reads
    this group and overwrites its ``velocity_commands`` slice with the
    high-level action before feeding the frozen policy.
    """

    policy: K1DribblePolicyCfg = K1DribblePolicyCfg()
    critic: K1DribbleCriticCfg = K1DribbleCriticCfg()
    low_level: K1PolicyCfg = K1PolicyCfg()


@configclass
class K1DribbleCommandsCfg(CommandsCfg):
    """FlatEnv の ``base_velocity`` に加えて、ワールド座標系のキック方向コマンドを追加。"""

    kick_direction = KickDirectionCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
    )


@configclass
class K1DribbleEnvCfg(K1FlatEnvCfg):
    """K1FlatEnv + a soccer ball, used as the dribble training env."""

    scene: K1DribbleSceneCfg = K1DribbleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1DribbleObservationsCfg = K1DribbleObservationsCfg()
    commands: K1DribbleCommandsCfg = K1DribbleCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 高レベルの観測には FlatEnv の ``velocity_commands`` 項は不要なので外す。
        # ※ ``low_level`` 観測グループには元のまま残しておく (frozen の入力スロットが必要)。
        self.observations.policy.velocity_commands = None
        self.observations.critic.velocity_commands = None

        # FlatEnv 由来のカリキュラム (base_velocity 追従/push_robot 強化) は
        # 高レベルタスクには不要なので全て無効化。
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.push_robot_stage1 = None

        # FlatEnv/RoughEnv 由来の報酬項は全て無効化し、dribble 側で定義する項のみ残す。
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_phase = None
        self.rewards.feet_air_time = None
        self.rewards.feet_slide = None
        self.rewards.dof_pos_limits_ankle = None
        self.rewards.joint_deviation_hip = None
        self.rewards.base_height_penalty = None
        self.rewards.feet_close_penalty = None
        self.rewards.feet_parallel_to_ground = None
        self.rewards.foot_clearance_ji_pen = None
        self.rewards.dof_vel_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_pos_limits = None
        self.rewards.feet_landing_impact = None

        # Spawn the ball in a random position around the robot on reset.
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "pose_range": {
                    "x": (0.15, 1.5),
                    "y": (-1.5, 1.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("soccer_ball"),
            },
        )

        # キック方向 (ワールド座標) に沿ったボール速度成分が大きいほど高い報酬。
        self.rewards.ball_velocity_along_kick = RewTerm(
            func=ball_velocity_along_kick,
            weight=5.0,
            params={"command_name": "kick_direction", "max_speed": 3.0},
        )
        # ボール速度の大きさ (方向問わず)。max_speed で正規化、上限 1.0。
        self.rewards.ball_speed = RewTerm(
            func=ball_speed,
            weight=1.5,
            params={"max_speed": 3.0},
        )
        # Shaping: ロボットがボールに向かって進んでいる成分。重みは小さめ。
        self.rewards.robot_velocity_toward_ball = RewTerm(
            func=robot_velocity_toward_ball,
            weight=0.5,
            params={"max_speed": 1.0, "min_distance": 0.25},
        )
        
        # -----------------------ペナルティ-----------------------
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
        self.rewards.action_smoothness_l2 = RewTerm(
            func=action_smoothness_l2,
            weight=-0.2,
        )
        self.rewards.ang_vel_xy_l2 = RewTerm(
            func=mdp.ang_vel_xy_l2,
            weight=-0.2,
        )
        # ロボット root body COM の jerk (加速度の時間微分) ペナルティ。
        # 値域が大きくなりやすいので重みは非常に小さめから始める。
        self.rewards.com_jerk_l2 = RewTerm(
            func=com_jerk_l2,
            weight=-1.0e-6,
        )


@configclass
class K1DribbleEnvCfg_PLAY(K1DribbleEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
