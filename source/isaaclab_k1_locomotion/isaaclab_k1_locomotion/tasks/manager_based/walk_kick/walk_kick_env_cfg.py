# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークキック環境。

K1FlatEnvCfg を継承し、ボール蹴りタスク向けの観測・報酬・コマンドを追加する。

追加観測:
  ball_pos_rel     (3次元): ロボットフレームでのボール位置（蹴り込み距離）
  ball_vel_rel     (3次元): ロボットフレームでのボール速度
  kick_dir_sincos  (2次元): 蹴り方向コマンドを (sin θ, cos θ) で表現

追加コマンド:
  kick_direction: エピソードごとにランダムな蹴り方向 θ ∈ [-π, π] をサンプリング

報酬:
  + touch_ball         : 足がボールに触れたとき
  + ball_distance      : ボールへの接近
  + kick_ball_velocity : ボールが蹴り方向コマンドに沿って動いているとき
"""

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass

from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from . import mdp

_BALL_RADIUS = 0.11


@configclass
class K1WalkKickEnvCfg(K1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # Scene: サッカーボール + 足-ボール接触センサー
        # ------------------------------------------------------------------ #
        self.scene.soccer_ball = RigidObjectCfg(
            prim_path="/World/envs/env_.*/SoccerBall",
            spawn=sim_utils.SphereCfg(
                radius=_BALL_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
                mass_props=MassPropertiesCfg(mass=0.45),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0),
                    metallic=0.0,
                    roughness=0.7,
                ),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, _BALL_RADIUS)),
        )
        self.scene.contact_balls_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
        )
        self.scene.contact_balls_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
            update_period=0.0,
            history_length=1,
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
        )

        # ------------------------------------------------------------------ #
        # Commands: 蹴り方向コマンドを追加（既存の base_velocity はそのまま維持）
        # ------------------------------------------------------------------ #
        self.commands.kick_direction = mdp.KickDirectionCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=False,
            debug_vis=False,
            ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(-math.pi, math.pi),  # 蹴り方向のサンプリング範囲
            ),
        )

        # ------------------------------------------------------------------ #
        # Observations: ボール関連の観測を追加
        # ------------------------------------------------------------------ #
        # ボール位置（ロボットフレーム）= 蹴り込み距離
        self.observations.policy.ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)
        # ボール速度（ロボットフレーム）
        self.observations.policy.ball_vel_rel = ObsTerm(func=mdp.ball_vel_rel)
        # 蹴り方向コマンド (sin θ, cos θ)
        self.observations.policy.kick_dir_sincos = ObsTerm(
            func=mdp.kick_dir_sincos,
            params={"command_name": "kick_direction"},
        )

        # ------------------------------------------------------------------ #
        # Rewards: ボール接触・接近・方向蹴り
        # ------------------------------------------------------------------ #
        self.rewards.touch_ball = RewTerm(func=mdp.touch_ball, weight=50.0)
        self.rewards.ball_distance = RewTerm(func=mdp.ball_distance, weight=1.0)
        self.rewards.kick_ball_velocity = RewTerm(
            func=mdp.kick_ball_velocity,
            weight=5.0,
            params={
                "ball_cfg": SceneEntityCfg("soccer_ball"),
                "command_name": "kick_direction",
            },
        )

        # ------------------------------------------------------------------ #
        # Events: ボール位置リセット
        # ------------------------------------------------------------------ #
        self.events.reset_ball = EventTerm(
            func=loco_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                "pose_range": {
                    "x": (-0.05, 0.10),
                    "y": (-0.10, 0.10),
                    "z": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("soccer_ball"),
            },
        )

        # ------------------------------------------------------------------ #
        # Commands: キックタスク向けに歩行速度指令の範囲を調整
        # ------------------------------------------------------------------ #
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------------------ #
        # その他
        # ------------------------------------------------------------------ #
        self.episode_length_s = 12.0


@configclass
class K1WalkKickEnvCfg_PLAY(K1WalkKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
