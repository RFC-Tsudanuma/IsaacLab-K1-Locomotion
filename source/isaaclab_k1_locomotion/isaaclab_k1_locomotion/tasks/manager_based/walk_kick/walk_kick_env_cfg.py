# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from . import mdp


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
                radius=0.11,
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.11)),
        )

        # 右足・左足とボールの接触を個別に検知（filter_prim_paths_expr でボールのみ）
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
        # Observations: ボール相対位置を追加
        # ------------------------------------------------------------------ #
        self.observations.policy.ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)

        # ------------------------------------------------------------------ #
        # Rewards: ロコモーション報酬を維持しつつキック報酬を追加
        # ------------------------------------------------------------------ #
        self.rewards.touch_ball = RewTerm(func=mdp.touch_ball, weight=50.0)
        self.rewards.ball_distance = RewTerm(func=mdp.ball_distance, weight=1.0)
        self.rewards.kick_ball_velocity = RewTerm(
            func=mdp.kick_ball_velocity,
            weight=5.0,
            params={"command_name": "base_velocity"},
        )

        # ------------------------------------------------------------------ #
        # Events: ボール位置リセット（ロボット前方にランダム配置）
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
        # Commands: キックタスク向けに前進メインに絞る
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
