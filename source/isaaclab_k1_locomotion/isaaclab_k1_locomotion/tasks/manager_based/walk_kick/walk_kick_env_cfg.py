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
"""

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
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
        # Rewards: track_lin_vel_xy を無効化し，ボール保持と前進速度に置き換え
        # ------------------------------------------------------------------ #
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        #self.rewards.action_rate_l2.weight = 0.0

        # Phase 2 以降で有効化（初期 weight=0、カリキュラムで増加）
        self.rewards.ball_in_front = RewTerm(
            func=mdp.ball_in_front,
            weight=0.0,
            params={"fov_half_angle": 0.524},  # ±30°
        )
        self.rewards.robot_xy_speed = RewTerm(
            func=mdp.robot_xy_speed,
            weight=0.0,
        )
        self.rewards.approach_ball = RewTerm(
            func=mdp.approach_ball,
            weight=0.0,
        )
        self.rewards.align_to_kick_direction = RewTerm(
            func=mdp.align_to_kick_direction,
            weight=0.0,
            params={"command_name": "kick_direction"},
        )
        # Phase 3 以降で有効化
        self.rewards.kick_direction_exp = RewTerm(
            func=mdp.kick_direction_exp,
            weight=0.0,
            params={"command_name": "kick_direction", "sigma": 0.5},
        )
        self.rewards.kick_velocity_exp = RewTerm(
            func=mdp.kick_velocity_exp,
            weight=0.0,
            params={"command_name": "kick_direction", "sigma": 1.0},
        )
        self.rewards.single_foot_contact = RewTerm(
            func=mdp.single_foot_contact,
            weight=0.0,
        )

        # ------------------------------------------------------------------ #
        # Curriculum: 3フェーズで報酬重みを段階的に切り替え
        #   Phase 1 (    0-1000): 速度追従のみ
        #   Phase 2 (1000-1500): 速度追従を切り，ボール接近報酬を導入
        #   Phase 3 (1500-2000): さらにキック関連報酬を追加
        # ------------------------------------------------------------------ #

        # steps_per_iteration = num_steps_per_env (PPO config) = 24
        # start_step / end_step はiteration数で指定する
        _spi = 24

        # Phase 1→2: 速度追従報酬をフェードアウト
        self.curriculum.track_lin_vel_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "track_lin_vel_xy_exp", "start_weight": 1.0, "end_weight": 0.0,
                    "start_step": 1000, "end_step": 1500, "steps_per_iteration": _spi},
        )
        self.curriculum.track_ang_vel_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "track_ang_vel_z_exp", "start_weight": 1.0, "end_weight": 0.0,
                    "start_step": 1000, "end_step": 1500, "steps_per_iteration": _spi},
        )

        # Phase 2: ボール接近報酬をフェードイン
        self.curriculum.ball_in_front_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "ball_in_front", "start_weight": 0.0, "end_weight": 10.0,
                    "start_step": 1000, "end_step": 1500, "steps_per_iteration": _spi},
        )
        self.curriculum.approach_ball_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "approach_ball", "start_weight": 0.0, "end_weight": 50.0,
                    "start_step": 1000, "end_step": 1500, "steps_per_iteration": _spi},
        )
        self.curriculum.align_to_kick_direction_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "align_to_kick_direction", "start_weight": 0.0, "end_weight": 10.0,
                    "start_step": 1000, "end_step": 1500, "steps_per_iteration": _spi},
        )

        # Phase 3: キック関連報酬をフェードイン
        self.curriculum.kick_direction_exp_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "kick_direction_exp", "start_weight": 0.0, "end_weight": 30.0,
                    "start_step": 1500, "end_step": 2000, "steps_per_iteration": _spi},
        )
        self.curriculum.kick_velocity_exp_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "kick_velocity_exp", "start_weight": 0.0, "end_weight": 10.0,
                    "start_step": 1500, "end_step": 2000, "steps_per_iteration": _spi},
        )
        self.curriculum.single_foot_contact_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={"term_name": "single_foot_contact", "start_weight": 0.0, "end_weight": -5,
                    "start_step": 1500, "end_step": 2000, "steps_per_iteration": _spi},
        )

        # # ボールに到達したときの成功ボーナス（タイムアウト終了は除外）
        # self.rewards.reach_ball_bonus = RewTerm(
        #     func=mdp.reach_ball_bonus,
        #     weight=10.0,
        # )

        # # ------------------------------------------------------------------ #
        # # Terminations: ボールに到達したらエピソード終了
        # # ------------------------------------------------------------------ #
        # self.terminations.reached_ball = DoneTerm(
        #     func=mdp.reached_ball,
        #     params={"threshold": 0.3},
        # )

        # ------------------------------------------------------------------ #
        # その他
        # ------------------------------------------------------------------ #
        self.scene.env_spacing = 100.0


@configclass
class K1WalkKickEnvCfg_PLAY(K1WalkKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 10
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
