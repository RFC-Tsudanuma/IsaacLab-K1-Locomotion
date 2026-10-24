# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行 (model_34995, K1FlatEnvCfg の履歴入力版) の上に、B-Human の walk-kick の学習を載せたタスク。

参照元: https://github.com/bhuman/MachineLearning (IsaacGymRL/envs/t1_ball.py, K1_Ball.yaml, README)

土台は歩行のまま変えない:

* ロボット・アクチュエータ・DR・制御周期・観測の作り (100 ステップ履歴 + 最新コマンド) は K1FlatEnvCfg。
* 1 ステップの観測は「歩行の 49 項 (同じ並び)」の後ろにキック用の項を足したもの
  (``locomotion.history_layout.KICK_*``)。歩行の重みは足した入力の重みを 0 にして移す
  (``scripts/rsl_rl/convert_walk_to_bhuman_kick.py``) ので、学習開始時の動きは歩行のまま。
* 歩行の報酬 (姿勢・足の運び・位相・関節の負担) はそのまま残す。速度コマンドの追従報酬だけ外す。

B-Human から取り入れるもの:

* 13 番目の action = 歩行周波数 (位相の進み方を方策が変える)。0 なら歩行と同じ。
* ボールの置き方、キック方向・強さの指令 (強キック / 距離指定)、キックの検出、ボール観測のノイズ、
  ボールのリセット (BHumanKickCommand)。
* ボール関係の報酬 (キック方向・ボール速度・強キック・歩き方・行き過ぎ・キック足の向き) と重み。
* 段階学習 (Stage 1..)。

速度コマンド (歩行の方策の入力) は B-Human の歩行目標から作る (mdp/walk_command.py)。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass

from ..locomotion.flat_env_cfg import (
    K1FlatCommandCfg,
    K1FlatCriticHistoryCfg,
    K1FlatEnvCfg,
    K1FlatPolicyHistoryCfg,
)
from ..locomotion.velocity_env_cfg import JOINT_NAMES_K1
from . import mdp

BALL_RADIUS = 0.11  # このリポジトリのキック系タスクと同じボール
BALL_MASS = 0.45
JOINT_ACTION_SCALE = 0.5


# --------------------------------------------------------------------------- #
# Actions / Commands
# --------------------------------------------------------------------------- #
@configclass
class BHumanWalkKickActionsCfg:
    # 並び: 脚 12 (歩行と同じ) → 歩行周波数 1 (B-Human の 13 番目)
    joint_pos = loco_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_K1,
        preserve_order=True,
        scale=JOINT_ACTION_SCALE,
        use_default_offset=True,
    )
    gait = mdp.GaitFrequencyActionCfg(asset_name="robot")


@configclass
class BHumanWalkKickCommandsCfg:
    # kick を先に置く (base_velocity は kick の状態から作るので、CommandManager で後に計算する)
    kick = mdp.BHumanKickCommandCfg(joint_names=tuple(JOINT_NAMES_K1), ball_radius=BALL_RADIUS)
    base_velocity = mdp.BallWalkVelocityCommandCfg()


# --------------------------------------------------------------------------- #
# Observations: 歩行の項 (同じ並び) の後ろにキック用の項 (history_layout.KICK_EXTRA_TERM_SPECS と同じ順)
# --------------------------------------------------------------------------- #
@configclass
class BHumanWalkKickPolicyCfg(K1FlatPolicyHistoryCfg):
    # 歩行の actions 項 (12) と同じ幅にする。周波数の action は gait_frequency_action 項で入れる。
    actions = ObsTerm(func=mdp.last_joint_action)
    ball_position = ObsTerm(func=mdp.ball_position, scale=0.1)
    kick_direction = ObsTerm(func=mdp.kick_direction)
    kick_direction_sincos = ObsTerm(func=mdp.kick_direction_sincos)
    kick_flags = ObsTerm(func=mdp.kick_flags)
    ball_velocity = ObsTerm(func=mdp.ball_velocity, scale=0.1)
    expected_ball_velocity = ObsTerm(func=mdp.expected_ball_velocity, scale=0.1)
    gait_frequency_action = ObsTerm(func=mdp.gait_frequency_action)


@configclass
class BHumanWalkKickCriticCfg(K1FlatCriticHistoryCfg):
    actions = ObsTerm(func=mdp.last_joint_action)
    ball_position = ObsTerm(func=mdp.ball_position, scale=0.1)
    kick_direction = ObsTerm(func=mdp.kick_direction)
    kick_direction_sincos = ObsTerm(func=mdp.kick_direction_sincos)
    kick_flags = ObsTerm(func=mdp.kick_flags)
    ball_velocity = ObsTerm(func=mdp.ball_velocity, scale=0.1)
    expected_ball_velocity = ObsTerm(func=mdp.expected_ball_velocity, scale=0.1)
    gait_frequency_action = ObsTerm(func=mdp.gait_frequency_action)
    kick_pose_state = ObsTerm(func=mdp.kick_pose_state)


@configclass
class BHumanWalkKickObservationsCfg:
    policy: BHumanWalkKickPolicyCfg = BHumanWalkKickPolicyCfg()
    critic: BHumanWalkKickCriticCfg = BHumanWalkKickCriticCfg()
    command: K1FlatCommandCfg = K1FlatCommandCfg()


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #
@configclass
class K1BHumanWalkKickEnvCfg(K1FlatEnvCfg):
    """Stage 1 (B-Human README の「1) Train with default parameters」)。"""

    def __post_init__(self):
        super().__post_init__()

        self.observations = BHumanWalkKickObservationsCfg()
        self.actions = BHumanWalkKickActionsCfg()
        self.commands = BHumanWalkKickCommandsCfg()

        self.scene.num_envs = 1024
        self.scene.env_spacing = 5.0
        self.episode_length_s = 40.0

        # 地面は平面 (B-Human の足裏接地・ボール配置は地面 z = 0 を前提にしている)。
        # 歩行は平面 0.3 / 凹凸 0.7 で学習しているので分布の内側。
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ---- ball ----
        self.scene.soccer_ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SoccerBall",
            spawn=sim_utils.SphereCfg(
                radius=BALL_RADIUS,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.75,  # B-Human _create_ball_asset
                    angular_damping=0.5,
                    max_linear_velocity=20.0,
                ),
                mass_props=MassPropertiesCfg(mass=BALL_MASS),
                collision_props=CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.1
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, BALL_RADIUS + 0.02)),
        )
        self.events.ball_mass = EventTerm(
            func=loco_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "mass_distribution_params": (-0.01, 0.01),
                "operation": "add",
            },
        )

        # ---- curriculum: 速度コマンド系は使わない。押しは歩行の最終段 (model_34995 は通過済み) に固定 ----
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.extreme_commands = None
        self.curriculum.push_robot_stage1 = None
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02)
        }

        # ---- rewards: 歩行の報酬から速度追従だけ外し、B-Human のボール報酬を足す ----
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_lin_vel_xy_coarse = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.ball_kick_direction = RewTerm(func=mdp.ball_kick_direction, weight=10.0)
        self.rewards.ball_kick_velocity = RewTerm(func=mdp.ball_kick_velocity, weight=4.0, params={"sigma_scale": 1.0})
        self.rewards.ball_kick_velocity_strong = RewTerm(func=mdp.ball_kick_velocity_strong, weight=5.0)
        self.rewards.ball_walk_speed = RewTerm(func=mdp.ball_walk_speed, weight=2.0)
        self.rewards.ball_walk_target_overshoot = RewTerm(func=mdp.ball_walk_target_overshoot, weight=-100.0)
        self.rewards.ball_sole_yaw = RewTerm(func=mdp.ball_sole_yaw, weight=0.0)
        self.rewards.gait_phase_factor = RewTerm(func=mdp.gait_phase_factor, weight=-0.1)

        # ---- terminations: 歩行の転倒判定 + B-Human のエピソード区切り ----
        self.terminations.kicks_done = DoneTerm(func=mdp.kicks_done, time_out=True)
        self.terminations.ball_timeout = DoneTerm(func=mdp.ball_timeout, time_out=True)


@configclass
class K1BHumanWalkKickEnvCfg_PLAY(K1BHumanWalkKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.kick.obs_noise = False
        self.events.push_robot = None
