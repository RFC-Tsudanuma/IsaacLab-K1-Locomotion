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
    # kick_through_speed: キック姿勢に入った後も 0.5 m/s で歩き込ませる (s1b までは 0。walk_command.py 参照)
    base_velocity = mdp.BallWalkVelocityCommandCfg(kick_through_speed=0.5)


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

        # ボールの摩擦のランダム化 (B-Human には無い。s4b から)。ボール-床は 床 1.0 × ボール (multiply)、
        # ボール-足は (ボール + 足) / 2 (average) になる。MuJoCo で床との摩擦 1.0 のときだけ、蹴った足が
        # ボールの上をこすって引き戻す・倒れることがあり (0.5〜0.7 なら起きない)、摩擦に頼った蹴り方になっていたため。
        self.events.ball_material = EventTerm(
            func=loco_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "static_friction_range": (0.4, 1.0),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.1, 0.1),
                "num_buckets": 64,
                "make_consistent": True,
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
        # B-Human は 5。強く蹴る動きが育たず (s1/s1b: 1000 iter でボール最高速度 1.3 m/s のまま)、
        # B-Human の学習量 (4096 env × 何万 iter) に比べてうちは試行回数が少ないので、強キックの報酬を 2 倍にする。
        # (README も片足しか使わない時に ball_kick_velocity_strong / ball_kick_velocity の重みを動かしている)
        self.rewards.ball_kick_velocity_strong = RewTerm(func=mdp.ball_kick_velocity_strong, weight=10.0)
        self.rewards.ball_walk_speed = RewTerm(func=mdp.ball_walk_speed, weight=2.0)
        self.rewards.ball_walk_target_overshoot = RewTerm(func=mdp.ball_walk_target_overshoot, weight=-100.0)
        self.rewards.ball_sole_yaw = RewTerm(func=mdp.ball_sole_yaw, weight=0.0)
        self.rewards.gait_phase_factor = RewTerm(func=mdp.gait_phase_factor, weight=-0.1)

        # ---- terminations: 歩行の転倒判定 + B-Human のエピソード区切り ----
        self.terminations.kicks_done = DoneTerm(func=mdp.kicks_done, time_out=True)
        self.terminations.ball_timeout = DoneTerm(func=mdp.ball_timeout, time_out=True)


# --------------------------------------------------------------------------- #
# 段階学習 (B-Human README「Training kick」の 2) 以降)。段の番号は README の 2), 3), 3), 4), 5), 6) を
# 順に 2..7 とした (README は「3)」が 2 回ある)。前の段の checkpoint から --resume で続ける。
# --------------------------------------------------------------------------- #
@configclass
class K1BHumanWalkKickStage2EnvCfg(K1BHumanWalkKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        # B-Human はここで target_sole_yaw_angle = 1.5705 (足をキック方向に対して 90° 横に向ける = インサイドで
        # 押し出すキック) にする。今回の目的は「強く・正確に」なので、Stage 1 で育った
        # 「足先をキック方向に向けて振り抜く」蹴り方を保ち、0 のままにする (B-Human からの変更)。
        # 足の向きの報酬 (ball_sole_yaw) と sigma の絞り込みは B-Human どおり使う (足先をキック方向へ揃える向きに働く)。
        k.target_sole_yaw_angle = 0.0
        k.disable_sole_yaw_direction_penalty = True
        k.disable_deviation_flag = False
        k.ball_pos_noise_x = (0.03, k.ball_pos_noise_x[1])
        k.ball_pos_noise_y = (0.03, k.ball_pos_noise_y[1])
        self.rewards.ball_sole_yaw.weight = 25.0
        # Stage 1 で強キックが目標 (3.46 m/s) に届いたので、強キックの重みは B-Human の 5 に戻す
        # (距離指定キックまで強く蹴りすぎるのを抑える)。
        self.rewards.ball_kick_velocity_strong.weight = 5.0
        # B-Human は蹴ってから 20 ステップ (0.4 秒) でボールを置き直す。MuJoCo で蹴った後に転がるボールが
        # 脚に絡んで倒れることがあったので、1 秒 (50 ステップ) ボールを残して蹴った後の動きも学習させる (B-Human からの変更)。
        k.time_after_kick = 50


@configclass
class K1BHumanWalkKickStage3EnvCfg(K1BHumanWalkKickStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # README: target_sole_yaw_sigma を「何段かに分けて」0.1 へ。ここでは 1 段で 0.1 にする
        self.commands.kick.target_sole_yaw_sigma = 0.1
        self.rewards.ball_walk_speed.weight = 1.0
        self.rewards.ball_sole_yaw.weight = 15.0


@configclass
class K1BHumanWalkKickStage4EnvCfg(K1BHumanWalkKickStage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.disable_sole_yaw_direction_penalty = False
        k.kick_direction_negative_factor = 3.0
        self.rewards.ball_sole_yaw.weight = 5.0
        self.rewards.ball_walk_target_overshoot.weight = -10.0
        # B-Human はここで feet_swing (足上げ報酬) を 3 → 0.75 にする。うちの walk の位相報酬 feet_phase を同じく 1/4
        # (1.5 → 0.375) にしたら (s4/s4b)、蹴り足をボールの上の方へ振り下ろす蹴り方に変わり、MuJoCo で倒れる・ボールを
        # 引きずるようになったので、feet_phase は 1.5 のまま下げない (B-Human からの変更)。
        # さらにボールの中心より上を踏みつけるように当てたキックは褒めない (strike_height_range, B-Human に無い)。
        # 当たる直前の足裏の一番低い点が 0.05 m 以下なら満額、0.09 m で 0
        # (IsaacLab の評価で強キックの平均は s3: 0.056 m、s4b: 0.072 m。MuJoCo で s4b は倒れる・引きずる)。
        k.strike_height_range = (0.05, 0.09)


@configclass
class K1BHumanWalkKickStage5EnvCfg(K1BHumanWalkKickStage4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.kick.kick_direction_sigma_range = 0.75
        self.commands.kick.kick_direction_sigma_just_hit = 0.5


@configclass
class K1BHumanWalkKickStage6EnvCfg(K1BHumanWalkKickStage5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.kick_direction_sigma_range = 0.25
        k.random_ball_distance_reset = 0.0005
        self.rewards.ball_sole_yaw.weight = 0.0
        self.rewards.ball_walk_speed.weight = 1.0


@configclass
class K1BHumanWalkKickStage7EnvCfg(K1BHumanWalkKickStage6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.random_ball_reset = (0.01, 0.02, 0.02)
        k.reset_ball_request = 0.005
        k.add_ball_vel = 0.0001
        k.add_ball_vel_in_direction = 0.0001
        k.kick_pose_increase_value = 0.2
        k.ball_distance_offset = 0.7
        k.ball_rotation_offset = 0.2
        self.rewards.ball_kick_direction.weight = 5.0
        self.rewards.ball_walk_speed.weight = 0.75
        self.rewards.ball_kick_velocity.params["sigma_scale"] = 0.5


@configclass
class K1BHumanWalkKickEnvCfg_PLAY(K1BHumanWalkKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.kick.obs_noise = False
        self.events.push_robot = None
