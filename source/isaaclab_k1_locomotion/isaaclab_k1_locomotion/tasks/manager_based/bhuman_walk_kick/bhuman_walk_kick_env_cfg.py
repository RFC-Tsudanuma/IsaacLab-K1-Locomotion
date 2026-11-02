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
import isaaclab.envs.mdp as mdp_isaac
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
from ..locomotion.mdp import rewards as loco_rewards
from ..locomotion.velocity_env_cfg import JOINT_NAMES_K1
from . import mdp

# B-Human K1_Ball.yaml (asset.ball) と同じボール (2026-10-26 から。それまではこのリポジトリの他のキックと同じ 0.11 m・0.45 kg)
BALL_RADIUS = 0.095
BALL_MASS = 0.29
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
                # B-Human (_create_ball_asset) と同じ: 摩擦 1.0、反発 0.1。摩擦の決め方は既定 ("average")。
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

        # ボールの摩擦のランダム化 (s4b〜2026-10-26、B-Human に無い) はやめた。
        # Stage 2 の model_2900 を MuJoCo で回すと、ボール-足の摩擦 0.4 / 0.6 では転倒 0 / 1・向きのずれ 9.7° / 9.0° なのに
        # 0.8 / 1.0 では転倒 27 / 33・ずれ 38° / 57° (B-Human の完成品はどれでも転倒 0〜1・ずれ 2.4°)。学習で高い摩擦を
        # あまり経験していない。IsaacLab で測ると、ランダム化したボールでは ボール-床の摩擦 = ボールの値 (0.40〜0.87) で、
        # ボールの材質の決め方 ("max" にしても) はランダム化で材質が作り直されて効かなかった
        # (scripts/rsl_rl/check_ball_friction.py。1.0 固定に戻した後は ボール-床 0.99)。B-Human はボール 1.0 固定、ロボット 0.3〜1.0 の
        # "average" で、ボール-足 = 0.65〜1.0。ボールを 1.0 固定に戻すと、ボール-床 = 1.0 (MuJoCo の確認と同じ)、
        # ボール-足 = (1.0 + 足) / 2 = 0.65〜1.0 (ロボットの材質が "multiply" として効く場合は 足の値 0.3〜1.0)。

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
        # B-Human に無い (mdp/rewards.py kick_foot_speed)。既定は 0 (使わない)。2026-10-27 の Stage 6 の試し (s6d) で学習コマンドから重みを入れる。
        self.rewards.kick_foot_speed = RewTerm(func=mdp.kick_foot_speed, weight=0.0)
        self.rewards.gait_phase_factor = RewTerm(func=mdp.gait_phase_factor, weight=-0.1)
        # B-Human に無い (mdp/rewards.py ball_bad_touch)。既定は 0 (使わない)。2026-10-31 Stage 8 で学習コマンドから重みを入れる。
        # 理由: s7b model_9400 の MuJoCo 格子 (rel = キック方向 - ボール方位 が -10〜+5° の 384 場面) で、10° 超・失敗の
        # 50 場面のうち約 15 がつま先やすねで近づく途中にボールを押したもの (当たった瞬間の足の向き 0〜35°、ボールが足の前)。
        self.rewards.ball_bad_touch = RewTerm(func=mdp.ball_bad_touch, weight=0.0)

        # ---- 新しい歩行 model_52700 (frozen_policies/best_walk_52700) の学習にあって、model_34995 の K1FlatEnvCfg に無い報酬 ----
        # 既定は 0 (使わない)。model_52700 の上でキックを学習し直すときに学習コマンドから重みを入れる
        # (値は best_walk_52700/env.yaml: no_fly 4.0、joint_power_l2 -1.5e-5。上書きのまとめは train_bhuman_walk_kick.sh)。
        self.rewards.no_fly = RewTerm(
            func=loco_rewards.both_feet_not_in_contact,
            weight=0.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")},
        )
        self.rewards.joint_power_l2 = RewTerm(
            func=loco_rewards.joint_power_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # ---- terminations: 歩行の転倒判定 + B-Human のエピソード区切り ----
        self.terminations.kicks_done = DoneTerm(func=mdp.kicks_done, time_out=True)
        self.terminations.ball_timeout = DoneTerm(func=mdp.ball_timeout, time_out=True)


# --------------------------------------------------------------------------- #
# 段階学習 (B-Human README「Training kick」(IsaacGymRL/envs/readme.md) の 2) 以降)。段の番号は README の
# 2), 3), 3), 4), 5), 6) を順に 2..7 とした (README は「3)」が 2 回ある)。前の段の checkpoint から --resume で続ける。
# キック側 (B-Human) の値は README どおり。探索ノイズ・学習率・desired_kl は学習コマンドの側
# (scripts/rsl_rl/train_bhuman_walk_kick.sh) で指定する。
# --------------------------------------------------------------------------- #
@configclass
class K1BHumanWalkKickStage2EnvCfg(K1BHumanWalkKickEnvCfg):
    """README 2): 蹴り足をキック方向に対して 90° 横に向ける (インサイドキック)。"""

    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.target_sole_yaw_angle = 1.5705
        k.disable_sole_yaw_direction_penalty = True
        k.disable_deviation_flag = False
        k.ball_pos_noise_x = (0.03, k.ball_pos_noise_x[1])
        k.ball_pos_noise_y = (0.03, k.ball_pos_noise_y[1])
        self.rewards.ball_sole_yaw.weight = 25.0
        # Stage 1 だけ強キックの重みを 10 にしていた (B-Human からの変更、s1c)。ここから B-Human の 5。
        self.rewards.ball_kick_velocity_strong.weight = 5.0
        # キック姿勢に入った後に 0.5 m/s で歩き込ませる指令 (B-Human に無い、Stage 1 の s1c で入れた) はやめる。
        # 前回はこれで「歩いたまま足先で押す」蹴り方になった。歩行コマンドは B-Human の歩行目標へ向かうだけにする。
        self.commands.base_velocity.kick_through_speed = 0.0


@configclass
class K1BHumanWalkKickStage3EnvCfg(K1BHumanWalkKickStage2EnvCfg):
    """README 3) の 1 つ目。target_sole_yaw_sigma は「何段かに分けて」0.1 へ下げる。この class は最後の 0.1 で、
    途中の 0.5 / 0.25 は学習コマンドで env.commands.kick.target_sole_yaw_sigma を上書きして回す。"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.kick.target_sole_yaw_sigma = 0.1
        self.rewards.ball_walk_speed.weight = 1.0
        self.rewards.ball_sole_yaw.weight = 15.0


@configclass
class K1BHumanWalkKickStage4EnvCfg(K1BHumanWalkKickStage3EnvCfg):
    """README 3) の 2 つ目。"""

    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.disable_sole_yaw_direction_penalty = False
        k.kick_direction_negative_factor = 3.0
        self.rewards.ball_sole_yaw.weight = 5.0
        self.rewards.ball_walk_target_overshoot.weight = -10.0
        # B-Human は足上げ報酬 feet_swing を 3 → 0.75 (1/4) にする。うちで同じ役目は歩行の位相報酬 feet_phase なので、
        # 同じく 1/4 (1.5 → 0.375) にする。
        self.rewards.feet_phase.weight = self.rewards.feet_phase.weight * 0.25


@configclass
class K1BHumanWalkKickStage5EnvCfg(K1BHumanWalkKickStage4EnvCfg):
    """README 4)。"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.kick.kick_direction_sigma_range = 0.75
        self.commands.kick.kick_direction_sigma_just_hit = 0.5


@configclass
class K1BHumanWalkKickStage6EnvCfg(K1BHumanWalkKickStage5EnvCfg):
    """README 5)。kick_direction_sigma_range は 0.25 → 0.2 (良ければ 0.15, 0.1, 0.05) を学習コマンドで上書きして下げる。
    desired_kl 0.003 と探索ノイズ (エントロピー [-16.6, -16.5] 相当) は学習コマンドの側。"""

    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.kick_direction_sigma_range = 0.25
        k.random_ball_distance_reset = 0.0005
        self.rewards.ball_sole_yaw.weight = 0.0
        self.rewards.ball_walk_speed.weight = 1.0
        # ball_kick_velocity の sigma_scale (K1_Ball.yaml の ball_velocity_range_sigma_scale「1 for training from scratch,
        # 0.5 for final training」) は Stage 6 では 1 のまま、Stage 7 で 0.5 にする。
        # 2026-10-27 に Stage 6 から 0.5 にして 500 iteration 回した (run *_s6、s5 model_7699 → 8200) が、距離指定キックは
        # 良くならなかった (MuJoCo の必要な初速との差 0.91 → 0.85〜1.00 m/s、IsaacLab 0.79 (s5 7500) → 0.90)。
        # 報酬 exp(-差²/幅) の「差が縮むと報酬が増える度合い」は、差 1.5 m/s で 幅 1: 0.32 / 幅 0.5: 0.07、差 1.2 m/s で
        # 0.57 / 0.27、差 0.6 m/s で 0.84 / 1.17。いまの方策は近い距離 (必要な初速 0.5〜1.5 m/s) で 1.3〜1.7 m/s 強すぎ、
        # 遠い距離では差 0.5 m/s 前後なので、幅 0.5 は「遠い距離の速さを合わせる力」を強め「近い距離で弱く蹴る力」を
        # 1/2〜1/5 にしてしまう。B-Human の言う「仕上げ」(差が十分小さくなってから) まで 1 のままにする。


@configclass
class K1BHumanWalkKickStage7EnvCfg(K1BHumanWalkKickStage6EnvCfg):
    """README 6): ボールの置き直し・速度付与の確率を最終値へ (最初は最終値の 10% を学習コマンドで上書きして回す)。
    heavy_ball (README: 0.00001 → 0.0001 の確率でボールを 100 倍重くする) は、B-Human では env を作る時に env ごとに 1 回
    決まるが、1024 env × 0.0001 では重い env が 0.1 個 (ほぼ 0 個) になるので、新しいボールを出すたびに同じ確率で引き直す
    (kick_command.BHumanKickCommand._resample_heavy_ball)。重いボールで蹴る回数の割合は B-Human と同じ。"""

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
        k.heavy_ball = 0.0001
        self.rewards.ball_kick_direction.weight = 5.0
        self.rewards.ball_walk_speed.weight = 0.75
        # K1_Ball.yaml ball_velocity_range_sigma_scale: 「1 for training from scratch, 0.5 for final training」
        # (Stage 6 で先に 0.5 にした結果は Stage 6 の class のコメント参照)
        self.rewards.ball_kick_velocity.params["sigma_scale"] = 0.5


@configclass
class K1BHumanWalkKickStage8EnvCfg(K1BHumanWalkKickStage7EnvCfg):
    """(B-Human の段に無い) Stage 7b の設定 (sigma_range 0.2 = s7b) のまま、MuJoCo の格子で見つけた苦手な場面を直す追加学習。

    - ボールの大きさのランダム化 (B-Human K1_Ball.yaml の ball_radius DR、上限 +0.01 m = ball_radius_margin) を戻す。
      今までは 0.095 m 固定だった。s7b model_9400 を MuJoCo の細かい格子 (rel -30〜+60、1824 場面) で、ボールだけ
      0.085 m にすると 失敗 32 → 68、10° 超 88 → 123 と大きく崩れ、ボールの大きさに合わせ込みすぎている。B-Human の完成品は
      同じ格子で 失敗 6・10° 超 36。
    - 苦手な向き (weak_rel_prob 0.25、kick_command の説明)。

    結果 (2026-10-31、run s8r): 9500〜9900 で細かい格子の失敗 67/49/52/62/42 (s7b 9400 は 32) → 不採用。試しの記録として残す。
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.kick.kick_direction_sigma_range = 0.2
        self.commands.kick.weak_rel_prob = 0.25
        self.scene.replicate_physics = False  # env ごとに違う大きさのボールを PhysX に読ませるため (IsaacLab の説明どおり)
        self.events.ball_scale = EventTerm(
            func=mdp_isaac.randomize_rigid_body_scale,
            mode="prestartup",
            params={"asset_cfg": SceneEntityCfg("soccer_ball"), "scale_range": (1.0, (BALL_RADIUS + 0.01) / BALL_RADIUS)},
        )


@configclass
class K1BHumanWalkKickStage8EnvCfg_PLAY(K1BHumanWalkKickStage8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.events.ball_scale = None  # 再生・評価は MuJoCo と同じ 0.095 m


@configclass
class K1BHumanWalkKickStage9EnvCfg(K1BHumanWalkKickStage7EnvCfg):
    """(B-Human の段に無い) Stage 7b + 速度指令の作り直し + 苦手な向き + 蹴り足の速さ合わせ。2026-10-31 の最良 s9c model_11400。

    学習の順: s7b model_9400 → s9a (下の指令・苦手な向き・回り込む側のしきい値) 9400〜10100 → s9b (+ kick_foot_speed 4)
    10100〜10300 → s9c (kick_foot_speed 20) 10300〜11400。この class は s9c の値 (学習は hydra の上書きで回した。
    claude_tmp/bwk2/train_s9*.sh)。
    - 速度指令 (walk_command): min_side_off 0.15 m、near_yaw_off_deg 20°。理由と測定値は walk_command.py の cfg の説明。
      実機で作る式は docs/bhuman_walk_kick_spec.md「実機のキック側で作る速度指令」。
    - 回り込む側のしきい値 walk_side_hysteresis 0.05 m (kick_command の説明)。目標点の横ずれを 0.15 m にすると側が変わるたびに
      目標点が 0.3 m 跳ぶので入れた。
    - weak_rel_prob 0.25 (kick_command の説明)。
    - kick_foot_speed 20 (rewards.kick_foot_speed の説明)。理由 (MuJoCo、s9b model_10300、距離指定 51 場面): 新しい指令では
      蹴り足が必要な初速によらず 2.98 / 2.99 / 3.15 m/s (近い / 中 / 遠い) で当たり、弱キックの差 0.92〜1.01 m/s (≤0.8 未達)。
      重み 4 は 1 回のキックで 0.5 程度しか払わず強キックの報酬 (約 5) に負けていた。20 にして s9c 11400 で
      近い距離の蹴り足 2.6 m/s 前後・弱キックの差 0.67〜0.79 (5 通り)。
    結果 (MuJoCo、学習と同じ指令): 細かい格子 (1824) 10° 超 4・失敗 2 (s7b 9400 は 92・13、B-Human 完成品 36・6)、
    粗い格子 (1152) 失敗 0.1%、追加条件 最大 0.8%、仕様 5 通りすべて 10/10。
    """

    def __post_init__(self):
        super().__post_init__()
        k = self.commands.kick
        k.kick_direction_sigma_range = 0.2
        k.weak_rel_prob = 0.25
        k.walk_side_hysteresis = 0.05
        self.commands.base_velocity.min_side_off = 0.15
        self.commands.base_velocity.near_yaw_off_deg = 20.0
        self.rewards.kick_foot_speed.weight = 20.0


@configclass
class K1BHumanWalkKickStage9EnvCfg_PLAY(K1BHumanWalkKickStage9EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


def _play(cfg):
    """再生・評価用: env 16、観測ノイズ・ボール観測ノイズ・押しなし。"""
    cfg.scene.num_envs = 16
    cfg.observations.policy.enable_corruption = False
    cfg.commands.kick.obs_noise = False
    cfg.events.push_robot = None
    cfg.commands.kick.heavy_ball = 0.0  # 評価ではボールを重くしない (MuJoCo の確認と同じ 0.29 kg だけ)


@configclass
class K1BHumanWalkKickStage2EnvCfg_PLAY(K1BHumanWalkKickStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1BHumanWalkKickStage3EnvCfg_PLAY(K1BHumanWalkKickStage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1BHumanWalkKickStage4EnvCfg_PLAY(K1BHumanWalkKickStage4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1BHumanWalkKickStage5EnvCfg_PLAY(K1BHumanWalkKickStage5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1BHumanWalkKickStage6EnvCfg_PLAY(K1BHumanWalkKickStage6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1BHumanWalkKickStage7EnvCfg_PLAY(K1BHumanWalkKickStage7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1BHumanWalkKickEnvCfg_PLAY(K1BHumanWalkKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
