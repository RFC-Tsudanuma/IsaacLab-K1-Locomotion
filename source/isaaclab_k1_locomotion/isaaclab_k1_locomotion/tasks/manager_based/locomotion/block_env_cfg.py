# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール阻止 (キーパー) 環境 (Isaac-Block-K1-v0, 2026-11-01)。

凍結した歩行 expert (frozen_policies/best_walk) を事前分布として、飛んでくるボール
(0.4〜6.0 m/s, 自分の左右 ±1.0 m を狙って来る, 転がり or 20 cm 程度のバウンド) を
自分の後ろへ通さない (止める or 前に弾く) 方策を学習する。

環境は歩行 expert の学習環境 `K1FlatStancePlanePolishCfg` を継承し (waypoint と同じ流儀)、以下を差し替える:

* scene: サッカーボール + 足とボールの接触センサ (dribble と同じ) を追加。ボールは反発・減衰あり。
* action: ``joint_pos`` (12, expert + 残差の関節指令) + ``walk_command`` (3, 歩行コマンド) +
  ``joint_residual`` (12, 残差の保持用)。方策は 15 次元 (コマンド + 残差) を出し、
  `BlockPPO.act` が 27 次元に組み立てる。
* command: ``base_velocity`` = `ExternalVelocityCommand` (walk_command のコピー)、
  ``ball_launch`` = `BallLaunchCommand` (射出 + ゴールライン + 判定)。
* observation (方策用に追加、expert 用の ``command`` / ``policy`` / ``critic`` は据え置き):
    - ``block_policy``: 内界センサ (歩行 K1PolicyCfg から線速度を除いたもの、ノイズあり) + 残差 + ゴールライン
    - ``block_critic``: 同上 + 線速度 (ノイズなし)
    - ``ball_noisy``:   遅延 + 30 fps + ノイズ付きボール観測の履歴 (actor の encoder 入力)
    - ``ball_true``:    真値ボール観測の履歴 (critic の encoder 入力 / latent 蒸留の教師)
* reward: 阻止 (+) / 失点 (−) / 転倒 (−、止めた後の保持中はさらに −) / 足を予測到達点へ (shaping) / 残差・コマンドの大きさと変化率 (−)
* termination: 失点・阻止で終了 (どちらも time_out=False)。
* reset: 立位スタート (関節 ±5°, 初速 ±0.1)。

学習は train.py (``--task Isaac-Block-K1-v0``)。方策・PPO は `agents/block_policy.py`、
runner cfg は `K1BlockPPORunnerCfg`。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .dribble_env_cfg import K1DribbleSceneCfg
from .flat_env_cfg import PLANE_HEAVY_TERRAIN_CFG, K1FlatObservationsCfg, K1FlatStancePlanePolishCfg
from .rough_env_cfg import K1CriticCfg, K1PolicyCfg
from .velocity_env_cfg import ActionsCfg
from .mdp.events import randomize_floor_friction
from .mdp.block import (
    BALL_HISTORY,
    JOINT_DIM,
    BallLaunchCommandCfg,
    BallSpeedCurriculum,
    BallTrapCommandCfg,
    DelayedBallObs,
    TouchVelocityKill,
    ball_conceded,
    ball_escaped,
    ball_held_at_feet,
    ball_speed_after_touch,
    ball_resolved,
    fell_after_resolve,
    ball_state_rel,
    ball_stopped,
    body_forward_l2,
    body_on_intercept,
    foot_on_intercept,
    foot_yield_on_contact,
    joint_residual_l2,
)
from .mdp.waypoint import ExternalVelocityCommandCfg, WalkCommandActionCfg, walk_command_rate_l2


@configclass
class K1BlockSceneCfg(K1DribbleSceneCfg):
    """dribble のシーン (ボール + 足接触センサ) のボールだけ反発・減衰付きに差し替える。"""

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                # 転がり抵抗・空気抵抗の代わり (PhysX の球は無減衰で永久に転がる)
                linear_damping=0.1,
                angular_damping=0.1,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.7),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            # 地形側は restitution 0 × multiply なので、球側を max にして反発を有効にする
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, restitution=0.6, restitution_combine_mode="max"
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.11)),
    )


@configclass
class K1BlockActionsCfg(ActionsCfg):
    """joint_pos (12, expert + 残差) / walk_command (3) / joint_residual (12, 残差の保持用)。"""

    walk_command = WalkCommandActionCfg(asset_name="robot")
    joint_residual = WalkCommandActionCfg(asset_name="robot", dim=JOINT_DIM)


@configclass
class K1BlockPolicyCfg(K1PolicyCfg):
    """方策の内界観測: 歩行 K1PolicyCfg (線速度なし・ノイズあり) + 前回残差 + ゴールライン。履歴なし。"""

    actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    last_residual = ObsTerm(func=mdp.last_action, params={"action_name": "joint_residual"})
    goal_line = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_launch"})


@configclass
class K1BlockCriticCfg(K1CriticCfg):
    """critic の内界観測: 歩行 K1CriticCfg (線速度・ZMP 込み・ノイズなし) + 前回残差 + ゴールライン。"""

    actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    last_residual = ObsTerm(func=mdp.last_action, params={"action_name": "joint_residual"})
    goal_line = ObsTerm(func=mdp.generated_commands, params={"command_name": "ball_launch"})


@configclass
class K1BallTrueCfg(ObsGroup):
    """真値ボール観測 (相対位置 xy + 速度 xy) の履歴。"""

    ball = ObsTerm(func=ball_state_rel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = BALL_HISTORY
        self.flatten_history_dim = True


@configclass
class K1BallNoisyCfg(ObsGroup):
    """カメラ検出を模したボール観測 (遅延 + 30 fps + 検出欠け + ガウスノイズ) の履歴。ノイズは項の中で乗せる。"""

    ball = ObsTerm(
        func=DelayedBallObs,
        params={
            "delay_range": (0.16, 0.30),  # 認識まで含めた遅延 (env ごとに抽選、速度側)
            "camera_fps": 30.0,
            # σ の (近, 遠): 距離の 2 乗で補間し 10 m 以上で遠側に飽和 (2026-11-18)。±最大 ≈ 2σ で
            # 位置 ±10 cm → ±50 cm、速度 ±0.5 → ±3.0 m/s
            "pos_noise": (0.05, 0.25),
            "vel_noise": (0.25, 1.5),
            "noise_dist": 10.0,
            "drop_prob": 0.1,  # 検出欠け
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = BALL_HISTORY
        self.flatten_history_dim = True


@configclass
class K1BlockObservationsCfg(K1FlatObservationsCfg):
    block_policy: K1BlockPolicyCfg = K1BlockPolicyCfg()
    block_critic: K1BlockCriticCfg = K1BlockCriticCfg()
    ball_noisy: K1BallNoisyCfg = K1BallNoisyCfg()
    ball_true: K1BallTrueCfg = K1BallTrueCfg()


@configclass
class K1BlockCommandsCfg:
    base_velocity = ExternalVelocityCommandCfg(asset_name="robot", action_name="walk_command")
    ball_launch = BallLaunchCommandCfg(asset_name="robot")


@configclass
class K1BlockRewardsCfg:
    """阻止タスクの報酬 (歩行用の報酬は継承しない)。値は RewardManager が dt (0.02) を掛ける。

    既定値は best_block (f1 model_23400、2026-11-20、b5@20800 を歩行 expert R9 model_64440 + 摩擦 U(0.3, 3.0) env ごとで再学習) の学習設定。
    """

    # 主目的: 阻止 (+2 実効) / 失点 (−4 実効)。転倒も失点と同程度に罰する。
    blocked = RewTerm(func=mdp.is_terminated_term, weight=100.0, params={"term_keys": "ball_resolved"})
    conceded = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "ball_conceded"})
    fell = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": ["base_contact", "base_height"]})
    # 止めた後の保持 (3 s) 中の転倒は fell に上乗せして計 −600 (実機で止めた直後に転ぶ事例への対策、2026-11-19)
    fell_after_resolve = RewTerm(func=fell_after_resolve, weight=-400.0)
    # 阻止終了時にボールが足元にあるほど blocked に上乗せ (v19d/e で 100〜200: 足元保持 +1pt、失点 +2〜3pt で不採用 → 0)
    stopped_bonus = RewTerm(func=ball_stopped, weight=0.0, params={"std": 0.5})
    # shaping: ボール接近中、予測到達点 (ライン上の y) に足を置く / 胴体を横に合わせる
    foot_on_intercept = RewTerm(func=foot_on_intercept, weight=0.5, params={"std": 0.15})
    body_on_intercept = RewTerm(func=body_on_intercept, weight=0.5, params={"std": 0.6})  # std 0.3 → 0.6: 遠い横球でも早くから胴体を寄せる勾配 (v22d, 2026-11-11)
    # ラインより前へ出る (ボールへ突っ込む) 動きは減点: ライン上で横に動いて止める方針 (2026-11-02 ユーザー指示) margin 0.2 → 1.0 (v22c, 2026-11-11): 射出 12 m・狙い ±2 m では斜め前方の迎撃を許す方が失点が減る (実際の胴体位置はライン後方に留まる)
    body_forward_l2 = RewTerm(func=body_forward_l2, weight=-2.0, params={"margin": 1.0})
    # 阻止後はボールを足元に遅く留めるほど加点 (保持 2.5 s で最大 +1 実効 = blocked の半分)
    ball_held_at_feet = RewTerm(func=ball_held_at_feet, weight=0.4, params={"pos_std": 0.3, "vel_std": 0.5})
    # 弾いた後のボール速度は密に罰する (1.5 m/s で弾いて保持 2.5 s → −0.75 実効、上限 −1。−0.4 は失点率が戻らず −0.2 を採用、v17f)
    ball_speed_after_touch = RewTerm(func=ball_speed_after_touch, weight=-0.2, params={"max_speed": 2.0})
    # 接触フレームでボールの速さを落とした分を加点 (JOGA の touch velocity kill)。v20 (10) / v20b (40) とも弾き出し速度は変わらず → 0 (2026-11-10)
    touch_velocity_kill = RewTerm(func=TouchVelocityKill, weight=0.0, params={"dist": 0.35})
    # 接触時に足をボールの進行方向へ引くと加点 (v18b/d, v19b: 2.0 は無効、5〜10 は「よける」になり失点 30〜60% → 0)
    foot_yield_on_contact = RewTerm(func=foot_yield_on_contact, weight=0.0, params={"dist": 0.3, "min_speed": 0.3})
    # 事前分布 (歩行 expert) からのずれと、その変化率・コマンド変化率は軽く抑える
    joint_residual_l2 = RewTerm(func=joint_residual_l2, weight=-0.05)
    joint_residual_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-1.0, params={"action_name": "joint_residual"})
    walk_command_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.5, params={"action_name": "walk_command"})
    # 関節加速度 (expert + 残差の合算が実際に関節を揺らした分)。歩行環境と同じ量。実機の足首振動対策 (2026-11-18): 変化率 2 項と合わせ
    # ||Δa||² を w2 比 −49%、関節加速度² を −47% にし、失点は悪化しなかった (b4)。弱い −0.3/−0.2/−1e-6 (b3) は失点が 0.6 pt 低く滑らかさは 16% 劣る
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-3.0e-6)


@configclass
class K1BlockEnvCfg(K1FlatStancePlanePolishCfg):
    scene: K1BlockSceneCfg = K1BlockSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1BlockObservationsCfg = K1BlockObservationsCfg()
    actions: K1BlockActionsCfg = K1BlockActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 親の __post_init__ は歩行コマンド cfg を触るので、コマンドはその後で差し替える
        self.commands = K1BlockCommandsCfg()

        # expert が見る actions 観測は関節指令 (12) のみ (action 空間が 12 + 3 + 12 になるため)
        self.observations.policy.actions.params = {"action_name": "joint_pos"}
        self.observations.critic.actions.params = {"action_name": "joint_pos"}

        self.rewards = K1BlockRewardsCfg()

        self.terminations.ball_conceded = DoneTerm(func=ball_conceded)
        self.terminations.ball_resolved = DoneTerm(func=ball_resolved)
        # 止めてから 3 s 立ち続けて阻止 (2.5 → 3.0 s、2026-11-19 ユーザー指示: 止めた直後の転倒を減らす)
        self.commands.ball_launch.resolve_hold_s = 3.0

        # 足の摩擦 DR: 歩行 expert R9 (best_walk model_64440) の学習と同じ U(0.3, 3.0) を床 (env) ごとに 1 つ引く
        # (2026-11-20 ユーザー指示。b5 までは U(0.15, 3.0) を shape ごと = 両足とも滑る床がほぼ出なかった)。
        # 地形 1.0 × multiply なので実効値そのまま
        self.events.physics_material.func = randomize_floor_friction
        self.events.physics_material.params.update(
            {"static_friction_range": (0.3, 3.0), "dynamic_friction_range": (0.3, 3.0),
             "per_env": True, "log_uniform": False}
        )

        # 歩行コマンド系のカリキュラムは全て無効 (base_velocity は ExternalVelocityCommand)
        self.curriculum.lin_vel_command = None
        self.curriculum.extreme_commands = None
        self.curriculum.command_resampling_time_range = None
        self.curriculum.push_robot_stage1 = None

        # 立位スタート (waypoint と同じ: 親の ±30° / ±0.5 m/s は開始直後の転倒を増やすだけ)
        self.events.reset_robot_joints.params["position_range"] = (-math.radians(5.0), math.radians(5.0))
        self.events.reset_base.params["velocity_range"] = {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
        }

        # ボールは最長 12 m 先から来て弾かれて飛ぶので平面の縁を広げる (env 間の衝突は filter_collisions で切れている)
        self.scene.terrain.terrain_generator = PLANE_HEAVY_TERRAIN_CFG.replace(border_width=15.0)

        # 到達 (12 m を 4 m/s、減衰 0.1 で ≈ 3.6 s) + 阻止保持 3 s + 余裕。阻止・失点で早期終了するので大半はこれより短い。
        self.episode_length_s = 10.5


@configclass
class K1BlockEnvCfg_PLAY(K1BlockEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.block_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# 横移動を速くした変種 (2026-11-04 ユーザー提案)。2026-11-16 まで: 旧 expert (vy ±0.9) に 2.3 Hz@1.5 m/s の
# 位相を与えて横 0.92 m/s を引き出す hack (コマンド上限 (1.5, 1.5, 2.0))。
# 2026-11-16 から: 横 ±1.9 m/s で学習した best_walk model_56444 に合わせ、位相周波数は expert と同じ
# 1.8 → 2.0 Hz@1.8 m/s + 横加算 (|vy| − 0.5) × 0.75 Hz (frozen_policies/best_walk/README.md)。
# コマンド上限は K1BlockFastLateralPPORunnerCfg 側で (2.0, 1.9, 2.0) + |vy| > 1.0 で vx を絞る膝 (BlockActorCritic.command)。
# ---------------------------------------------------------------------------
@configclass
class K1BlockFastLateralEnvCfg(K1BlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        fast = {"low_speed": 1.0, "high_speed": 1.8, "low_freq": 1.8, "high_freq": 2.0, "lateral_low": 0.5, "lateral_gain": 0.75}
        # 位相の積分は 1 step 内で最初に gait_phase を評価したグループの params で進む (policy が先頭) が、
        # 4 グループとも同じ値にして齟齬を無くす
        for group in (self.observations.policy, self.observations.critic, self.observations.block_policy, self.observations.block_critic):
            group.gait_phase.params.update(fast)


@configclass
class K1BlockFastLateralEnvCfg_PLAY(K1BlockFastLateralEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.block_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1BlockFastRawPosEnvCfg(K1BlockFastLateralEnvCfg):
    """実機の入力に合わせ、actor のボール位置は生の検出 (遅延 0.03〜0.10 s)、速度はフィルタ越し (0.16〜0.30 s) にする (2026-11-10、幅は 2026-11-18 に拡大)。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.ball_noisy.ball.params["pos_delay_range"] = (0.03, 0.10)


@configclass
class K1BlockFastRawPosEnvCfg_PLAY(K1BlockFastLateralEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.ball_noisy.ball.params["pos_delay_range"] = (0.03, 0.10)


# ---------------------------------------------------------------------------
# トラップ (パスを足元に止める) タスク Isaac-Trap-K1 (2026-11-14)。カットとトラップは別タスク (2026-11-09 ユーザー方針)。
# 環境・観測・方策は Block (FastRawPos) と同一で、射出分布 (1〜4 m/s、ほぼ正面)・判定 (BallTrapCommand)・報酬だけ違う。
# 失点罰なし: Block で「止める」が出なかったのは失点罰との競合と 8 m/s の速い球だったので、ここで切り分ける。
# ---------------------------------------------------------------------------
@configclass
class K1TrapRewardsCfg:
    """JOGA 型: 足元ゾーン (毎 step) + 吸収 shaping + 弾き出し罰 + 接近中の足の進路合わせ。値は RewardManager が dt を掛ける。
    既定値は best_trap (trap_t3 model_26900、2026-11-21、t2@25798 を歩行 expert R9 model_64440 の上で再学習) の学習設定。
    t2 からの差分は実機で止めた直後に転んだ対策 (2026-11-19 ユーザー指示): 保持中転倒罰 fell_after_resolve、
    保持 3 s と摩擦 DR 0.3〜3.0 (_configure_trap)、Block b4 と同じ滑らかさ罰則 3 項。手順と結果は scripts/rsl_rl/TRAP_HANDOFF.md。"""

    trapped = RewTerm(func=mdp.is_terminated_term, weight=100.0, params={"term_keys": "ball_resolved"})
    fell = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": ["base_contact", "base_height"]})
    fell_after_resolve = RewTerm(func=fell_after_resolve, weight=-400.0)  # 止めた後の保持中の転倒に上乗せ (fell と合わせ −600)。Block と同じ
    lost = RewTerm(func=mdp.is_terminated_term, weight=-300.0, params={"term_keys": "ball_conceded"})  # 後方 2 m に抜けた。−200 → −300 でロスト 4.9 → 4.1 % (成功 −1.7 pt)
    escaped = RewTerm(func=mdp.is_terminated_term, weight=-100.0, params={"term_keys": "ball_escaped"})  # 触った後に前方 1.5 m 超へ逃がした (終了にして成功 +4〜5 pt)
    # 足元ゾーン exp(−(d/0.3)² − (v/0.5)²): JOGA の trapped zone exp(−4v²) と同じ速度スケール
    trap_zone = RewTerm(func=ball_held_at_feet, weight=3.0, params={"pos_std": 0.3, "vel_std": 0.5, "latched": False})
    # 広いゾーン: 届かない所で止まった/遅いボールへ歩いて寄る勾配 (trap_zone は 0.6 m 程で消える)
    trap_zone_wide = RewTerm(func=ball_held_at_feet, weight=1.0, params={"pos_std": 1.5, "vel_std": 0.5, "latched": False})
    touch_velocity_kill = RewTerm(func=TouchVelocityKill, weight=0.0, params={"dist": 0.35})  # 硬い跳ね返しにも加点されるので不使用
    ball_speed_after_touch = RewTerm(func=ball_speed_after_touch, weight=-0.5, params={"max_speed": 2.0})
    # 接触中に足を引いて勢いを殺すと正、蹴ると負 (吸収 shaping: 成功 +2〜4 pt、弾き出し減)
    foot_yield_on_contact = RewTerm(func=foot_yield_on_contact, weight=1.0, params={"dist": 0.3, "min_speed": 0.3})
    foot_on_intercept = RewTerm(func=foot_on_intercept, weight=0.5, params={"std": 0.15})
    body_on_intercept = RewTerm(func=body_on_intercept, weight=0.3, params={"std": 0.6})
    joint_residual_l2 = RewTerm(func=joint_residual_l2, weight=-0.05)
    # 滑らかさ 3 項は Block b4 と同じ値 (t3)。t2 までの弱い値 (−0.02 / −0.05 / 0) だとロスト −1.4 pt・成功 +2.8 pt だが ‖Δa‖² が 1.7 倍 (t4)
    joint_residual_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-1.0, params={"action_name": "joint_residual"})
    walk_command_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.5, params={"action_name": "walk_command"})
    # 関節加速度 (expert + 残差の合算が実際に関節を揺らした分)。歩行環境と同じ量。実機の足首振動対策、0 で無効
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-3.0e-6)


def _configure_trap(cfg) -> None:
    """Block cfg (学習 / Play 共通) をトラップ用に差し替える。終了項の名前は Block と同じ
    (ball_conceded = ロスト、ball_resolved = トラップ成功) にして eval_block.py を流用する。"""
    cfg.commands.ball_launch = BallTrapCommandCfg(asset_name="robot")
    cfg.rewards = K1TrapRewardsCfg()
    cfg.terminations.ball_escaped = DoneTerm(func=ball_escaped)
    # 止めてから 3 s 立ち続けて成功 (1.0 → 3.0 s、2026-11-19 ユーザー指示: 実機で止めた直後の転倒を減らす)。保持中の転倒は fell_after_resolve
    cfg.commands.ball_launch.resolve_hold_s = 3.0
    # 到達 ≤ 3 s + トラップ + 保持 3 s。成功・ロスト・escaped・転倒で早期終了する
    cfg.episode_length_s = 10.0
    # 実機の床の摩擦変動対策 (2026-11-19 ユーザー指示: 0.3〜3.0。Block は 0.15〜3.0、歩行 expert R9 は 0.3〜3.0 per_env)
    cfg.events.physics_material.params["static_friction_range"] = (0.3, 3.0)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.3, 3.0)


@configclass
class K1TrapEnvCfg(K1BlockFastRawPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_trap(self)


@configclass
class K1TrapEnvCfg_PLAY(K1BlockFastRawPosEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_trap(self)


@configclass
class K1BlockCurriculumEnvCfg(K1BlockFastLateralEnvCfg):
    """球速カリキュラム: 上限 2 m/s から始め、「阻止して足元に保持」の割合が 25% を超えるたびに 0.5 m/s ずつ 8 m/s まで上げる。"""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.ball_speed = CurrTerm(
            func=BallSpeedCurriculum, params={"start_speed": 2.0, "final_speed": 8.0, "step": 0.5, "threshold": 0.25}
        )


# ---------------------------------------------------------------------------
# 観測アブレーション (2026-11-04 ユーザー提案): actor のボール観測を遅延 + ノイズ付きの **位置 xy のみ** にする。
# 実機の速度推定は遅延 ≈ 0.6 s あり、速度を観測に入れても sim2real しないため。critic は真値 (位置 + 速度) のまま、
# ノイズ側 encoder には履歴から速度相当を latent に落とすことを期待する (latent 蒸留で真値 latent に寄せる)。
# ---------------------------------------------------------------------------
@configclass
class K1BlockPosOnlyEnvCfg(K1BlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.ball_noisy.ball.params["pos_only"] = True


@configclass
class K1BlockPosOnlyEnvCfg_PLAY(K1BlockEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.ball_noisy.ball.params["pos_only"] = True
