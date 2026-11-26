# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール奪取環境 (Isaac-Steal-K1-v0, 2026-11-03)。

円柱障害物 (敵ロボット想定) の周囲 50 cm 以内 (全周、裏側も含む) に置かれた静止ボールへ近づき、足で触れて
障害物に当てない方向へ動かす (敵の圏外 = 中心から 0.8 m へ出す) 方策を、凍結歩行 expert
(frozen_policies/best_walk) + 関節残差で学習する。構成は Block (block_env_cfg.py) に準じる:

* scene: Block のシーン (ボール + 足接触センサ) + kinematic 円柱 ``obstacle``。
* action: Block と同じ ``joint_pos`` (12) + ``walk_command`` (3) + ``joint_residual`` (12)。
* command: ``base_velocity`` = `ExternalVelocityCommand`、``steal`` = `StealCommand` (配置 + 判定)。
* observation (方策用に追加、expert 用の ``command`` / ``policy`` / ``critic`` は据え置き):
    - ``steal_policy``: 歩行 K1PolicyCfg (線速度なし・ノイズあり) + 前回残差 + ボール位置/速度 + 障害物位置
    - ``steal_critic``: 歩行 K1CriticCfg (線速度・ZMP 込み・ノイズなし) + 同上
* reward: 奪取 (+) / 失敗 (−) / 転倒 (−) / 胴体→ボール接近 / ボールが障害物から離れる速度 / 時間罰則 /
  残差・コマンドの大きさと変化率 (−)
* termination: 奪取 (hold 後)・失敗で終了 (time_out=False)。胴体接触 (base_contact) では終了しない。

方向指定版 (Isaac-StealDir-K1, `K1StealDirEnvCfg`): 上記に目標方向 (前/左/後/右) の観測 ``target_dir`` (2) を
末尾に足し、報酬の「障害物から離れる速度」を「目標方向の速度」に置き換え、方向違いで圏外へ出たら
``ball_wrong_dir`` (−20) で終了する。配置・判定の詳細は mdp/steal.py。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .block_env_cfg import K1BlockActionsCfg, K1BlockSceneCfg
from .flat_env_cfg import K1FlatObservationsCfg, K1FlatStancePlanePolishCfg
from .mdp.block import joint_residual_l2
from .mdp.events import randomize_floor_friction
from .mdp.observations import ball_pos_rel, ball_vel
from .mdp.rewards import base_ang_acc_l2, base_lin_acc_l2
from .mdp.steal import (
    StealCommandCfg,
    ball_away_speed,
    ball_lost,
    ball_no_foot,
    ball_stolen,
    ball_circling,
    ball_wait,
    ball_target_speed,
    ball_wrong_dir,
    body_ball_progress,
    foot_near_ball,
    steal_ctrl,
    steal_hist_obs,
    steal_hist_teacher,
    steal_hist_true,
    steal_move_speed_curriculum,
    steal_neutral_end,
    steal_obs_ball_pos,
    steal_obs_ball_vel,
    steal_obs_opponent,
    steal_obstacle_vel,
    steal_phase_obs,
    steal_target_dir,
    steal_timeout,
    steal_forward,
    touch_forward,
    touch_speed,
    trunk_obstacle_contact,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from .mdp.waypoint import ExternalVelocityCommandCfg, walk_command_rate_l2
from .rough_env_cfg import K1CriticCfg, K1PolicyCfg

OBSTACLE_RADIUS = 0.15
OBSTACLE_HEIGHT = 1.0


@configclass
class K1StealSceneCfg(K1BlockSceneCfg):
    """Block のシーン (反発・減衰付きボール + 足接触センサ) + 敵を模した kinematic 円柱。"""

    obstacle: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=OBSTACLE_RADIUS,
            height=OBSTACLE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=MassPropertiesCfg(mass=10.0),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2), roughness=0.7),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, restitution=0.3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, OBSTACLE_HEIGHT / 2)),
    )
    # 胴体と障害物 (敵) の接触だけを拾うセンサ。歩行の base_contact 終了判定は敵との体当たりでも
    # 発火するので steal では外し、こちらで軽い罰則にする (2026-11-09: 敵の裏側のボールを取りに行く
    # 以上ある程度の体当たりは許容する、というユーザー方針)
    contact_trunk_obstacle: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Obstacle"],
    )


@configclass
class K1StealPolicyCfg(K1PolicyCfg):
    """方策の内界観測: 歩行 K1PolicyCfg (線速度なし・ノイズあり) + 前回残差 + ボール + 障害物。履歴なし。"""

    actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    last_residual = ObsTerm(func=mdp.last_action, params={"action_name": "joint_residual"})
    ball_pos = ObsTerm(func=ball_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
    ball_vel = ObsTerm(func=ball_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    obstacle_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "steal"}, noise=Unoise(n_min=-0.05, n_max=0.05))


@configclass
class K1StealCriticCfg(K1CriticCfg):
    """critic の内界観測: 歩行 K1CriticCfg (線速度・ZMP 込み・ノイズなし) + 前回残差 + ボール + 障害物。"""

    actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    last_residual = ObsTerm(func=mdp.last_action, params={"action_name": "joint_residual"})
    ball_pos = ObsTerm(func=ball_pos_rel)
    ball_vel = ObsTerm(func=ball_vel)
    obstacle_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "steal"})


@configclass
class K1StealObservationsCfg(K1FlatObservationsCfg):
    steal_policy: K1StealPolicyCfg = K1StealPolicyCfg()
    steal_critic: K1StealCriticCfg = K1StealCriticCfg()


@configclass
class K1StealCommandsCfg:
    base_velocity = ExternalVelocityCommandCfg(asset_name="robot", action_name="walk_command")
    # ボールは障害物の全周 (±180°) に置く。v5 系 (best_steal model_11000) はロボット側 ±75° で学習した (2026-11-09 変更)
    # 敵の方位もロボットの初期向きから全周 (±180°、背を向けた状態を含む)。best_steal_dir model_21997 までは ±45° (2026-11-10 変更)
    steal = StealCommandCfg(
        asset_name="robot",
        obstacle_radius=OBSTACLE_RADIUS,
        obstacle_height=OBSTACLE_HEIGHT,
        ball_side_angle=math.pi,
        bearing=math.pi,
    )


@configclass
class K1StealRewardsCfg:
    """奪取タスクの報酬 (歩行用の報酬は継承しない)。値は RewardManager が dt (0.02) を掛ける。"""

    # 主目的: 奪取 (+2 実効) / 敵に当てた (−4 実効) / 転倒 (−4 実効)
    stolen = RewTerm(func=mdp.is_terminated_term, weight=100.0, params={"term_keys": "ball_stolen"})
    lost = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "ball_lost"})
    fell = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "base_height"})
    # 胴体が敵に触れている間の軽い罰則 (−0.01/step)。体当たり自体は許容する
    trunk_obstacle_contact = RewTerm(func=trunk_obstacle_contact, weight=-0.5)
    # 足以外で圏外へ出した (不成立): 弱い罰則で終了。−200 だと「触らない」方策に落ちる (v4)
    no_foot = RewTerm(func=mdp.is_terminated_term, weight=-20.0, params={"term_keys": "ball_no_foot"})
    # shaping: ボールへ近づく (ポテンシャル, 1 m 接近 = +1) / ボールを障害物から離す速度 / 速く終えるほど得
    body_ball_progress = RewTerm(func=body_ball_progress, weight=50.0)
    ball_away_speed = RewTerm(func=ball_away_speed, weight=2.0, params={"max_speed": 3.0})
    foot_near_ball = RewTerm(func=foot_near_ball, weight=1.0, params={"std": 0.2})
    time_penalty = RewTerm(func=mdp.is_alive, weight=-0.5)
    # 事前分布 (歩行 expert) からのずれと、その変化率・コマンド変化率は軽く抑える (Block と同じ値)
    joint_residual_l2 = RewTerm(func=joint_residual_l2, weight=-0.05)
    joint_residual_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.02, params={"action_name": "joint_residual"})
    walk_command_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.05, params={"action_name": "walk_command"})


@configclass
class K1StealEnvCfg(K1FlatStancePlanePolishCfg):
    scene: K1StealSceneCfg = K1StealSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: K1StealObservationsCfg = K1StealObservationsCfg()
    actions: K1BlockActionsCfg = K1BlockActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 親の __post_init__ は歩行コマンド cfg を触るので、コマンドはその後で差し替える
        self.commands = K1StealCommandsCfg()

        # expert が見る actions 観測は関節指令 (12) のみ (action 空間が 12 + 3 + 12 になるため)
        self.observations.policy.actions.params = {"action_name": "joint_pos"}
        self.observations.critic.actions.params = {"action_name": "joint_pos"}

        self.rewards = K1StealRewardsCfg()

        # 胴体接触では終了しない (敵との体当たりを許容)。転倒は base_height (0.35 m) で検出する
        self.terminations.base_contact = None
        self.terminations.ball_stolen = DoneTerm(func=ball_stolen)
        self.terminations.ball_lost = DoneTerm(func=ball_lost)
        self.terminations.ball_no_foot = DoneTerm(func=ball_no_foot)

        # 歩行コマンド系のカリキュラムは全て無効 (base_velocity は ExternalVelocityCommand)
        self.curriculum.lin_vel_command = None
        self.curriculum.extreme_commands = None
        self.curriculum.command_resampling_time_range = None
        self.curriculum.push_robot_stage1 = None

        # 歩容位相: 凍結 expert (best_walk model_56444, K1FlatStancePlaneWideCfg) と同じ写像にする。ノルム依存
        # (1.8 → 2.0 Hz@1.8 m/s) は親のまま、横移動時のみ (|vy| − 0.5) × 0.75 Hz を加算 (2026-11-18)。
        # 位相の積分は 1 step 内で最初に gait_phase を評価したグループの params で進むので 4 グループとも揃える
        # (Block の K1BlockFastLateralEnvCfg と同じ対処)。
        lateral_phase_freq = {"lateral_low": 0.5, "lateral_gain": 0.75}
        for group in (self.observations.policy, self.observations.critic, self.observations.steal_policy, self.observations.steal_critic):
            group.gait_phase.params.update(lateral_phase_freq)

        # 立位スタート (Block / waypoint と同じ)
        self.events.reset_robot_joints.params["position_range"] = (-math.radians(5.0), math.radians(5.0))
        self.events.reset_base.params["velocity_range"] = {k: (-0.1, 0.1) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
        }

        # 足の摩擦 DR: 歩行 expert R9 (best_walk model_64440) の学習と同じ U(0.3, 3.0) を床 (env) ごとに 1 つ引く
        # (2026-11-21 ユーザー指示。model_38200 までは親の U(0.6, 1.0)・shape ごと)。
        # 歩行 / Block と違い、掛けるのは足だけ: 全 body (.*) に掛けると μ > 1 の env で物理が発散し
        # (角速度が桁外れ → value loss 1e8 → 120 iter で方策が崩壊)、足だけなら発散しない (80 iter のプローブで確認。
        # 敵の円柱を無摩擦にしても発散は残ったので、足以外の body 同士か対ボールの接触が原因)。足と床の摩擦の分布は歩行と同じ
        self.events.physics_material.params["asset_cfg"].body_names = ".*_foot_link"
        self.events.physics_material.func = randomize_floor_friction
        self.events.physics_material.params.update(
            {"static_friction_range": (0.3, 3.0), "dynamic_friction_range": (0.3, 3.0),
             "per_env": True, "log_uniform": False}
        )

        # 平面地形 (サッカー場)。親の 30% 凹凸地形 (1〜4 cm) だと静置したボールが勝手に転がり、
        # ロボットが触れずに lost / stolen が 8% 程度出る (2026-11-03 ゼロ方策診断)。
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 最長 3 m の接近 + 蹴り + hold 1.5 s。成功・失敗で早期終了するので大半はこれより短い。
        self.episode_length_s = 8.0


@configclass
class K1StealEnvCfg_PLAY(K1StealEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.steal_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# 方向指定版 (Isaac-StealDir-K1)
# ---------------------------------------------------------------------------
@configclass
class K1StealDirPolicyCfg(K1StealPolicyCfg):
    # ボール・敵は StealCommand が作る「方策が見る値」(試合条件ではノードの代替則 + 劣化、それ以外は真値)。
    # 並び (69 次元) と一様ノイズは従来と同じ
    ball_pos = ObsTerm(func=steal_obs_ball_pos, noise=Unoise(n_min=-0.05, n_max=0.05))
    ball_vel = ObsTerm(func=steal_obs_ball_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    obstacle_pos = ObsTerm(func=steal_obs_opponent, noise=Unoise(n_min=-0.05, n_max=0.05))
    target_dir = ObsTerm(func=steal_target_dir)


@configclass
class K1StealDirCriticCfg(K1StealCriticCfg):
    target_dir = ObsTerm(func=steal_target_dir)
    # 動く敵 (2026-11-11): 敵の速度は critic のみ (方策の入力 69 次元と実機ノードの契約は変えない)
    obstacle_vel = ObsTerm(func=steal_obstacle_vel)
    # 試合条件 (2026-11-22): 混合重み w・起動前・解除中/後 (critic 76 → 79)
    steal_phase = ObsTerm(func=steal_phase_obs)


@configclass
class K1StealCtrlCfg(ObsGroup):
    """方策の外の制御量 [w, 歩行コマンド (3)]。StealActorCritic.env_actions が歩行との混合に使う。actor / critic には入れない。"""

    ctrl = ObsTerm(func=steal_ctrl)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1StealDirObservationsCfg(K1FlatObservationsCfg):
    steal_policy: K1StealDirPolicyCfg = K1StealDirPolicyCfg()
    steal_critic: K1StealDirCriticCfg = K1StealDirCriticCfg()
    steal_ctrl: K1StealCtrlCfg = K1StealCtrlCfg()


@configclass
class K1StealDirRewardsCfg(K1StealRewardsCfg):
    ball_away_speed = None
    # 360° 学習の v3 (2026-11-09) と同じ: 足をボールへ寄せる shaping を 3 倍 (裏側での転倒が減った)
    foot_near_ball = RewTerm(func=foot_near_ball, weight=3.0, params={"std": 0.2})
    # 2.0 より 5.0 の方が良かった (2026-11-10: 85.6% vs 82.4%、best_steal_dir は 5.0 で学習)
    ball_target_speed = RewTerm(func=ball_target_speed, weight=5.0, params={"max_speed": 3.0})
    # 以下は敵の方位 ±180° (背を向けた開始を含む) の学習で決めた値 (2026-11-11、best_steal_dir model_30700)。
    # 背後スタートで立位から後退・横・旋回と大きな残差を同時に出して転倒していたので、残差とコマンド変化率の
    # 罰則を 4 倍にし、転倒罰則を −200 → −400 → −800 → −1600 と強めるたびに奪取率も転倒も改善した
    # (24600: 奪取 85.2% / 転倒 7.1% → 30700: 94.0% / 1.0%)。最後に足以外・違う方向を −60 にして上積み。
    fell = RewTerm(func=mdp.is_terminated_term, weight=-1600.0, params={"term_keys": "base_height"})
    no_foot = RewTerm(func=mdp.is_terminated_term, weight=-60.0, params={"term_keys": "ball_no_foot"})
    wrong_dir = RewTerm(func=mdp.is_terminated_term, weight=-60.0, params={"term_keys": "ball_wrong_dir"})
    # 時間切れの罰 (2026-11-22): 試合条件で学習すると「挑まずに時間切れ」が失敗 (−60〜−1600) より得になり、
    # 動く敵で時間切れが 10 → 50% に増えた。0 で従来どおり
    timeout = RewTerm(func=steal_timeout, weight=0.0)
    # 回り込みの罰 (2026-11-24): ボールのまわりを回った角度 [rad] × weight × dt。0 で従来どおり
    ball_circling = RewTerm(func=ball_circling, weight=0.0)
    # 歩いてきた勢いのまま蹴らせる実験 (2026-11-25、STEAL_HANDOFF)。0 で従来どおり。
    # ball_wait: ボールが 0.5 m 以内で未接触の間 weight /s、touch_speed: 最初に触れた瞬間の胴体の速さ × weight (1 回だけ)
    ball_wait = RewTerm(func=ball_wait, weight=0.0)
    touch_speed = RewTerm(func=touch_speed, weight=0.0)
    touch_forward = RewTerm(func=touch_forward, weight=0.0)  # 最初に触れた瞬間、ボールが胴体の 0.35 m 以上前なら weight (1 回だけ)
    steal_forward = RewTerm(func=steal_forward, weight=0.0)  # 同じ量を、奪取が成立した step に払う (成功時だけ)
    joint_residual_l2 = RewTerm(func=joint_residual_l2, weight=-0.2)
    walk_command_rate_l2 = RewTerm(func=walk_command_rate_l2, weight=-0.2, params={"action_name": "walk_command"})
    # 上体の揺れ (2026-11-11 追加): 実機のボール検出 (頭部カメラ) を安定させるため、Trunk の傾き・ロール/ピッチ
    # 角速度・角加速度・並進加速度を罰する。best_steal_dir model_38200 の学習値 (最初に入れた値の 8 倍)。
    # 1 → 2 → 4 → 8 倍と段階的に上げ、奪取率を落とさず傾き 0.186 → 0.058 / 角速度 1.77 → 0.53 rad/s。
    # 揺れの小さい方策から学習を続ける前提の値で、揺れの大きい方策にいきなり掛けると揺れ罰が奪取報酬を
    # 大きく上回る (model_30700 の揺れでは 1 倍で合計 −0.6/s、時間罰則 −0.5/s と同程度)。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-40.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.8)
    base_ang_acc_l2 = RewTerm(func=base_ang_acc_l2, weight=-4.0e-4)
    base_lin_acc_l2 = RewTerm(func=base_lin_acc_l2, weight=-8.0e-3)


@configclass
class K1StealDirEnvCfg(K1StealEnvCfg):
    observations: K1StealDirObservationsCfg = K1StealDirObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.steal.directed = True
        # 試合条件 (2026-11-22, STEAL_HANDOFF.md「次の再学習」の 1〜6): 歩行中からの開始・途中解除、敵が寄ってくる配置と
        # 蹴られるボール、ボール / 敵の観測の劣化とノードの代替則。eval_steal.py は --match のときだけ有効にする
        self.commands.steal.match_enable = True
        self.terminations.steal_neutral = DoneTerm(func=steal_neutral_end, time_out=True)
        # 動く敵とボール (ドリブル対策, 2026-11-11): 速度上限 0 から奪取率 85% 以上で 0.1 m/s ずつ 1.0 m/s まで
        self.curriculum.steal_move_speed = CurrTerm(func=steal_move_speed_curriculum)
        self.rewards = K1StealDirRewardsCfg()
        self.terminations.ball_wrong_dir = DoneTerm(func=ball_wrong_dir)


@configclass
class K1StealDirEnvCfg_PLAY(K1StealDirEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.steal_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1StealHistObsCfg(ObsGroup):
    """actor 用の観測履歴 (T × 8, 現在の base yaw 系)。劣化は StealCommand が作る (一様ノイズは足さない)。"""

    hist = ObsTerm(func=steal_hist_obs)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1StealHistTrueCfg(ObsGroup):
    """critic 用の真値履歴 (T × 6)。"""

    hist = ObsTerm(func=steal_hist_true)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1StealHistObservationsCfg(K1StealDirObservationsCfg):
    steal_hist: K1StealHistObsCfg = K1StealHistObsCfg()
    steal_hist_true: K1StealHistTrueCfg = K1StealHistTrueCfg()


@configclass
class K1StealHistEnvCfg(K1StealDirEnvCfg):
    """Block 型 encoder 版 (Isaac-StealHist-K1-v0, 2026-11-22)。試合条件は StealDir と同じ。
    動く敵・ボールの速さの上限を 1.6 m/s に広げる (寄ってくる敵の速さの上限も連動)。"""

    observations: K1StealHistObservationsCfg = K1StealHistObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.steal.move_speed_limit = 1.6
        self.commands.steal.move_speed_max = 1.6


@configclass
class K1StealHistEnvCfg_PLAY(K1StealHistEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.steal_policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1StealTeacherPolicyCfg(K1StealPolicyCfg):
    """教師の actor 入力 (69): StealDir の方策観測と同じ並び・ノイズで、ボール・敵は真値 (劣化なし)。"""

    target_dir = ObsTerm(func=steal_target_dir)


@configclass
class K1StealTeacherHistCfg(ObsGroup):
    hist = ObsTerm(func=steal_hist_teacher)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1StealStudentObservationsCfg(K1StealHistObservationsCfg):
    teacher_policy: K1StealTeacherPolicyCfg = K1StealTeacherPolicyCfg()
    teacher_hist: K1StealTeacherHistCfg = K1StealTeacherHistCfg()


@configclass
class K1StealStudentEnvCfg(K1StealHistEnvCfg):
    """教師-生徒の蒸留 (Isaac-StealStudent-K1-v0, 2026-11-23、scripts/rsl_rl/distill_steal.py)。試合条件・観測劣化は
    StealHist と同じ。教師 (劣化なしで学習した StealHist) 用に真値の観測グループ teacher_policy / teacher_hist を足す。"""

    observations: K1StealStudentObservationsCfg = K1StealStudentObservationsCfg()
