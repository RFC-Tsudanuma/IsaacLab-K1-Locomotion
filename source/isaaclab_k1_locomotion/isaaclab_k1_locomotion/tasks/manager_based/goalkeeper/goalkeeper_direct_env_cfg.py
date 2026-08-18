# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (直接制御版) の環境定義。

階層版 (``goalkeeper_env_cfg.py``: 凍結歩行 + 速度コマンド) の後継。
**12 関節を直接制御する単一ポリシー**で、locomotion の歩行学習をベースに
「左右の横移動」に特化させる。

なぜ作り直したか (計測に基づく):
    凍結歩行ポリシー 0524_walk.pt の横移動は 0.66 m/s が上限で、指令を上げても
    伸びず (1.0 以上は転倒)、体を斜めに向けて前進を流用する案も旋回コストが
    利益を上回るため成立しなかった。一方でエンベロープ計測から、セーブ可否は
    ほぼ「必要横移動量」だけで決まり (0.7m で成功率が半減)、横移動を 2 倍に
    できればセーブ率は 61% → 85% 相当まで伸びる見込みが立った。
    そこで横移動そのものを学習対象にする。

2 ステージ構成 (観測レイアウトは全ステージ共通 = 59 次元):
    * Stage 1 (``Isaac-GoalkeeperDirect-Stage1-K1-v0``)
        locomotion そのままの速度コマンド追従タスク。ただしコマンド範囲を
        横重視 (vx ±1.0 / vy ±1.5) にし、横方向の追従・速度に追加報酬を掛ける。
        ボール系スロットはゼロのダミーで次元だけ確保する。
        ゴール・ボールはシーンに置かない (自由に走り回らせるため)。
    * Stage 2 (``Isaac-GoalkeeperDirect-Stage2-K1-v0``)
        ゴール + ボールを置き、ボール系スロットに実値を入れてセーブを学習。
        ``velocity_commands`` スロットはタスク由来の「移動要求」に差し替える
        (:func:`mdp.task_drive_vector`)。Stage 1 の ckpt から ``--resume``。
        セーブ成功率 (EMA) に応じて難易度を段階的に上げる適応カリキュラム付き
        (:func:`mdp.adaptive_difficulty`: 狙い先の広さ → ボール初速の順)。

観測スロット (順序厳守。変えると warmstart / ステージ間の重み引き継ぎが壊れる):
    先頭 49 = 歩行 K1PolicyCfg と完全同一 (base_ang_vel 3 / projected_gravity 3 /
    velocity_commands 3 / joint_pos 12 / joint_vel 12 / actions 12 / gait_phase 4)
    → ``train.py --warmstart_actor <歩行 ckpt>`` で歩行の重みを引き継げる。
    続く 10 = ball_pos_rel 2 / ball_vel 2 / ball_active 1 / target_y 1 / self_state 4

歩行周期は locomotion と同じ規約 (``_PHASE_FREQ`` = 1.6Hz 固定 +
``randomize_phase_freq`` による env ごとの ±0.05Hz ランダム化) をそのまま使う。
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from ..locomotion.flat_env_cfg import K1FlatEnvCfg
from ..locomotion.mdp.rewards import foot_clearance_ji
from ..locomotion.rough_env_cfg import (
    _COMMAND_THRESHOLD,
    _PHASE_FREQ,
    _STANCE_RATIO,
    K1CriticCfg,
    K1ObservationsCfg,
    K1PolicyCfg,
)

from .goalkeeper_env_cfg import (
    BALL_RADIUS,
    GOAL_HALF_WIDTH,
    GoalkeeperParamsCfg,
    K1GoalkeeperSceneCfg,
)
from .mdp.curriculums import adaptive_difficulty
from .mdp.events import (
    relaunch_ball_after_save,
    reset_ball_perception,
    reset_ball_shot,
    reset_gk_buffers,
    sync_task_command,
)
from .mdp.observations import (
    gk_ball_active,
    gk_ball_pos_rel,
    gk_ball_pos_rel_perceived,
    gk_ball_vel,
    gk_ball_vel_perceived,
    gk_self_state,
    gk_target_y,
    task_drive_phase_obs,
    task_drive_vector,
    zeros_obs,
    zmp_xy_base,
)
from .mdp.rewards import (
    face_field,
    # ★ 待機ゲートを差し替えた action ペナルティ (使用箇所のコメント参照)
    gk_action_rate_l2,
    gk_action_smoothness_l2,
    # ★ 足裏基準の足上げ報酬 (Stage2 で foot_clearance_ji を差し替える。理由は使用箇所を参照)
    foot_clearance_sole,
    hold_default_pose_after_save,
    lateral_speed_bonus,
    save_clearance_bonus,
    save_touch_bonus,
    stance_foot_flat,
    stay_on_goal_line,
    target_reach_velocity_direct,
    track_lin_vel_y_exp,
)
from .mdp.terminations import goal_conceded, robot_out_of_bounds

# 横移動の目標速度 [m/s]。lateral_speed_bonus の正規化基準。
LATERAL_TARGET_SPEED = 1.3

# Stage2/3 の移動要求 (task_drive_vector) の横成分クリップ [m/s]。
# ☠ **Stage1 の横指令レンジ (commands.base_velocity.ranges.lin_vel_y) と揃えること**。
#   小さいと、下位がどれだけ速く動けても Stage2 でその速度に到達する指令が出ない。
# ★ 2026-08-18: 1.3 → 1.5 にしたが、**1.3 に戻した**。
#
#   1.5 は新しい横移動ポリシー (k1_gk_lateral/2026-08-17_12-53-08) の学習レンジ ±1.5 に
#   合わせた値で、**その系譜の Stage2 (13500 から開始) では正しい**。
#   しかし試合 (2026-08-22、予選は 20 日の可能性) までの日数から、Stage2 を
#   **旧系譜 (2026-08-17_10-37-49/model_35200) から継続**する判断をした。
#   旧系譜の Stage1 は 07-28 で横指令レンジが ±1.3 なので、1.5 では
#   **学習レンジ外の指令が出る**ことになる。上の ☠ の条件に反するため戻す。
#
#   新系譜へ移るときは 1.5 に戻すこと。判断材料 (2026-08-18 時点):
#     旧系譜 = Stage2 35200 iter / aim_y_range 1.1 / ball_speed_hi 4.30 / 4.3m/s で 91〜93%
#     新系譜 = Stage2   600 iter / aim_y_range 0.4 / ball_speed_hi 1.00 (登り直しに 2万 iter 必要)
#   歩容 (足上げ) は新系譜が上だが、セーブ性能の積み上げが間に合わない。
# ☠ この値は **観測 (policy/critic の velocity_commands) と sync_task_command の
#   両方**に配ること。片方だけ変えると「ポリシーが見る指令」と「報酬の停止判定が
#   見る指令」がズレる。
TASK_DRIVE_VY_SCALE = 1.3


# ---------------------------------------------------------------------------

@configclass
class K1GKDirectStage1PolicyCfg(K1PolicyCfg):
    """Stage 1 の actor 観測。先頭 49 は歩行 K1PolicyCfg をそのまま継承する。

    ボール系スロットはゼロのダミー。ゼロ入力の列には勾配が流れないので、
    該当する重みは初期値のまま Stage 2 へ渡る (ball_kick と同じ次元一致方式)。

    **項の定義順 = スロット順。ここを変えると warmstart と
    ステージ間の重み引き継ぎが壊れる。**
    """

    ball_pos_rel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_vel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_active = ObsTerm(func=zeros_obs, params={"dim": 1})
    target_y = ObsTerm(func=zeros_obs, params={"dim": 1})
    self_state = ObsTerm(func=zeros_obs, params={"dim": 4})


@configclass
class K1GKDirectStage1CriticCfg(K1CriticCfg):
    """Stage 1 の critic 観測 (歩行 K1CriticCfg + 同じダミースロット)。"""

    # ★ 2026-08-09: world 絶対座標の ZMP を自機基準に差し替える。
    #   K1CriticCfg の既定 (compute_zmp_xy) は env 原点オフセット込みの world 座標で、
    #   正規化後に信号が潰れるうえ左右反転もできない。:func:`zmp_xy_base` 参照。
    #   dataclass の項順は最初の定義位置を保つので、スロット位置は変わらない。
    zmp_position = ObsTerm(func=zmp_xy_base, params={"asset_cfg": SceneEntityCfg("robot")})

    ball_pos_rel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_vel = ObsTerm(func=zeros_obs, params={"dim": 2})
    ball_active = ObsTerm(func=zeros_obs, params={"dim": 1})
    target_y = ObsTerm(func=zeros_obs, params={"dim": 1})
    self_state = ObsTerm(func=zeros_obs, params={"dim": 4})


@configclass
class K1GKDirectStage1ObservationsCfg(K1ObservationsCfg):
    policy: K1GKDirectStage1PolicyCfg = K1GKDirectStage1PolicyCfg()
    critic: K1GKDirectStage1CriticCfg = K1GKDirectStage1CriticCfg()


@configclass
class K1GKDirectPolicyCfg(K1GKDirectStage1PolicyCfg):
    """Stage 2/3 の actor 観測。スロットの順序・次元は Stage 1 と同一で中身だけ差し替える。

    ``velocity_commands`` スロットは、Stage 1 では外部から与えられる速度コマンド
    だったが、Stage 2/3 ではタスク由来の「移動要求」(守備面までの前後ずれ /
    目標 y までの横ずれ / 向きの誤差) に差し替える。意味が揃っているので
    「このスロットが大きい方向へ速く動く」という Stage 1 の学習がそのまま活きる。
    """

    velocity_commands = ObsTerm(
        func=task_drive_vector,
        params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True, "vy_scale": TASK_DRIVE_VY_SCALE},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    )
    # ★ gait_phase もタスク駆動に差し替える (2026-07-24)。
    gait_phase = ObsTerm(
        func=task_drive_phase_obs,
        params={
            "phase_freq": _PHASE_FREQ,
            # cmd_threshold は関数側の既定 (0.12 [m]) を使う。_COMMAND_THRESHOLD
            # (0.05) は速度 [m/s] 用のしきい値なので、位置ずれには流用しない。
            "max_y": GOAL_HALF_WIDTH,
            "use_perceived": True,
            # ★ 2026-08-18: 実測速度ゲートをやめ、**指令ノルム判定に戻す**。
            #
            #   実測速度ゲートは閉ループ (ゲート → 位相 → 方策 → 関節 → 速度 → ゲート)
            #   の中にあり、関数の docstring 自身が「判定がハードしきい値なので
            #   **遅延だけでも自励振動する**」と警告している。実機で問題が出た報告は
            #   2 件ともこのゲート絡みだった (待機時の震え / 起動できないデッドゾーン)。
            #
            #   一方 07-28 Stage1 は指令ノルム判定 (locomotion の phase_obs) で
            #   **実機デプロイ済み・良好**。閉ループが無いので発振しない。
            #   シムでの ||Δaction|| は 07-28 のほうが大きい (0.2075 対 0.1326) のに
            #   実機では静かで、シムの指標が実機の震えを予測していないことも確認済み。
            #   → 実績のある構成に寄せる。
            #
            #   ★ 実測速度に移した動機 (MCL の跳びで指令が 0.12 付近をまたいでトグル)
            #     は is_idle_hold で解消済み。待機中は指令が **厳密ゼロ**、脅威時は
            #     drive_t_fast で飽和して 1.5 付近まで跳ねるので、ゲート入力は
            #     「0 か、しきい値のはるか上」の二択になり、トグルする余地が無い。
            "use_measured_speed": False,
        },
    )
    # ボール系は知覚DR (レイテンシ/更新レート/ドロップ/距離依存ノイズ) 付きの実値。
    # ノイズは関数内で付加するので ObsTerm 側の noise は付けない。
    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel_perceived)
    ball_vel = ObsTerm(func=gk_ball_vel_perceived)
    ball_active = ObsTerm(func=gk_ball_active)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True})
    # 自己位置は実機の MCL 誤差 (バイアス/ドリフト/跳び) 込み。Unoise は残留ジッタ分。
    self_state = ObsTerm(
        func=gk_self_state,
        params={"use_perceived": True},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    )


@configclass
class K1GKDirectCriticCfg(K1GKDirectStage1CriticCfg):
    """Stage 2/3 の critic 観測 (真値・ノイズなし)。"""

    velocity_commands = ObsTerm(
        func=task_drive_vector,
        params={"max_y": GOAL_HALF_WIDTH, "use_perceived": False, "vy_scale": TASK_DRIVE_VY_SCALE},
    )
    # policy と同じくタスク駆動位相 (真値版)。actor/critic で位相の定義がズレると
    # 価値推定が actor の見ている状態と食い違うので必ず揃える。
    gait_phase = ObsTerm(
        func=task_drive_phase_obs,
        params={
            "phase_freq": _PHASE_FREQ,
            # cmd_threshold は関数側の既定 (0.12 [m]) を使う。_COMMAND_THRESHOLD
            # (0.05) は速度 [m/s] 用のしきい値なので、位置ずれには流用しない。
            "max_y": GOAL_HALF_WIDTH,
            "use_perceived": False,
            # ★ policy 側と揃える (2026-08-18 に指令ノルム判定へ戻した。理由は policy 側参照)
            "use_measured_speed": False,
        },
    )
    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel)
    ball_vel = ObsTerm(func=gk_ball_vel)
    ball_active = ObsTerm(func=gk_ball_active)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": False})
    self_state = ObsTerm(func=gk_self_state)


@configclass
class K1GKDirectObservationsCfg(K1ObservationsCfg):
    policy: K1GKDirectPolicyCfg = K1GKDirectPolicyCfg()
    critic: K1GKDirectCriticCfg = K1GKDirectCriticCfg()


# ---------------------------------------------------------------------------

@configclass
class K1GKDirectStage1EnvCfg(K1FlatEnvCfg):
    """Stage 1: 速度コマンド追従を横重視で学習する (ゴール・ボールなし)。

    報酬・イベント・終了条件は locomotion (K1FlatEnvCfg) をそのまま使い、
    横方向の追従報酬と速度ボーナスだけを上乗せする。歩容や姿勢の作り込みは
    歩行タスク側の資産をそのまま活かす。
    """

    observations: K1GKDirectStage1ObservationsCfg = K1GKDirectStage1ObservationsCfg()
    # ステージ 2/3 と同じフィールドを持たせておく (JSON 上書きのパス互換のため)
    goalkeeper: GoalkeeperParamsCfg = GoalkeeperParamsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- 横重視のコマンド範囲 ---
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.3, 1.3)

        # カリキュラムも横重視の段階に差し替える (前進は早々に上限へ、横は細かく伸ばす)
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-0.6, 0.6), (-0.8, 0.8), (-1.0, 1.0), (-1.0, 1.0),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.6, 0.6), (-0.9, 0.9), (-1.1, 1.1), (-1.3, 1.3),
        ]

        # --- 横移動特化の追加報酬 ---
        self.rewards.track_lin_vel_y = RewTerm(
            func=track_lin_vel_y_exp,
            weight=2.5,
            params={"command_name": "base_velocity", "std": 0.3},
        )
        # 2) 実速度そのものへの線形報酬。追従報酬 (ガウス) は達成不能な高速域で
        #    勾配が消えて「諦め」に落ちるので、上限付近でも勾配が残る項を足す。
        self.rewards.lateral_speed_bonus = RewTerm(
            func=lateral_speed_bonus,
            weight=2.0,
            params={
                "v_ref": LATERAL_TARGET_SPEED,
                "command_name": "base_velocity",
                "min_cmd": 0.6,
            },
        )
        # 3) 遊脚のクリアランス (足上げ)。目視で「足が上がりきらず擦って崩れかける」
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance_ji,
            weight=2.5,
            params={
                "command_name": "base_velocity",
                # 0.095 (6cm) → 0.105 (7cm) に引き上げ (2026-07-29)。
                "target_clearance": 0.095,
                "phase_freq": _PHASE_FREQ,
                "stance_ratio": _STANCE_RATIO,
                "cmd_threshold": _COMMAND_THRESHOLD,
            },
        )
        # 上下動 (跳躍) のペナルティを強化する (flat_env_cfg の既定は -0.8)。
        self.rewards.lin_vel_z_l2.weight = -2.5
        self.rewards.feet_parallel_to_ground.params["enable_potential"] = False
        self.rewards.feet_parallel_to_ground.weight = 1.5
        # 4) 足首 (Ankle_Pitch) の基準姿勢からの逸脱ペナルティ。
        self.rewards.ankle_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch"])},
        )
        # 5) 接地脚の足裏水平ペナルティ (つま先立ちの本命対策)。
        self.rewards.stance_foot_flat = RewTerm(
            func=stance_foot_flat,
            weight=-8.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
                "force_threshold": 1.0,
            },
        )
        # 6) 向き維持の強化。wz=0 指令でも横移動中に体が回り「円を描いて横移動」する
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        # 7) 外股 (ガニ股) の矯正。9000iter 版の目視で「股関節を外に開いたまま横移動する」
        self.rewards.joint_deviation_hip.weight = -0.4


@configclass
class K1GKDirectStage1EnvCfg_PLAY(K1GKDirectStage1EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------

@configclass
class K1GKDirectEnvCfg(K1GKDirectStage1EnvCfg):
    """Stage 2: Stage 1 の歩容の上に、ゴール・ボール・セーブ課題を載せる。"""

    scene: K1GoalkeeperSceneCfg = K1GoalkeeperSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: K1GKDirectObservationsCfg = K1GKDirectObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # エピソード継続モードなので長めに取る。ボール到達に最長 10s かかる
        self.episode_length_s = 25.0

        # 完全平面に戻す (凹凸の上ではボールが勝手に転がり判定を汚す)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # --- 初期配置: 守備面 (ゴールラインの guard_x [m] 前) 付近 ---
        # ★ 2026-08-12: guard_x を JSON で変えても初期配置が追従するよう、直値
        #   (0.3, 0.5) をやめて guard_x ± 0.1 から導出する。追従しないと、守備面を
        #   前に出したのに初期位置だけ後ろに残り、エピソード開始直後に必ず前進指令
        #   (dx = (guard_x - x) / 1.0) が出る状態になる。
        _gx = float(self.goalkeeper.guard_x)
        self.events.reset_base.params["pose_range"]["x"] = (_gx - 0.1, _gx + 0.1)
        self.events.reset_base.params["pose_range"]["y"] = (-0.5, 0.5)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.3, 0.3)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }

        # --- リセット順序: 状態バッファ → ボール発射 → 知覚DR 初期化 ---
        self.events.reset_gk_buffers = EventTerm(func=reset_gk_buffers, mode="reset")
        self.events.reset_ball = EventTerm(func=reset_ball_shot, mode="reset")
        self.events.reset_ball_perception = EventTerm(func=reset_ball_perception, mode="reset")
        # エピソード継続モード: セーブ確定から 1.0s 後に次の球を撃つ (毎ステップ判定)。
        # ステージ1 の resample_stage1_target と同じ「成功したら切らずに次の課題」方式。
        _dt = self.sim.dt * self.decimation
        self.events.relaunch_ball = EventTerm(
            func=relaunch_ball_after_save,
            mode="interval",
            interval_range_s=(_dt, _dt),
            is_global_time=True,
            params={"respawn_delay_steps": 50},
        )
        self.events.ball_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "static_friction_range": (0.5, 1.0),
                "dynamic_friction_range": (0.35, 0.8),
                "restitution_range": (0.0, 0.3),
                "num_buckets": 64,
            },
        )

        # --- base_velocity コマンドをタスク由来の移動要求に置き換える (2026-07-31) ---
        self.events.sync_task_command = EventTerm(
            func=sync_task_command,
            params={"vy_scale": TASK_DRIVE_VY_SCALE},
            mode="interval",
            interval_range_s=(_dt, _dt),
            is_global_time=True,
        )
        # 上書きした値をコマンドマネージャに戻されないよう、再計算・再サンプルを止める。
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        # 停止判定のしきい値を 3 箇所で揃える。
        _STOP_TOL = 0.12

        # --- 足上げ報酬を **足裏基準** に差し替える (2026-08-16) ---
        #
        # 継承元 (K1GKDirectEnvCfg) は foot_clearance_ji で、測っているのは
        # **足リンク原点** の高さ。ところが原点は足裏から 3.82cm 上にあり、足首が
        # 底屈しているとつま先はほとんど上がらない。07-28 の実測で
        # **原点 3.2〜3.9cm に対しつま先 0.7cm** = 上げた高さの 4〜7 割が足の傾きで
        # 失われていた。実際につまずくのはつま先なので、この測り方では
        # 「上げているつもりで擦っている」解に満額を払ってしまう。
        #
        # foot_clearance_sole は足裏 4 隅の最小高さを支持脚基準で測り、位相と整合を
        # 取る (詳細は同関数の docstring と goalkeeper_lateral_env_cfg.py の
        # TARGET_SOLE_CLEARANCE の解説)。横移動特化タスクで先に導入した実装をそのまま使う。
        #
        # 目標 0.03m は人工芝のパイル (20〜30mm) を上回り、かつ 07-28 の脚上げ量
        # (3.2〜3.9cm) より低い。**脚を高く上げなくても足首を水平にすれば届く**ので、
        # 速度を削る方向の圧にならない。weight/target を上げる軸は探索済みで、
        # 上げると跳躍に退行することが分かっている (だから測り方を変える)。
        #
        # speed_gate_frac=0.9 は「遅く歩いて足だけ上げる」解を報酬上ありえなくする保険。
        #
        # ★ 2026-08-16: weight 2.5 → 3.5 (ユーザー要望「少し上げて」)。
        #   足リンク原点で測っていた頃は weight を上げると跳躍に退行したが、足裏基準では
        #   **足首を水平に保つ** という速度に無影響な手段でも目標 0.03m に届くので、
        #   圧を上げても跳躍へ逃げる必要がない。跳躍側は lin_vel_z_l2 = -2.5 が押さえる。
        #   それでも跳躍が出たら 2.5 に戻すこと (Episode_Reward/lin_vel_z_l2 の悪化で分かる)。
        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance_sole,
            weight=3.5,
            params={
                "command_name": "base_velocity",
                "target_clearance": 0.03,
                "phase_freq": _PHASE_FREQ,
                "stance_ratio": _STANCE_RATIO,
                "cmd_threshold": _STOP_TOL,
                "speed_gate_frac": 0.9,
            },
        )
        if self.rewards.feet_phase is not None:
            self.rewards.feet_phase.params["cmd_threshold"] = _STOP_TOL

        # --- 待機中の振動ペナルティ (2026-08-17 に判定を作り直した) ---
        #
        # 実機・MuJoCo で「指令ゼロの待機中に小刻みに震える」件への対処。経緯:
        #
        #   1. locomotion の action_smoothness_l2 / action_rate_l2 は
        #      「停止指令かつ base が実際に静止」のとき penalty を stand_still_scale 倍
        #      する仕組みを持つ (rewards._stand_still_boost)。まずこの倍率を 5.0 にした。
        #   2. それでも震えが止まらないので weight も 2.0 / 1.5 倍にした (08-17)。
        #      → **かえって悪化した**。実測 (diag_gk_standstill.py, 静止ボール条件):
        #        待機中ベース速度 0.0124 → 0.0236 m/s、||Δaction|| 0.0443 → 0.0951。
        #   3. 原因を計測して判明したのは、**ゲートがほぼ開いていなかった**こと:
        #        * 静止ブーストの成立率は待機中わずか 1.1%
        #        * 内訳を分解すると ``|ang_vel_z| < 0.2`` の成立率が 7.9%。
        #          待機中の |ang_vel_z| は **平均 0.955 rad/s (≒55°/s)** あった。
        #      つまり **震えているという理由で、震えを抑える罰が無効化されていた**。
        #      weight や scale をどう上げても効かないのは当然だった。
        #
        # 対策は 2 段。どちらも「待機」を 1 つの明示的な状態として定義するもの:
        #
        #   (a) 指令側: is_idle_hold で指令を厳密ゼロにする (observations.py)。
        #       脅威が無く定位置の近くに居れば、post_save_hold と同じ状態にする。
        #   (b) 報酬側: 倍率のゲートから「ベースが静止しているか」を外す
        #       (gk_action_smoothness_l2 / gk_action_rate_l2 の _idle_boost)。
        #       指令が厳密ゼロなら、体が動いていること自体が抑制対象。
        #
        # weight は 08-17 の値を維持する。エピソード全体で見れば意図通り効いており
        # (生のペナルティ量が smoothness -28% / rate -26%)、移動性能も落ちていない
        # (target_reach_velocity・success_ema が同等、難易度は 3.58 → 4.30 m/s に上昇)。
        # 壊れていたのは待機区間だけで、そこは (a)(b) で直す。
        #
        # ★ 調整はまず stand_still_scale で行うこと。ゲート内でしか効かないので
        #   移動性能への影響がゼロ。weight は移動中にも効くので最後の手段。
        # ★ 2026-08-18: scale 5.0 → 8.0、lin_vel_max 0.5 → 1.0。
        #
        #   実測 (diag_gk_standstill.py, 静止ボール) で震えの主因が **方策観測のノイズ**
        #   と確定した。指令がゼロ (is_idle_hold 98%) でも、joint_pos/joint_vel/
        #   base_ang_vel/projected_gravity のノイズに方策が反応して震える:
        #
        #     条件                    base_speed   ||Δaction||
        #     クリーン                  0.0070       0.0047
        #     知覚DR のみ                0.0194       0.0267
        #     **方策観測ノイズのみ**      0.0313       0.1326   ← ほぼ全量を説明
        #
        #   このとき静止ブーストの成立率は 67.5% (クリーンなら 98.5%)。lin_vel_max=0.5 に
        #   引っかかって **震えている env ほど圧が抜ける** 構図が残っていた。1.0 に緩めて
        #   待機中は常時掛かるようにし、併せて倍率も上げる。
        #
        #   ★ どちらも is_idle_hold の中でしか効かないので、移動性能への影響はゼロ。
        #     調整はこの 2 つで行い、weight (移動中にも効く) は触らないこと。
        self.rewards.action_smoothness_l2 = RewTerm(
            func=gk_action_smoothness_l2,
            weight=-0.24,  # locomotion 既定 -0.12 の 2.0 倍
            params={"stand_still_scale": 8.0, "lin_vel_max": 1.0},
        )
        self.rewards.action_rate_l2 = RewTerm(
            func=gk_action_rate_l2,
            weight=-0.6,  # locomotion 既定 -0.4 の 1.5 倍
            params={"stand_still_scale": 8.0, "lin_vel_max": 1.0},
        )

        # --- 平滑化まわりの weight をここに集約 (2026-08-18) ---
        #
        # 元は locomotion (rough_env_cfg / flat_env_cfg) と Stage1 cfg に散らばっていて、
        # どこを直せば何が変わるのか追いにくかった。**値は現状と同一**で、
        # 「ここの数字を書き換えれば調整できる」形に揃えただけ (挙動は変わらない)。
        #
        # ★ 上の action_smoothness_l2 / action_rate_l2 との違い:
        #     あの 2 つは **指令 (action) の時間変化** を罰する = 出力の滑らかさ。
        #     以下は **実際の動き** を罰する = 挙動の滑らかさ。
        #   実機の震え対策で効くのは前者、歩容の質に効くのは後者。
        #
        # ★ 強めるほど「動かない」方向に圧がかかる。過去に何度も落ちた
        #   諦め足踏み・立ち尽くしの均衡に注意すること。変えたら
        #   Episode_Reward/target_reach_velocity と success_ema が落ちていないか確認する。

        # 動きの滑らかさ
        self.rewards.com_jerk_l2.weight = -1.0e-6      # 重心の加加速度
        self.rewards.dof_acc_l2.weight = -1.0e-6       # 関節加速度
        self.rewards.dof_torques_l2.weight = -5.0e-5   # 関節トルク
        self.rewards.ang_vel_xy_l2.weight = -0.25      # ロール/ピッチ角速度
        self.rewards.lin_vel_z_l2.weight = -2.5        # 上下動 (跳躍の抑制。flat 既定 -0.8 から強化)
        self.rewards.feet_slide.weight = -0.5          # 接地足の滑り

        # 姿勢の維持
        self.rewards.flat_orientation_l2.weight = -20.0   # 胴体の傾き
        self.rewards.stance_foot_flat.weight = -8.0       # つま先立ち対策
        self.rewards.joint_deviation_hip.weight = -0.4    # ガニ股矯正
        self.rewards.ankle_deviation.weight = -0.3        # 足首の基準姿勢からの逸脱

        # --- 腰が下がりすぎるのを抑える (2026-07-31) ---
        self.rewards.base_height_penalty.params["min_height"] = 0.55

        # --- タスク報酬を上乗せ (歩容まわりの locomotion 報酬はそのまま残す) ---
        # 速度コマンドはもう外部から来ないので、コマンド追従系は無効化する。
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.track_lin_vel_y = None
        self.rewards.lateral_speed_bonus = None

        # ★ y 方向の dense 報酬 (2026-07-24 追加)。
        self.rewards.target_reach_velocity = RewTerm(
            func=target_reach_velocity_direct,
            weight=4.0,
            params={
                "deadband": 0.12,
                "v_cap": LATERAL_TARGET_SPEED,
                "stop_speed": 0.5,
                "max_y": GOAL_HALF_WIDTH,
            },
        )

        self.rewards.save_touch_bonus = RewTerm(func=save_touch_bonus, weight=100.0)
        # セーブの「質」への上乗せ (2026-07-24)。触れただけで満額だと、ゴール正面に
        self.rewards.save_clearance = RewTerm(func=save_clearance_bonus, weight=50.0)
        # ★ 2026-08-11 (ユーザー指示): セーブ後にゴール中央へ戻る動作を廃止し、
        #   **止めた地点で初期姿勢のまま数秒間立つ** に差し替えた。転倒しないかを
        #   切り分けて確認するのが目的。
        #   保持区間 (touched〜次の球、約3.0s) では task_drive_vector が指令をゼロに
        #   するので歩容が止まり、この報酬が関節を既定姿勢へ引き戻す。
        self.rewards.return_to_center = None
        self.rewards.hold_pose_after_save = RewTerm(
            func=hold_default_pose_after_save, weight=1.0, params={"std": 0.35}
        )
        self.rewards.stay_on_goal_line = RewTerm(func=stay_on_goal_line, weight=1.0, params={"std": 0.3})
        self.rewards.face_field = RewTerm(func=face_field, weight=1.0, params={"std": 0.5})

        # --- 終了条件 ---
        self.terminations.goal_conceded = DoneTerm(func=goal_conceded)
        self.terminations.out_of_bounds = DoneTerm(
            func=robot_out_of_bounds,
            params={"x_range": (-0.1, 2.5), "y_abs_max": 2.2},
        )

        # --- 歩行由来のカリキュラムは無効化 (コマンドが無いので意味を持たない) ---
        self.curriculum.command_resampling_time_range = None
        self.curriculum.lin_vel_command = None
        self.curriculum.push_robot_stage1 = None


@configclass
class K1GKDirectStage2EnvCfg(K1GKDirectEnvCfg):
    """Stage 2 本体: セーブ成功率 (EMA) に応じて難易度を段階的に上げる (2 軸)。

    ★ 2026-07-31: 旧 ``K1GKDirectStage3EnvCfg`` から改名。Stage 2/3 を統合済みで
      「Stage 3」という段は存在しないため、番号とログ出力先を実態に合わせた。
      親の :class:`K1GKDirectEnvCfg` は「ゴール + ボールを置くが難易度は固定」の
      土台で、Play 用 cfg も継承しているので残してある (タスク登録のみ廃止)。

    ★ 2026-07-31: 初速のみを動かす ``adaptive_ball_speed`` から、
      「狙い先の広さ → 初速」の順で上げる ``adaptive_difficulty`` に変更。
      難易度の主因は初速ではなく必要横移動量 (実測: 0.7m で成功率半減) であり、
      その主因を決める aim_y_range が最初から最大値固定だったため、成功率が
      62% 前後で頭打ちになりカリキュラムが不感帯で休眠していた。
      (階層版 goalkeeper_env_cfg.py は従来の adaptive_ball_speed のまま。)
    """

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.difficulty = CurrTerm(func=adaptive_difficulty)


def _make_play_clean(cfg: K1GKDirectEnvCfg) -> None:
    """PLAY 用: 外乱と知覚DR を切って挙動を見やすくする。"""
    cfg.scene.num_envs = 32
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # VirtualPerception + 速度バイアスをクリーン化 (真値・遅延なし・見失いなし)。
    # キーパーの動きそのものを純粋に確認するため。
    cfg.goalkeeper.perception_clean = True


@configclass
class K1GKDirectEnvCfg_PLAY(K1GKDirectEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _make_play_clean(self)
