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
)
from .mdp.rewards import (
    face_field,
    lateral_speed_bonus,
    return_to_center_after_save,
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
        params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True},
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
        },
    )
    # ボール系は知覚DR (レイテンシ/更新レート/ドロップ/距離依存ノイズ) 付きの実値。
    # ノイズは関数内で付加するので ObsTerm 側の noise は付けない。
    ball_pos_rel = ObsTerm(func=gk_ball_pos_rel_perceived)
    ball_vel = ObsTerm(func=gk_ball_vel_perceived)
    ball_active = ObsTerm(func=gk_ball_active)
    target_y = ObsTerm(func=gk_target_y, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": True})
    self_state = ObsTerm(func=gk_self_state, noise=Unoise(n_min=-0.02, n_max=0.02))


@configclass
class K1GKDirectCriticCfg(K1GKDirectStage1CriticCfg):
    """Stage 2/3 の critic 観測 (真値・ノイズなし)。"""

    velocity_commands = ObsTerm(
        func=task_drive_vector, params={"max_y": GOAL_HALF_WIDTH, "use_perceived": False}
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

        # --- 初期配置: 守備面 (ゴールラインの guard_x=0.4m 前) 付近 ---
        self.events.reset_base.params["pose_range"]["x"] = (0.3, 0.5)
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
        self.rewards.foot_clearance.params["cmd_threshold"] = _STOP_TOL
        if self.rewards.feet_phase is not None:
            self.rewards.feet_phase.params["cmd_threshold"] = _STOP_TOL

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
        self.rewards.return_to_center = RewTerm(
            func=return_to_center_after_save, weight=1.0, params={"std": 0.5}
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
