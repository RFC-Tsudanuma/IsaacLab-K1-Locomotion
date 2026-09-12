# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .rough_env_cfg import K1RoughEnvCfg, K1PolicyCfg, K1CriticCfg, _COMMAND_THRESHOLD
from .velocity_env_cfg import CurriculumCfg
from .history_layout import HISTORY_LENGTH
from .mdp.obs_noise_models import SensorArtifactNoiseCfg
import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.events import randomize_phase_freq_offset, randomize_rigid_body_inertia
from .mdp.commands import ExtremeVelocityCommandCfg, LateralVelocityCommandCfg
from .mdp.rewards import (
    feet_landing_impact,
    feet_landing_vel,
    feet_heel_strike,
    com_jerk_l2,
    base_ang_acc_l2,
    joint_power_l2,
    feet_stride_length,
    feet_slide_deadband,
    both_feet_not_in_contact,
)
from .mdp.curriculums import (
    modify_command_resampling_time_range,
    lin_vel_command_curriculum,
    modify_push_robot,
    extreme_command_curriculum,
)


# 段差・坂道なし、ランダムノイズのみの軽く凹凸した地面
NOISY_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
    },
)


# 平面重視版 (2026-09-09): 実機デプロイは完全平面のみという運用実態に合わせ、
# 平面 0.7 / 凹凸 0.3 に反転した地形。凹凸を少し残すのはロバスト性と足上げ高さの保険。
PLANE_HEAVY_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
    },
)


# センサアーティファクト (バイアス/EMA/ホールド) のランダム化スイッチ。
# 実機 dual_first の後退転倒対策として導入したが、これを入れた頑健化 finetune
# (2500it) では転倒が改善しなかった (2026-08-04)。原因は別 (深い履歴の内容が
# CNN の暗黙状態推定を狂わせている疑い、obs_log 解析で調査中) の可能性が高く、
# 推測ベースのノイズは run 比較の交絡源になるためデフォルト無効にする。
# analyze_real_obs.py で実機とシミュの分布差が特定できたら、実測に合わせた
# 範囲に調整して再有効化すること。
_USE_SENSOR_ARTIFACT_NOISE: bool = False

# センサ遅延 DR (2026-08-05)。通信・ドライバ由来の伝送遅延を per-env 0〜20ms で
# ランダム化する (SensorArtifactNoiseModel の遅延ステージのみを使用)。
# _USE_SENSOR_ARTIFACT_NOISE とは独立に有効化でき、対象は joint_pos /
# projected_gravity / base_ang_vel。アーティファクトノイズ有効時はそちらの cfg に統合される。
# 上限 20ms = 制御周期 1 ステップ分 (β=1.0 → 丸ごと前ステップ値)。
_USE_SENSOR_DELAY: bool = True
_SENSOR_DELAY_RANGE: tuple[float, float] = (0.0, 0.020)


# ---------------------------------------------------------------------------
# Observations (履歴バッファ構成)
# ---------------------------------------------------------------------------
# 観測を「最新コマンド」と「観測 (コマンド込み) の履歴」の 3 グループで構成する。
# HistoryActorCritic (agents/history_actor_critic.py) が MLP 入力として
# command + 履歴の直近 MLP_HISTORY_STEPS ステップ分を取り出す。
# レイアウト定義は history_layout.py に集約しており、項の追加・削除時は
# そちらと mdp/symmetry.py も更新すること。


@configclass
class K1FlatCommandCfg(ObsGroup):
    """最新の歩行コマンドのみを持つグループ (履歴なし)。"""

    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1FlatPolicyHistoryCfg(K1PolicyCfg):
    """Actor 用: 観測 (コマンド込み・ノイズあり) を HISTORY_LENGTH ステップ分バッファする。

    2026-08-05: 履歴グループにも velocity_commands を含める仕様に変更
    (4 ステップ MLP 履歴・100 ステップ CNN 履歴の両方にコマンド系列が入る)。
    command グループ (最新値の直接入力) は従来どおり併存する。

    ノイズは ObservationManager が履歴 push 前に適用するため、各ステップの
    ノイズは 1 度だけ引かれて履歴に固定される。

    _USE_SENSOR_ARTIFACT_NOISE が有効な場合のみ、センサ由来の項 (gyro / gravity /
    joint_pos / joint_vel) には白色ノイズに加えて SensorArtifactNoiseCfg (定数
    バイアス + EMA フィルタ + フレームホールド) を適用し、実機で観測される
    「履歴の質の劣化」に頑健にする (mdp/obs_noise_models.py 参照)。
    actions / gait_phase はデプロイ側でも内部生成の正確な値なので白色ノイズのみ・
    アーティファクトなしのまま。

    センサ遅延 DR (_USE_SENSOR_DELAY) は上記フラグと独立: joint_pos /
    projected_gravity / base_ang_vel に per-env 0〜10ms の伝送遅延 (分数ステップ線形補間) を
    入れる。アーティファクト有効時はその cfg に統合、無効時は遅延+白色ノイズ
    のみの SensorArtifactNoiseCfg でラップする。
    """

    def __post_init__(self):
        super().__post_init__()
        # グループレベルの履歴設定は全項に適用される
        self.history_length = HISTORY_LENGTH
        self.flatten_history_dim = True

        # 遅延 DR は _USE_SENSOR_ARTIFACT_NOISE と独立に制御する。
        delay_range = _SENSOR_DELAY_RANGE if _USE_SENSOR_DELAY else (0.0, 0.0)

        if _USE_SENSOR_ARTIFACT_NOISE:
            # 実機センサのアーティファクトのランダム化。白色ノイズ幅は従来の
            # K1PolicyCfg の Unoise と同値を内包させる。
            # - bias_range: 取付誤差・推定器バイアス相当 (gravity ±0.035 ≒ ±2°)
            # - filter_alpha_range: 内蔵 LPF + デプロイ側移動平均相当 (α=0.6 ≒ 5Hz@50Hz)
            # - hold_prob_range: 受信タイミングずれによる重複フレーム相当 (最大 10%)
            # - delay_range: 伝送遅延 (joint_pos / projected_gravity / base_ang_vel、独立フラグ)
            self.base_ang_vel.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
                bias_range=0.05,
                filter_alpha_range=(0.0, 0.6),
                hold_prob_range=(0.0, 0.1),
                delay_range=delay_range,
            )
            self.projected_gravity.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
                bias_range=0.035,
                filter_alpha_range=(0.0, 0.6),
                hold_prob_range=(0.0, 0.1),
                delay_range=delay_range,
            )
            self.joint_pos.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-0.03, n_max=0.03),
                bias_range=0.01,
                filter_alpha_range=(0.0, 0.4),
                hold_prob_range=(0.0, 0.1),
                delay_range=delay_range,
            )
            self.joint_vel.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-1.5, n_max=1.5),
                bias_range=0.2,
                filter_alpha_range=(0.0, 0.6),
                hold_prob_range=(0.0, 0.1),
            )
        elif _USE_SENSOR_DELAY:
            # 遅延のみ有効: 白色ノイズは従来の Unoise と同値を内包し、
            # バイアス/EMA/ホールドは全て無効 (= dual_first + 遅延 のみの差分)。
            self.base_ang_vel.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
                delay_range=delay_range,
            )
            self.projected_gravity.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
                delay_range=delay_range,
            )
            self.joint_pos.noise = SensorArtifactNoiseCfg(
                noise_cfg=Unoise(n_min=-0.03, n_max=0.03),
                delay_range=delay_range,
            )


@configclass
class K1FlatCriticHistoryCfg(K1CriticCfg):
    """Critic 用: 観測 (コマンド込み・ノイズなし・特権情報込み) の履歴。"""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = HISTORY_LENGTH
        self.flatten_history_dim = True


@configclass
class K1FlatObservationsCfg:
    """K1 Flat 環境の観測グループ (command + actor/critic 履歴)。"""

    policy: K1FlatPolicyHistoryCfg = K1FlatPolicyHistoryCfg()
    critic: K1FlatCriticHistoryCfg = K1FlatCriticHistoryCfg()
    command: K1FlatCommandCfg = K1FlatCommandCfg()


@configclass
class K1FlatCurriculumCfg(CurriculumCfg):
    """K1 Flat 環境用のカリキュラム設定。"""

    # ステップ数が5000を超えたら、コマンドのリサンプリング時間分布の範囲を (1.0, 5.0) に変更
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (1.0, 7.0),
            "num_steps": 8000 * 48,
        },
    )

    # より細かいコマンド変動に対応
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (0.5, 7.0),
            "num_steps": 14000 * 48,
        },
    )

    # 線速度コマンド範囲を段階的に拡げるカリキュラム
    # 追従誤差(EMA)が threshold を下回るとステージが進む: ±0.3 → ±0.6 → ±1.0
    lin_vel_command = CurrTerm(
        func=lin_vel_command_curriculum,
        params={
            "command_name": "base_velocity",
            # 上限は ±1.5 m/s にキャップ (2026-07-21)。±1.8 は K1 の脚長では位相通りの
            # 歩容で物理的に届かず、「位相を無視した速い足踏み」で追従する崩れた歩容を
            # 誘発していた。±1.5 に抑えることで位相ロック (一致率 0.78) と速度追従が両立。
            "stages_x": [(-0.6, 0.6), (-1.2, 1.2), (-1.5, 1.5)],
            "stages_y": [(-0.5, 0.5), (-0.7, 0.7), (-0.9, 0.9)],
            # 各ステージを「本物の関門」にするための閾値。広い範囲ほど絶対誤差は出やすいので
            # わずかに緩めるが、緩めすぎると「狭い範囲を習得した時点で広い範囲のゆるい閾値も
            # 満たしてしまい」0→1→2 と一気に遷移する。実測では stage0(±0.6)の到達誤差が ~0.30、
            # その直後の ±1.2 での誤差が ~0.43、±1.8 で ~0.75。旧設定 [0.30, 0.60, 0.55] は
            # stage1/2 の閾値が「到達済みの誤差」より緩く、ゲートとして機能していなかった。
            # そこで stage1 は ±1.2 でまだ達成していない 0.34 まで締めて再学習を要求する
            # (stage0=0.30 は約500iter かけて到達する適切なゲートなので維持。
            #  最終 stage2 の値は遷移判定に使われずログ表示専用)。
            "error_threshold": [0.30, 0.39, 0.43],
            "asset_name": "robot",
            "ema_alpha": 0.026,
            "min_updates": 50,
            # ステージを進めた直後、新しい(広い)コマンド範囲が全 env に行き渡るまで
            # 誤差計測を止めて次の遷移判定を待つ。これが無いと、各 env がまだ旧範囲の
            # コマンドを保持したまま EMA が低いため、緩い次ステージ閾値を即満たして
            # 0→1→2 と一気に遷移してしまう。resampling_time_range の最大値の倍数で指定。
            "stage_cooldown_resamples": 1.5,
            # 切替直後は EMA を「閾値 × post_switch_ema_scale」で固定し、この最小ステップ数の間は
            # 計測・更新・判定を止める。hold 明けも高い値から減衰させることで、運良く低い誤差を
            # 1 回引いただけで即次ステージへ進む(一気な遷移)のを確実に防ぐ。
            "post_switch_hold_steps": 500,
            "post_switch_ema_scale": 2.0,
        },
    )

    # 学習後半に「レンジ端の組み合わせコマンド」を段階導入するカリキュラム。
    # 上位の最適制御プランナはコマンド範囲の上限付近を多用するが、一様サンプリングでは
    # vx・vy 同時上限 + 大旋回のような corner がほぼ出ず、その領域や極端なコマンド間の
    # 遷移で転倒していた。extreme_prob の確率で「|v| ∈ [0.7*max, max] (符号ランダム) +
    # 飽和 yaw を作る heading」を引かせ、ランプ完了後はリサンプリングも速めて
    # 極端コマンド間の遷移にも晒す (ExtremeVelocityCommand 参照)。
    # num_steps / ramp_steps は「学習イテレーション数」基準 (--max_iterations と同じ単位)。
    # 関数内部で common_step_counter ÷ 48 (rsl_rl_ppo_cfg の num_steps_per_env) で換算する。
    # NOTE: resume 時は 0 から数え直すので、即時有効化するなら num_steps を override で 0 に。
    extreme_commands = CurrTerm(
        func=extreme_command_curriculum,
        params={
            "command_name": "base_velocity",
            "num_steps": 10000,    # このイテレーション数を超えたら extreme 導入開始
            "ramp_steps": 2000,    # このイテレーション数かけて 0 → extreme_prob へ線形導入
            "extreme_prob": 0.35,
            "extreme_frac": 0.7,
            "resampling_time_range": (0.8, 8.0),
        },
    )

    # push_robot を段階的に強くするカリキュラム
    # 初期値 (EventCfg): interval 7-10s, vel ±0.5 → ±0.5
    push_robot_stage1 = CurrTerm(
        func=modify_push_robot,
        params={
            "term_name": "push_robot",
            "num_steps": 6000 * 48,
            "interval_range_s": (4.0, 8.0),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02)},
        },
    )
    # push_robot_stage2 = CurrTerm(
    #     func=modify_push_robot,
    #     params={
    #         "term_name": "push_robot",
    #         "num_steps": 16000,
    #         "interval_range_s": (3.0, 8.0),
    #         "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3)},
    #     },
    # )

@configclass
class K1FlatEnvCfg(K1RoughEnvCfg):
    curriculum: K1FlatCurriculumCfg = K1FlatCurriculumCfg()
    # 観測を「最新コマンド + 履歴バッファ」構成に置き換える (上記参照)
    observations: K1FlatObservationsCfg = K1FlatObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 地面との摩擦ランダム化を広げる (2026-08-04): (0.4-0.8 / 0.2-0.6) → 0.3〜1.5。
        # 地形マテリアルは摩擦 1.0 × multiply 結合なので、足側マテリアルの値が
        # そのまま実効摩擦になる。make_consistent=True で dynamic ≤ static を保証。
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["make_consistent"] = True

        # --- リンク物性ランダム化 (2026-08-04) ---
        # 各リンクの質量を 0.5〜1.5 倍でランダム化 (startup で env 毎に1度)。
        # recompute_inertia=True で慣性も質量比に追従 (default × 質量比)。
        # NOTE: 標準の randomize_rigid_body_mass は default 値基準で書き直すため、
        #       Trunk だけの add_base_mass (±1.5kg) はこの全リンクスケールに上書きされて
        #       意味を失う。×0.5〜1.5 は ±1.5kg より広い DR なので add_base_mass は無効化。
        self.events.add_base_mass = None
        self.events.randomize_link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        # 各リンクの慣性を質量とは独立に 0.7〜1.3 倍でランダム化 (自作イベント)。
        # 上の mass ランダム化の後に実行され合成される (最終慣性 = default × 質量比 × 本倍率)。
        self.events.randomize_link_inertia = EventTerm(
            func=randomize_rigid_body_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "inertia_distribution_params": (0.9, 1.1),
            },
        )

        # 環境毎に歩行周波数オフセットを ±0.05 Hz の範囲でランダム化 (startup で1度だけ)。
        # 基本周波数はコマンド速度に応じて線形遷移し (rough_env_cfg._PHASE_FREQ_PARAMS 参照)、
        # このオフセットがそれに常時加算される。phase_obs / feet_phase が自動で参照する。
        self.events.randomize_phase_freq = EventTerm(
            func=randomize_phase_freq_offset,
            mode="startup",
            params={
                "offset_range": (-0.05, 0.05),
            },
        )

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 軽い凹凸のみの地面 (段差・坂道なし)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = NOISY_FLAT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        # No height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # Flat では脚部の接触ペナルティ (undesired_contacts) は不要なので削除する。
        # これにより接触センサで読む必要があるのは足 (.*_foot_link: 着地系報酬/air_time) と
        # 胴体 (Trunk: base_contact 終了判定) だけになるので、センサも 2 部位に絞り収集を更に軽くする。
        # NOTE: rough は undesired_contacts (股関節/すね) を使うため velocity_env_cfg 側のセンサ
        #       (足+股関節+すね+胴体) は据え置き、ここ (flat) でのみ上書きする。
        # NOTE: dribble の足-ボール接触は専用センサ (contact_balls_left/right, SoccerBall フィルタ)
        #       を使っており、この contact_forces とは独立なので影響しない。
        self.rewards.undesired_contacts = None
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/(Trunk|.*_foot_link)"

        # Rewards
        # 速度追従の「粗い」項を追加する
        # 既存の track_lin_vel_xy_exp は std=0.25 と鋭く、誤差が ~0.4 m/s を超えると
        # exp(-err²/std²) が飽和して勾配が消える。これにより速度コマンドのカリキュラム上端
        # (±1.8 など) でロボットが追従を諦め、その場足踏みの局所最適に落ちていた。
        # 鋭い項 (重み 3.5) はそのまま残しつつ、std を広げた同じ報酬を小さい重みで加算する。
        # 誤差 0.8 m/s でも exp(-0.64/0.36)=0.17 と勾配が残り「もっと速く」の信号が生きる一方、
        # 誤差が小さい領域では鋭い項が支配して追従精度を保つ。
        # 重みはコマンド依存位相周波数の導入時に 15 サイクルのチューニングで決定 (2026-07)。
        # 目的: track_lin_vel_xy_coarse と track_ang_vel_z_exp の正規化スコア (÷weight) を
        # 両立させ調和平均を最大化。3seed 検証で (sharp, coarse, ang) = (1.5, 2.4, 4.2) が
        # 平均 0.554 / 最悪 0.534 でベスト。coarse↑はカリキュラム最終段階 (±1.8 m/s) 到達に
        # 必須、sharp は 1.5 未満に下げるとカリキュラムが進まない、ang は coarse に対し
        # 比率 ~1.75 を外れるとどちらかが崩れる (ang=3.8/coarse=2.0 で lin 崩壊を確認)。
        self.rewards.track_lin_vel_xy_coarse = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=2.4,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        # 4.2→4.8 (2026-07-21 h06): 旋回重視の要望に伴い増量。±1.5 キャップ+位相 w1.5 の
        # 構成では lin と競合せず err_yaw 0.82→0.76 に改善。
        self.rewards.track_ang_vel_z_exp.weight = 4.8
        # 上体の傾き抑制 (2026-07-28〜29 l/m系列): flat_orientation_l2 はデフォルト -20
        # (rough_env_cfg) のまま。-40 をフルパイプラインに入れると extreme 期に lin 追従が
        # 崩壊する (m01: -40固定, m02: extreme期に-20へ緩和するカリキュラム、いずれも失敗)。
        # 実証済みレシピは「フルパイプライン 20000it (このconfigのまま) → extreme無効の
        # 通常分布で flat_orientation_l2=-40 を override して +2500it の仕上げ学習」(m04):
        #   傾き -34%、lin/位相/振動/extreme頑健性は全て維持。詳細は
        #   scripts/rsl_rl/phase_freq_weight_tuning.md の l/m 系列を参照。

        # 上体(頭部)振動抑制 (2026-07-24〜25 j/k系列): 頭は Trunk 剛結合のため、Trunk の
        # roll/pitch 回転が頭の振動の支配項 (振動proxy 0.51 (rad/s)² で lin_vel_z の20倍)。
        # NOTE: -0.75 (j02) は 5000it 検証では追従低下ノイズ内だったが、フルパイプライン
        # (k01) では extreme 期に corner コマンドの激しい体幹運動とペナルティが競合し、
        # lin 追従が漸減崩壊 (0.74→0.53)。-0.5 + base_ang_acc_l2 の組み合わせ (j03 相当)
        # に緩めて追従と両立させる。
        self.rewards.ang_vel_xy_l2.weight = -0.5
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.action_rate_l2.weight = -0.5
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Ankle_.*"]
        )
        # 重心(全身CoM)位置の jerk ペナルティ: CoM 速度の二階差分 (≒躍度) の二乗ノルムを罰する。
        # 体重移動の急変(カクつき)を抑え、滑らかな重心移動を促す。
        # jerk は dt² で割るため値が大きくなりやすい。重みは dof_acc_l2 (-1e-6) と同程度の桁から開始し、
        # reward logger で他項と桁を合わせて要チューニング。
        self.rewards.com_jerk_l2 = RewTerm(
            func=com_jerk_l2,
            weight=-1.0e-6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # base 高さの目標追従ペナルティ (上下動抑制, 2026-08-01 q系列)。
        # lin_vel_z (速度ペナルティ) は歩容の構造的下限 0.0226 で飽和し増量無効だったため、
        # 高さ「位置」の偏差を直接罰して vaulting の振幅自体を縮める。
        # weight=0.0 でパイプラインでは無効。仕上げ resume で override して使う
        # (その際 minimum_height の閾値 0.54 が目標 0.53 と干渉するので
        #  rewards.base_height_penalty.params.min_height=0.48 も同時に override すること)。
        self.rewards.base_height_track = RewTerm(
            func=mdp.base_height_l2,
            weight=0.0,
            params={"target_height": 0.53},
        )
        # Trunk 角加速度ペナルティ (頭部振動抑制)。頭は Trunk 剛結合なので、頭の振動 =
        # Trunk の回転ジッタ。ang_vel_xy_l2 が取りこぼす高周波成分を抑える。
        # j01 (この項なし) は追従 0.577 まで低下、j02 (あり) は 0.591 と同じ振動低減で
        # 追従を保った → 減衰を「角加速度側」で受け持たせる方が追従と両立する。
        self.rewards.base_ang_acc_l2 = RewTerm(
            func=base_ang_acc_l2,
            weight=-1.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # ガニ股対策 (2026-07-13): 左右 Hip_Yaw が外向きに開いた歩容になったため、
        # Hip_Yaw の偏差を独立項に分離して強く罰する。joint_deviation_hip (rough 側で
        # Yaw+Roll 合算 weight=-0.10) は Roll のみに変更。旧挙動は
        # joint_deviation_hip_yaw.weight=-0.10 と等価なので、重みだけで新旧比較できる。
        self.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_Roll"]
        )
        # weight は -0.1/-0.5/-1.0/-2.0 の比較で決定: -1.0 で Σ|yaw| 0.33→0.107 rad
        # (67%減) かつ追従スコアはむしろ向上。-2.0 は lin 追従が崩れるため過剰。
        self.rewards.joint_deviation_hip_yaw = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw"])},
        )

        # 速度コマンドは連続一様サンプリング + extreme corner サンプリング対応版。
        # extreme_prob=0.0 (既定) では UniformVelocityCommand と同一挙動で、
        # extreme_commands カリキュラムが学習後半 (10000 iter〜) に prob を上げる。
        # lin_vel_x / lin_vel_y の範囲は lin_vel_command カリキュラムが段階的に拡張する。
        prev = self.commands.base_velocity
        self.commands.base_velocity = ExtremeVelocityCommandCfg(
            asset_name=prev.asset_name,
            resampling_time_range=prev.resampling_time_range,
            rel_standing_envs=prev.rel_standing_envs,
            rel_heading_envs=prev.rel_heading_envs,
            heading_command=prev.heading_command,
            heading_control_stiffness=prev.heading_control_stiffness,
            debug_vis=prev.debug_vis,
            ranges=ExtremeVelocityCommandCfg.Ranges(
                lin_vel_x=prev.ranges.lin_vel_x,
                lin_vel_y=prev.ranges.lin_vel_y,
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )

@configclass
class K1FlatGoalkeeperCfg(K1FlatEnvCfg):
    """横移動特化 (ゴールキーパー) 用の FlatEnv 派生設定 (2026-08-17)。

    ゴール前の横っ飛びの代わりに「速いサイドステップ」で守るキーパー用ポリシーを
    作る。観測レイアウトは K1FlatEnvCfg と完全に同一 (policy 入力の変更なし) なので、
    既存の C++ デプロイスタックとモデル形状をそのまま使える。

    通常 FlatEnv との違い:

    * 速度カリキュラム: x は全ステージ ±0.7 固定、y を ±0.9 → ±1.2 → ±1.5 と拡張
      (通常版と x/y の役割を反転)。学習済み Flat ポリシー (y ±0.9 習得済み) からの
      warm-start を前提に stage0 を ±0.9 から始める。
    * コマンドサンプリング: LateralVelocityCommand で確率 lateral_prob により
      「|vy| 上端域 + vx 縮小域」の横重視コマンドを混ぜる。
    * extreme corner カリキュラム: warm-start 前提で 5000 iter から導入 (通常版 10000)。
    * Hip_Roll 偏差ペナルティ緩和: 横 1.5 m/s のサイドステップは股関節 roll の
      大きな外転を要するため -0.10 → -0.02 に弱める (Hip_Yaw の -1.0 は据え置き、
      足先はゴールライン正面向きを保つ)。

    使い方 (既存 Flat ポリシーから warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Goalkeeper --headless --distributed \\
            --num_envs 2048 --resume --load_run <既存run名> --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 速度カリキュラム: y 主体に置き換え ---
        # 閾値は通常版の実測知見 (±0.9 到達誤差 ~0.30 / 拡張直後 +0.1 程度) を流用。
        # 最終ステージの値は遷移判定に使われずログ表示専用。
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-0.7, 0.7),
            (-0.7, 0.7),
            (-0.7, 0.7),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.9, 0.9),
            (-1.2, 1.2),
            (-1.5, 1.5),
        ]
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.30, 0.38, 0.45]

        # --- コマンドを横重視サンプリング版に差し替え ---
        # 4 割の resample で |vy| ∈ [0.6*max, max] + vx 縮小域を引く。残り 6 割は
        # 通常の一様 (+学習後半は extreme corner) サンプリングで全域をカバーする。
        prev = self.commands.base_velocity
        self.commands.base_velocity = LateralVelocityCommandCfg(
            asset_name=prev.asset_name,
            resampling_time_range=prev.resampling_time_range,
            rel_standing_envs=prev.rel_standing_envs,
            rel_heading_envs=prev.rel_heading_envs,
            heading_command=prev.heading_command,
            heading_control_stiffness=prev.heading_control_stiffness,
            debug_vis=prev.debug_vis,
            ranges=LateralVelocityCommandCfg.Ranges(
                lin_vel_x=prev.ranges.lin_vel_x,
                lin_vel_y=prev.ranges.lin_vel_y,
                ang_vel_z=prev.ranges.ang_vel_z,
                heading=prev.ranges.heading,
            ),
            lateral_prob=0.4,
            lateral_frac=0.6,
            lateral_x_scale=0.4,
        )
        # warm-start なので最初から短めの再サンプリング周期でキーパー的な
        # 反応 (コマンド急変) に晒す。段階短縮カリキュラムは不要なので無効化。
        self.commands.base_velocity.resampling_time_range = (1.0, 7.0)
        self.curriculum.command_resampling_time_range = None

        # --- extreme corner: warm-start 前提で早めに導入 ---
        # x±0.7 + y±1.5 同時上端のような corner はキーパーの主要動作なので
        # 5000 iter から 2000 iter かけて導入する (num_steps/ramp_steps は iteration 基準)。
        self.curriculum.extreme_commands.params["num_steps"] = 5000
        self.curriculum.extreme_commands.params["ramp_steps"] = 2000

        # --- Hip_Roll 偏差ペナルティ緩和 (横ステップの外転を許す) ---
        self.rewards.joint_deviation_hip.weight = -0.02


@configclass
class K1FlatFastCfg(K1FlatEnvCfg):
    """通常歩行の高速実験版: x ±1.8 m/s + 最大ケイデンス 2.5 Hz (2026-08-20)。

    通常版 FlatEnv は「2.0 Hz ケイデンスでは ±1.8 m/s に脚長的に歩幅が届かず、
    位相を無視した崩れた歩容になる」ため x を ±1.5 にキャップした経緯がある
    (rough_env_cfg の lin_vel_command カリキュラム参照)。ゴールキーパー高速版
    (K1FlatGoalkeeperFastCfg, y ±1.8 + 2.5 Hz) で「ケイデンスを上げて歩数で稼ぐ」
    戦略が機能したので、同じ手法を前後方向に適用して ±1.8 m/s を再挑戦する。

    K1FlatEnvCfg との違い:

    * 位相周波数マッピング: 高速側を 2.0 → 2.5 Hz に引き上げ (1.0 m/s 以下 1.8 Hz は
      共通、1.8 m/s で 2.5 Hz 到達、以降同傾きで外挿)。obs (policy/critic) と
      feet_phase 報酬の 3 箇所を同時に上書きし、位相積分の整合を保つ。
    * 速度カリキュラム: x ±1.5 → ±1.8 の 1 段拡張 (±1.5 習得済みポリシーからの
      warm-start 前提)。y は ±0.9 固定のまま。
    * extreme corner: 既存ポリシーが ±1.5 で頑健化済みの warm-start 前提で
      3000 iter から導入。
    * 再サンプリング周期は最初から (1.0, 7.0) の短周期 (段階短縮カリキュラム不要)。

    使い方 (既存 Flat ポリシーから warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Fast --headless --distributed \\
            --num_envs 2048 --resume --load_run <既存run名> --reset_noise_std 0.05

    NOTE: デプロイ時は C++ 側 cmd_phase_freq() の高速側周波数も 2.5 Hz に
    合わせること (k1_constants_isaaclab.hpp)。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 位相周波数マッピング: 高速側 2.5 Hz ---
        # 位相アキュムレータは obs/reward のどちらが先に呼ばれても同じ周波数で
        # 積分されるよう、params を持つ全 3 項を同じ値で上書きする。
        fast_phase_freq = {
            "low_speed": 1.0,
            "high_speed": 1.8,
            "low_freq": 1.8,
            "high_freq": 2.5,
        }
        self.observations.policy.gait_phase.params.update(fast_phase_freq)
        self.observations.critic.gait_phase.params.update(fast_phase_freq)
        self.rewards.feet_phase.params.update(fast_phase_freq)

        # --- 速度カリキュラム: x ±1.5 開始 → ±1.8 (y は ±0.9 固定) ---
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-1.5, 1.5),
            (-1.8, 1.8),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.9, 0.9),
            (-0.9, 0.9),
        ]
        # stage0 閾値は ±1.5 習得済み実測 (~0.43)。最終値はログ表示専用。
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.43, 0.55]

        # --- extreme corner: warm-start 前提で前倒し ---
        self.curriculum.extreme_commands.params["num_steps"] = 3000
        self.curriculum.extreme_commands.params["ramp_steps"] = 1500

        # --- 再サンプリング周期: 最初から短周期 ---
        self.commands.base_velocity.resampling_time_range = (1.0, 7.0)
        self.curriculum.command_resampling_time_range = None


@configclass
class K1FlatFast2Cfg(K1FlatFastCfg):
    """通常歩行のさらなる高速版: x ±2.0 m/s + 最大ケイデンス 2.8 Hz (2026-08-24)。

    K1FlatFastCfg (±1.8 / 2.5 Hz, run 2026-08-20_08-58-47) で「ケイデンス引き上げで
    歩数を稼ぐ」戦略が ±1.8 で機能した (err 1.00) ので、同じ手法をもう 1 段押し進める。

    K1FlatFastCfg との違い:

    * 位相周波数マッピング: 1.0 m/s 以下 1.8 Hz は共通、高速側アンカーを
      1.8 m/s→2.5 Hz から 2.0 m/s→2.8 Hz へ変更 (中間 1.8 m/s では 2.6 Hz と
      旧マッピングより +0.1 Hz、以降同傾き 1.0 Hz/(m/s) で外挿)。
    * 速度カリキュラム: x ±1.8 開始 → ±2.0 の 1 段拡張 (±1.8 習得済みポリシー
      2026-08-20_08-58-47/model_44994 からの warm-start 前提)。y は ±0.9 固定。

    使い方 (±1.8 高速ポリシーから warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Fast2 --headless --distributed \\
            --num_envs 2048 --resume --load_run 2026-08-20_08-58-47 \\
            --checkpoint model_44994.pt --reset_noise_std 0.05

    NOTE: デプロイ時は C++ 側 cmd_phase_freq() の高速側アンカーも
    (2.0 m/s, 2.8 Hz) に合わせること (k1_constants_isaaclab.hpp)。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 位相周波数マッピング: 高速側 2.0 m/s で 2.8 Hz ---
        fast2_phase_freq = {
            "low_speed": 1.0,
            "high_speed": 2.0,
            "low_freq": 1.8,
            "high_freq": 2.8,
        }
        self.observations.policy.gait_phase.params.update(fast2_phase_freq)
        self.observations.critic.gait_phase.params.update(fast2_phase_freq)
        self.rewards.feet_phase.params.update(fast2_phase_freq)

        # --- 速度カリキュラム: x ±1.8 開始 → ±2.0 (y は ±0.9 固定) ---
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-1.8, 1.8),
            (-2.0, 2.0),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.9, 0.9),
            (-0.9, 0.9),
        ]
        # stage0 閾値は ±1.8 習得済み実測 (~0.55)。最終値はログ表示専用。
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.55, 0.65]


@configclass
class K1FlatStrideCfg(K1FlatEnvCfg):
    """歩幅誘導で高速化する通常歩行 (2026-08-25): 周期は通常 (最大 2.0 Hz) のまま x ±1.8 m/s。

    2.5 Hz 以上のケイデンスは実機では無理があるため (ユーザー判断)、周波数マッピングは
    K1FlatEnvCfg のまま変えず、歩幅を伸ばす方向で ±1.8 m/s を狙う。

    K1FlatEnvCfg との違い:

    * ``feet_stride_length`` 報酬 (mdp/rewards.py): コマンド速度と位相周波数から決まる
      目標歩幅 L = |v|/(2f) に、コマンド方向の左右足距離を位相追従させる。
      速度ゲート 0.6→1.0 m/s で低速歩容には影響させない。位相パラメータは
      obs / feet_phase と同一 (_PHASE_FREQ_PARAMS 経由)。
    * ``base_height_penalty`` の閾値 0.53 → 0.50: 歩幅が伸びると両脚支持期に CoM が
      必ず沈むため、既定の床が歩幅拡張を直接罰してしまう。
    * 速度カリキュラム x ±1.5 → ±1.8 (K1FlatFastCfg と同じ 1 段拡張、±1.5 習得済み
      ポリシーからの warm-start 前提)、extreme 3000/1500、再サンプリング (1.0, 7.0)。
    * Metrics/base_velocity/stride_touchdown (着地時実測歩幅) で効果を検証する。

    使い方 (2.0 Hz 世代の Flat ポリシーから warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Stride --headless --distributed \\
            --num_envs 2048 --resume --load_run 2026-08-10_03-07-48 \\
            --checkpoint model_34995.pt --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 歩幅誘導報酬 ---
        # プローブ結果 (2026-08-25, model_34995 から 300-600it):
        #   * σ=0.08 では現状歩幅 (0.17m) が目標 (0.39m@1.5m/s) から遠すぎて勾配ゼロ → σ=0.25。
        #   * weight 0.6/2.0/4.0/8.0 単独では歩幅 +12%/600it の漸進のみ。
        #   * 直接計測 (measure_stride.py) で、ベース方策は 1.8m/s 指令時に実速度の半分以上を
        #     接地中の足の滑り (0.38m/s) で稼いでいると判明 → feet_slide を -0.5→-8.0 に
        #     強化 (下記) してチートの旨味を消すと、600it で歩幅 +11%/滑り -31%/実速度 +5%。
        self.rewards.feet_stride_length = RewTerm(
            func=feet_stride_length,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "sigma": 0.25,
                "gate_low_speed": 0.6,
                "gate_high_speed": 1.0,
                **{k: self.rewards.feet_phase.params[k] for k in ("low_speed", "high_speed", "low_freq", "high_freq")},
            },
        )

        # --- 足の滑り (スケーティング) 抑制: デッドバンド版に置換 (2026-08-31) ---
        # 経緯: 素の feet_slide -8.0 は extreme 導入前は最良 (滑り -48%) だが extreme 導入後
        # に 1.8m/s 追従を放棄 (実速度 0.76)。-4.0 は追従維持で滑り -35% (2026-08-30)。
        # ただし一律罰は着地・蹴り出しの自然な足の動きにも課金され「接地を短くする」圧に
        # なり歩幅を妨げる (ユーザー観察: 大股歩行は接地時間が長い) ため、閾値 0.15 m/s
        # 以下を無罰にするデッドバンド版へ置換。重みは本物のスケーティングに対して強め。
        self.rewards.feet_slide = RewTerm(
            func=feet_slide_deadband,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
                "v_thresh": 0.15,
            },
        )

        # --- 大歩幅時の CoM 沈み込みを許容 ---
        self.rewards.base_height_penalty.params["min_height"] = 0.50

        # --- 速度カリキュラム: x ±1.5 開始 → ±1.8 (y は ±0.9 固定) ---
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-1.5, 1.5),
            (-1.8, 1.8),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.9, 0.9),
            (-0.9, 0.9),
        ]
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.43, 0.55]

        # --- extreme corner: warm-start 前提で前倒し ---
        self.curriculum.extreme_commands.params["num_steps"] = 3000
        self.curriculum.extreme_commands.params["ramp_steps"] = 1500

        # --- 再サンプリング周期: 最初から短周期 ---
        self.commands.base_velocity.resampling_time_range = (1.0, 7.0)
        self.curriculum.command_resampling_time_range = None


@configclass
class K1FlatStanceCfg(K1FlatEnvCfg):
    """接地重視版 (2026-09-04): stance_ratio 拡大 + 両足空中の禁止で速度追従率を最大化する。

    経緯: 歩幅誘導路線 (K1FlatStrideCfg) は歩幅 +17% を達成したが上体の揺れが増え、
    ユーザー判断で放棄。odometry の関係で空中区間 (フライト期) は不要どころか有害。
    直接計測ではベース方策も高速域で遊脚 0.32s/周期 0.5s の実質ランニングになっていた。

    K1FlatEnvCfg との違い:

    * ``feet_phase`` の stance_ratio 0.50 → 0.55: 位相スケジュール上、両足接地の
      重なり (double support) が各歩に必ず入り、フライト期がスケジュールから消える。
    * ``no_fly`` 報酬 (both_feet_not_in_contact): 両足同時空中を直接罰する。
      NOTE: 関数が内部で -1 を返すので weight は正 (+1.0) がペナルティになる。
    * extreme corner 3000/1500・再サンプリング (1.0, 7.0) は warm-start 前提の
      最近の系譜と同じ。速度カリキュラムはベースのまま (±1.5、歩幅拡大は目的外)。

    使い方 (ベース系譜 model_34995 から warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Stance --headless --distributed \\
            --num_envs 2048 --resume --load_run 2026-08-10_03-07-48 \\
            --checkpoint model_34995.pt --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 接地時間の拡大: stance_ratio 0.50 → 0.60 ---
        # 1000it プローブ (2026-09-04, 34995 起点・extreme 有効): 0.55/0.60 とも err 0.95 で
        # 追従は同等、フライト削減は 0.60 が上 (fly% 9.4 vs 6.6 @1.5m/s) → 0.60 採用。
        self.rewards.feet_phase.params["stance_ratio"] = 0.60

        # --- 両足空中 (フライト期) の直接禁止 ---
        # weight は追従とのトレードオフ (1000it プローブ, fly% は 1.5m/s 指令の実測):
        #   w1: err 0.95 / fly 6.6%、w4: err 1.00 / fly 2.8%、w8: err 1.07 / fly 0.9%。
        # 追従優先の方針 (ユーザー指示) で w4 を採用。長回しで fly はさらに漸減する傾向。
        self.rewards.no_fly = RewTerm(
            func=both_feet_not_in_contact,
            weight=4.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")},
        )

        # --- extreme corner: warm-start 前提で前倒し ---
        self.curriculum.extreme_commands.params["num_steps"] = 3000
        self.curriculum.extreme_commands.params["ramp_steps"] = 1500

        # --- 再サンプリング周期: 最初から短周期 ---
        self.commands.base_velocity.resampling_time_range = (1.0, 7.0)
        self.curriculum.command_resampling_time_range = None


@configclass
class K1FlatStanceFastCfg(K1FlatStanceCfg):
    """接地重視版の x ±1.8 m/s 拡張 (2026-09-07)。

    K1FlatStanceCfg (sr0.60 + no_fly4, 採用 2026-09-04_13-50-47/model_41000) の設定は
    そのままに、速度カリキュラムだけ x ±1.5 → ±1.8 の 1 段拡張を足す。周期は通常
    マッピング (最大 2.0 Hz) のまま。フライト禁止下では速度 ≈ 歩幅×ケイデンスに
    上限があるため、±1.8 にどこまで届くかは実測で判断する。

    使い方 (Stance 採用 ckpt から warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Stance-Fast --headless --distributed \\
            --num_envs 2048 --resume --load_run 2026-09-04_13-50-47 \\
            --checkpoint model_41000.pt --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 速度カリキュラム: x ±1.5 開始 → ±1.8 (y は ±0.9 固定) ---
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-1.5, 1.5),
            (-1.8, 1.8),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.9, 0.9),
            (-0.9, 0.9),
        ]
        # stage0 閾値は ±1.5 習得済み実測 (extreme 下 err ~0.95)。最終値はログ表示専用。
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.43, 0.55]


@configclass
class K1FlatStancePlaneCfg(K1FlatStanceCfg):
    """接地重視版の平面特化・±2.0 m/s 挑戦 (2026-09-09, 案1+案2)。

    実機デプロイは完全平面のみという運用実態に基づき、学習分布をデプロイに寄せて
    保守性の分の余力を最高速度に回す。±1.5 学習の Stance 採用ポリシー (model_41000)
    は実機/mujoco の平面では ~1.8 m/s まで追従できており、sim の過酷分布
    (凹凸 0.7 + 摩擦 0.3〜) が上限を下げていた。

    K1FlatStanceCfg との違い:

    * 地形: 凹凸 0.7/平面 0.3 → **平面 0.7/凹凸 0.3** (凹凸は頑健性と足上げの保険)。
    * 摩擦 DR: (0.3, 1.0) → **(0.6, 1.0)** (実機の乾いた平面相当に整合)。
    * 速度カリキュラム: x ±1.5 → ±1.8 → ±2.0 の 2 段拡張。
    * 高速域限定の保守ペナルティ緩和 (速度ゲート 1.5→1.8 m/s):
      - base_height_penalty の床 0.53 → 高速時 0.48 (大股時の CoM 沈み許容)
      - feet_parallel_to_ground を高速時 0.5 倍に減衰 (蹴り出しのつま先角度許容)
    * クリアランス系は feet_air_time のみ (foot_clearance_ji は不採用: ユーザー方針)。

    使い方 (Stance 採用 ckpt から warm-start)::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Stance-Plane --headless --distributed \\
            --num_envs 2048 --resume --load_run 2026-09-04_13-50-47 \\
            --checkpoint model_41000.pt --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 地形: 平面重視 ---
        self.scene.terrain.terrain_generator = PLANE_HEAVY_TERRAIN_CFG

        # --- 摩擦 DR: 実機の平面に整合 ---
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.0)

        # --- 速度カリキュラム: x ±1.5 → ±1.8 → ±2.0 (y は ±0.9 固定) ---
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-1.5, 1.5),
            (-1.8, 1.8),
            (-2.0, 2.0),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-0.9, 0.9),
            (-0.9, 0.9),
            (-0.9, 0.9),
        ]
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.43, 0.55, 0.65]

        # --- 高速域限定の保守ペナルティ緩和 ---
        self.rewards.base_height_penalty.params.update({
            "min_height_high": 0.48,
            "gate_low_speed": 1.5,
            "gate_high_speed": 1.8,
        })
        self.rewards.feet_parallel_to_ground.params.update({
            "gate_low_speed": 1.5,
            "gate_high_speed": 1.8,
            "gate_high_scale": 0.5,
        })


@configclass
class K1FlatStancePlanePolishCfg(K1FlatStancePlaneCfg):
    """Stance-Plane ポリシー (2026-09-09_18-32-40/model_50999) 用の省エネ・姿勢仕上げ (2026-09-11)。

    K1FlatImprovePostureCfg の仕上げレシピを Stance-Plane 環境 (平面 0.7・摩擦 0.6-1.0・
    stance_ratio 0.60・no_fly・±2.0) の上に移植したもの。旧 Posture タスクをそのまま使うと
    接地スケジュール・地形分布・速度レンジが食い違い、方策の核を壊すため専用化する。

    K1FlatStancePlaneCfg との違い:

    * joint_power_l2 追加: ±1.5 調整値 -3e-5 はパワー∝速度² で ±1.8 実効約 2 倍だった
      教訓 (2026-08-22) に従い、±2.0 では -1.5e-5 を既定とする (プローブで確認)。
    * flat_orientation_l2 -20 → -30: 仕上げ resume での増量は安全実績あり (-40 は
      ±1.8 高速版で追従を落とさなかったが効果も薄かったため中間の -30)。
    * カリキュラムを学習終了時点の状態に固定 (resume で 0 から再進行させない):
      lin_vel ±2.0/±0.9 固定、extreme 即時 α=1、push は stage1 相当の強状態。

    使い方::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Stance-Plane-Polish --headless --distributed \\
            --num_envs 2048 --resume --load_run 2026-09-09_18-32-40 \\
            --checkpoint model_50999.pt --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 機械パワー (torque * joint_vel)² の抑制 ---
        self.rewards.joint_power_l2 = RewTerm(
            func=joint_power_l2,
            weight=-1.5e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # --- 上体の傾き抑制 (仕上げ増量) ---
        self.rewards.flat_orientation_l2.weight = -30.0

        # --- カリキュラムを学習終了時点の状態に固定 ---
        self.curriculum.lin_vel_command = None
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.9, 0.9)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.curriculum.extreme_commands.params["num_steps"] = 0
        self.curriculum.extreme_commands.params["ramp_steps"] = 1
        self.commands.base_velocity.resampling_time_range = (0.8, 8.0)

        self.curriculum.push_robot_stage1 = None
        self.events.push_robot.interval_range_s = (0.5, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.7, 0.7),
            "y": (-0.7, 0.7),
            "roll": (-0.06, 0.06),
            "pitch": (-0.06, 0.06),
        }


@configclass
class K1FlatGoalkeeperFastCfg(K1FlatGoalkeeperCfg):
    """ゴールキーパーの高速実験版: y ±1.8 m/s + 最大ケイデンス 2.5 Hz (2026-08-18)。

    通常版 FlatEnv では ±1.8 m/s は「2.0 Hz ケイデンスでは脚長的に位相通りに届かず
    崩れた歩容になる」ため ±1.5 にキャップした経緯がある (rough_env_cfg 参照)。
    本設定はケイデンス上限を 2.5 Hz に引き上げることで、歩幅ではなく歩数で
    ±1.8 m/s の横移動に届くかを試す。

    K1FlatGoalkeeperCfg との違い:

    * 位相周波数マッピング: 高速側を 2.0 → 2.5 Hz に引き上げ (1.0 m/s 以下 1.8 Hz は
      共通、1.8 m/s で 2.5 Hz、以降も同じ傾きで外挿)。obs (policy/critic) と
      feet_phase 報酬の 3 箇所を同時に上書きし、位相積分の整合を保つ。
    * y カリキュラム: ±1.5 → ±1.8 の 1 段拡張 (±1.5 習得済みポリシーからの
      warm-start 前提)。x は ±0.7 固定のまま。
    * extreme corner: ±1.5 版で頑健化済みの warm-start 前提で 3000 iter から導入。

    NOTE: デプロイ時は C++ 側 cmd_phase_freq() の高速側周波数も 2.5 Hz に
    合わせること (k1_constants_isaaclab.hpp)。
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 位相周波数マッピング: 高速側 2.5 Hz ---
        # 位相アキュムレータは obs/reward のどちらが先に呼ばれても同じ周波数で
        # 積分されるよう、params を持つ全 3 項を同じ値で上書きする。
        gk_phase_freq = {
            "low_speed": 1.0,
            "high_speed": 1.8,
            "low_freq": 1.8,
            "high_freq": 2.5,
        }
        self.observations.policy.gait_phase.params.update(gk_phase_freq)
        self.observations.critic.gait_phase.params.update(gk_phase_freq)
        self.rewards.feet_phase.params.update(gk_phase_freq)

        # --- y カリキュラム: ±1.5 開始 → ±1.8 ---
        self.curriculum.lin_vel_command.params["stages_x"] = [
            (-0.7, 0.7),
            (-0.7, 0.7),
        ]
        self.curriculum.lin_vel_command.params["stages_y"] = [
            (-1.5, 1.5),
            (-1.8, 1.8),
        ]
        # stage0 閾値は ±1.5 習得済み実測 (~0.43)。最終値はログ表示専用。
        self.curriculum.lin_vel_command.params["error_threshold"] = [0.43, 0.55]

        # --- extreme corner: warm-start 前提でさらに前倒し ---
        self.curriculum.extreme_commands.params["num_steps"] = 3000
        self.curriculum.extreme_commands.params["ramp_steps"] = 1500


@configclass
class K1FlatEnvLearnStandingCfg(K1FlatEnvCfg):
    """追加学習で立ち姿勢を覚えるための環境設定。これは予め普通のFlatで学習したポリシーに追加学習する用途"""
    def __post_init__(self):
        super().__post_init__()
        # Rewards
        self.commands.base_velocity.resampling_time_range = (1.0, 5.0)  # コマンドのリサンプリング時間の範囲を変更
        self.commands.base_velocity.rel_standing_envs = 0.3

@configclass
class K1FlatImproveSteadynessCfg(K1FlatEnvCfg):
    """学習済のポリシーに対して安定化のための追加学習を行う際の環境設定"""
    def __post_init__(self):
        super().__post_init__()
        # Rewards
        self.commands.base_velocity.resampling_time_range = (1.0, 4.0)  # コマンドのリサンプリング時間の範囲を変更
        self.rewards.ang_vel_xy_l2.weight = -0.30 * 1.7
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.action_rate_l2.weight = -0.6 * 1.3
        self.rewards.dof_acc_l2.weight = -1.2e-6
        self.rewards.dof_torques_l2.weight = -1.0e-5

@configclass
class K1FlatImproveAngTrackingCfg(K1FlatEnvCfg):
    """学習済ポリシーに対して角速度(yaw)追従を強化するための追加学習用環境設定。

    背景: lin_vel 高速域の追従はカリキュラム+coarse項の追加で改善した一方、
    その過程で track_ang_vel_z_exp の重みが 3.0→2.0 に下げられ lin 偏重になり、
    結果として yaw 追従精度が低下した。本設定は lin の高速追従を維持しつつ
    ang の追従を取り戻すよう、報酬バランスを yaw 側に振り直して再学習する。

    使い方: 既存 Flat ポリシーの checkpoint から --resume で追加学習する。
        ./train_ang_tracking.sh --resume --load_run <既存run名>
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 角速度追従を強化 ---
        # 鋭い項 (std=0.25) の重みを 2.0 → 4.0 に引き上げ、yaw 追従を最優先にする。
        self.rewards.track_ang_vel_z_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        # lin 側の coarse 項と同じ思想で ang にも広い std の項を追加する。
        # 鋭い項 (std=0.25) は誤差 ~0.4 rad/s で exp(-err²/std²) が飽和し勾配が消えるため、
        # 旋回コマンドが大きく追従誤差が大きい領域で「もっと回せ」の信号が死ぬ。
        # std を広げた同形の報酬を小重みで加算し、高誤差域でも勾配を残す。
        self.rewards.track_ang_vel_z_coarse = RewTerm(
            func=mdp.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # --- lin の高速追従を「忘れさせない」ためカリキュラムを凍結 ---
        # checkpoint には curriculum の進捗が保存されないため、resume すると
        # lin_vel_command カリキュラムが stage0 (±0.6) から再進行してしまい、
        # せっかく獲得した高速追従を一時的に練習しなくなる。yaw 追従の再学習に
        # 集中するため、lin の段階的拡張は止めて最終ステージ相当の広い範囲で固定する。
        self.curriculum.lin_vel_command = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.8, 1.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.9, 0.9)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 多様な yaw コマンドに頻繁に晒すためリサンプリング間隔を短めに固定する。
        self.commands.base_velocity.resampling_time_range = (1.0, 5.0)


@configclass
class K1FlatImprovePostureCfg(K1FlatEnvCfg):
    """学習済ポリシーに対して上体の傾き・上下動を抑える仕上げ学習用の環境設定。

    2 万イテレーションのフルパイプライン学習済みモデルからの resume を前提とする。
    checkpoint にはカリキュラムの進捗が保存されないため、resume 時に 0 から
    再進行しないよう、各カリキュラムを「学習終了時点の状態」に固定する:

    * lin_vel_command: 段階拡張を止め、最終ステージ相当の範囲で固定
    * extreme_commands: 最初から発動 (num_steps=0, ramp_steps=1 で即 α=1 →
      extreme_prob=0.35 + resampling (0.8, 8.0))
    * push_robot: stage1 相当の値を直接設定

    使い方::

        ./train_posture.sh --resume --load_run <既存run名>
    """

    def __post_init__(self):
        super().__post_init__()

        # --- トルク系の抑制 ---
        self.rewards.dof_torques_l2.weight = -1.5e-5

        # --- 機械パワー (torque * joint_vel) の抑制 ---
        # dof_torques_l2 は姿勢を「支えているだけ」の静的な保持トルクも罰するため、
        # 傾き抑制 (flat_orientation_l2) と競合しやすい。パワーは実際に仕事をしている
        # 動き — 高速な振り回しや拮抗筋的な押し合い — だけを罰するので、姿勢保持を
        # 犠牲にせずエネルギー効率と滑らかさを促せる。
        # 絶対値和なので値のスケールは sum|τ·ω| [W] 相当 (数十〜数百 W)。重みは
        # dof_torques_l2 の寄与と同程度の桁になる初期値なので、reward logger で
        # 他項と桁を突き合わせて要チューニング (効かなければ増量、追従が落ちるなら減量)。
        # 重み探索 (2026-08-09, 300it プローブ × {-1e-5, -3e-5, -1e-4, -3e-4}):
        # -1e-4 で err_vel_xy 1.02→1.28 と追従悪化が進行、-3e-4 は歩行放棄で崩壊。
        # -3e-5 は追従 +0.04 (seed ノイズ帯上端)・姿勢維持でパワー約 3 割減となり、
        # 「追従を妨げない範囲の最大」として採用。-1e-5 は無害だが削減効果ほぼ無し。
        self.rewards.joint_power_l2 = RewTerm(
            func=joint_power_l2,
            weight=-3.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


        # --- 上体の傾き抑制 ---
        # m04 レシピ (2026-07-28〜29): フルパイプライン学習後の仕上げ resume で
        # -20 → -40 に増量すると傾き -34%、追従・位相・振動は維持。
        # NOTE: m01 (最初から -40 + extreme) はスクラッチ学習では lin 崩壊したが、
        # 本設定は「習得済みポリシーの仕上げ」なので条件が異なる。lin 追従
        # (error_vel_xy) が悪化し続ける場合は -30 程度へ緩めること。
        self.rewards.flat_orientation_l2.weight = -40.0
        self.rewards.ang_vel_xy_l2.weight = -0.85

        # --- feet flat  ---
        # なんかflatのペナルティがデカすぎるので減らす
        self.rewards.feet_parallel_to_ground.weight = 19.0

        # --- termination ---
        # 転倒も減らしたいので、死亡ペナルティを増やす
        self.rewards.termination_penalty.weight = -600.0

        # --- 上体の上下動抑制 ---
        # lin_vel_z_l2 (速度ペナルティ) は歩容の構造的下限で飽和し増量無効 (q系列)。
        # 高さ「位置」の偏差を直接罰して vaulting の振幅自体を縮める。
        # 重みは reward logger で他項 (ang_vel_xy_l2 ≈ 0.01〜0.1/step) と桁を
        # 合わせた初期値。効かなければ増やし、着地が硬くなるなら減らす。
        # self.rewards.base_height_track.weight = -30.0
        # ↑これは正直難しいのでおそらく無しの方が良い。

        # base_height_penalty (minimum_height, min_height=0.53, weight=-100) が
        # 目標高さ 0.53 と干渉する (目標付近で常に -100 が出る) ので閾値を下げる。
        # self.rewards.base_height_penalty.params["min_height"] = 0.48

        # --- カリキュラムを学習終了時点の状態に固定 ---
        # lin_vel_command: 最終ステージ相当で固定 (再進行による "忘れ" を防ぐ)
        self.curriculum.lin_vel_command = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.9, 0.9)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # extreme_commands: 最初から発動させる。num_steps=0 で 1 イテレーション目
        # から有効になり、ramp_steps=1 で即座に α=1 (extreme_prob=0.35 +
        # resampling_time_range (0.8, 8.0) 差し替え) に達する。
        self.curriculum.extreme_commands.params["num_steps"] = 0
        self.curriculum.extreme_commands.params["ramp_steps"] = 1
        # α=1 到達前の初回分も含め、リサンプリング間隔は最初から本番値にする
        self.commands.base_velocity.resampling_time_range = (0.8, 8.0)
        self.curriculum.command_resampling_time_range = None

        # push_robot: いきなり厳しい状態にする
        self.curriculum.push_robot_stage1 = None
        self.events.push_robot.interval_range_s = (0.5, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.7, 0.7),
            "y": (-0.7, 0.7),
            "roll": (-0.06, 0.06),
            "pitch": (-0.06, 0.06),
        }


@configclass
class K1FlatImprovePostureFastCfg(K1FlatImprovePostureCfg):
    """高速歩行版 (K1FlatFastCfg: x ±1.8 + 最大 2.5 Hz) ポリシー用の姿勢仕上げ設定 (2026-08-21)。

    K1FlatImprovePostureCfg は通常版 (x ±1.5 / 2.0 Hz) の学習終了時状態で
    カリキュラムを固定するため、高速版ポリシーに直接使うと位相周波数マッピングが
    学習条件と食い違い (2.0 vs 2.5 Hz)、コマンド範囲も ±1.5 に縮んでしまう。
    本設定は姿勢仕上げの報酬構成 (flat_ori -40 / joint_power -3e-5 等) を
    そのまま使い、位相と範囲だけ K1FlatFastCfg の学習終了時状態に合わせる。

    使い方::

        torchrun --standalone --nproc_per_node=2 train.py \\
            --task Isaac-Velocity-Flat-Posture-Fast --headless --distributed \\
            --num_envs 2048 --resume --load_run <高速版run名> --reset_noise_std 0.05
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 位相周波数マッピング: K1FlatFastCfg と同一 (高速側 2.5 Hz) ---
        fast_phase_freq = {
            "low_speed": 1.0,
            "high_speed": 1.8,
            "low_freq": 1.8,
            "high_freq": 2.5,
        }
        self.observations.policy.gait_phase.params.update(fast_phase_freq)
        self.observations.critic.gait_phase.params.update(fast_phase_freq)
        self.rewards.feet_phase.params.update(fast_phase_freq)

        # --- コマンド範囲: 高速版の学習終了時状態 (x ±1.8) で固定 ---
        self.commands.base_velocity.ranges.lin_vel_x = (-1.8, 1.8)

        # --- joint_power は ±1.8 向けに半減 (2026-08-22) ---
        # パワーは速度の 2 乗で増えるため、±1.5 で調整した -3e-5 は ±1.8 では実効的に
        # 約 2 倍効き、err_vel_xy 1.18→1.83 の漸進崩壊を起こした (300it プローブで
        # 主因を特定: flat_ori -40→-30 は効果なし、jp 半減で悪化がほぼ停止)。
        self.rewards.joint_power_l2.weight = -1.5e-5


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
