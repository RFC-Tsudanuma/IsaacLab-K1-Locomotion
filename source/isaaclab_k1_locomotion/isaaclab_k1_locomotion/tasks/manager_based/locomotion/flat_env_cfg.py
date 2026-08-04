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
from .mdp.commands import ExtremeVelocityCommandCfg
from .mdp.rewards import feet_landing_impact, feet_landing_vel, feet_heel_strike, com_jerk_l2, base_ang_acc_l2
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
            proportion=0.9,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            border_width=0.25,
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
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
# 観測を「最新コマンド」と「コマンドを除いた観測の履歴」に分離する。
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
    """Actor 用: コマンドを除いた観測 (ノイズあり) を HISTORY_LENGTH ステップ分バッファする。

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
        self.velocity_commands = None
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
    """Critic 用: コマンドを除いた観測 (ノイズなし・特権情報込み) の履歴。"""

    def __post_init__(self):
        super().__post_init__()
        self.velocity_commands = None
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
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.5)
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
                "mass_distribution_params": (0.5, 1.5),
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
                "inertia_distribution_params": (0.7, 1.3),
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
        # self.scene.terrain.terrain_type = "generator"
        # self.scene.terrain.terrain_generator = NOISY_FLAT_TERRAIN_CFG
        # self.scene.terrain.max_init_terrain_level = None
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

        # --- 上体の傾き抑制 ---
        # m04 レシピ (2026-07-28〜29): フルパイプライン学習後の仕上げ resume で
        # -20 → -40 に増量すると傾き -34%、追従・位相・振動は維持。
        # NOTE: m01 (最初から -40 + extreme) はスクラッチ学習では lin 崩壊したが、
        # 本設定は「習得済みポリシーの仕上げ」なので条件が異なる。lin 追従
        # (error_vel_xy) が悪化し続ける場合は -30 程度へ緩めること。
        self.rewards.flat_orientation_l2.weight = -40.0

        # --- 上体の上下動抑制 ---
        # lin_vel_z_l2 (速度ペナルティ) は歩容の構造的下限で飽和し増量無効 (q系列)。
        # 高さ「位置」の偏差を直接罰して vaulting の振幅自体を縮める。
        # 重みは reward logger で他項 (ang_vel_xy_l2 ≈ 0.01〜0.1/step) と桁を
        # 合わせた初期値。効かなければ増やし、着地が硬くなるなら減らす。
        self.rewards.base_height_track.weight = -30.0
        # base_height_penalty (minimum_height, min_height=0.53, weight=-100) が
        # 目標高さ 0.53 と干渉する (目標付近で常に -100 が出る) ので閾値を下げる。
        self.rewards.base_height_penalty.params["min_height"] = 0.48

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

        # push_robot: stage1 (6000 iter 到達時) 相当を直接設定
        self.curriculum.push_robot_stage1 = None
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
        }


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
