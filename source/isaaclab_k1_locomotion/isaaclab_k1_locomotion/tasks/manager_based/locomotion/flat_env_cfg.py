# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .rough_env_cfg import K1RoughEnvCfg, _COMMAND_THRESHOLD
from .velocity_env_cfg import CurriculumCfg
import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.commands import DiscreteVelocityCommandCfg
from .mdp.events import randomize_phase_freq_offset
from .mdp.rewards import feet_landing_impact, feet_landing_vel, feet_heel_strike, com_jerk_l2
from .mdp.curriculums import (
    modify_command_resampling_time_range,
    lin_vel_command_curriculum,
    modify_push_robot,
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


@configclass
class K1FlatCurriculumCfg(CurriculumCfg):
    """K1 Flat 環境用のカリキュラム設定。"""

    # ステップ数が5000を超えたら、コマンドのリサンプリング時間分布の範囲を (1.0, 5.0) に変更
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (1.0, 7.0),
            "num_steps": 8000,
        },
    )

    # より細かいコマンド変動に対応
    command_resampling_time_range = CurrTerm(
        func=modify_command_resampling_time_range,
        params={
            "command_name": "base_velocity",
            "resampling_time_range": (0.5, 7.0),
            "num_steps": 14000,
        },
    )

    # 線速度コマンド範囲を段階的に拡げるカリキュラム
    # 追従誤差(EMA)が threshold を下回るとステージが進む: ±0.3 → ±0.6 → ±1.0
    '''
    lin_vel_command = CurrTerm(
        func=lin_vel_command_curriculum,
        params={
            "command_name": "base_velocity",
            "stages_x": [(-0.6, 0.6), (-1.2, 1.2), (-1.5, 1.5), (-1.8, 1.8)],
            "stages_y": [(-0.5, 0.5), (-0.7, 0.7), (-0.8, 0.8), (-0.9, 0.9)],
            # 各ステージを「本物の関門」にするための閾値。広い範囲ほど絶対誤差は出やすいので
            # わずかに緩めるが、緩めすぎると「狭い範囲を習得した時点で広い範囲のゆるい閾値も
            # 満たしてしまい」0→1→2 と一気に遷移する。実測では stage0(±0.6)の到達誤差が ~0.30、
            # その直後の ±1.2 での誤差が ~0.43、±1.8 で ~0.75。旧設定 [0.30, 0.60, 0.55] は
            # stage1/2 の閾値が「到達済みの誤差」より緩く、ゲートとして機能していなかった。
            # そこで stage1 は ±1.2 でまだ達成していない 0.34 まで締めて再学習を要求する
            # (stage0=0.30 は約500iter かけて到達する適切なゲートなので維持。
            #  最終 stage2 の値は遷移判定に使われずログ表示専用)。
            "error_threshold": [0.30, 0.39, 0.45, 0.43],
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
    '''

    # push_robot を段階的に強くするカリキュラム
    # 初期値 (EventCfg): interval 7-10s, vel ±0.5 → ±0.5
    push_robot_stage1 = CurrTerm(
        func=modify_push_robot,
        params={
            "term_name": "push_robot",
            "num_steps": 6000,
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

    def __post_init__(self):
        super().__post_init__()

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
        self.rewards.track_ang_vel_z_exp.weight = 4.2
        self.rewards.ang_vel_xy_l2.weight = -0.25
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
        # 速度コマンドを離散格子からサンプリングする版に差し替える
        # lin_vel_x / lin_vel_y は lin_vel_command カリキュラムが段階的に拡張する
        prev = self.commands.base_velocity
        self.commands.base_velocity = DiscreteVelocityCommandCfg(
            asset_name=prev.asset_name,
            resampling_time_range=prev.resampling_time_range,
            rel_standing_envs=prev.rel_standing_envs,
            rel_heading_envs=prev.rel_heading_envs,
            heading_command=prev.heading_command,
            heading_control_stiffness=prev.heading_control_stiffness,
            debug_vis=prev.debug_vis,
            ranges=DiscreteVelocityCommandCfg.Ranges(
                lin_vel_x=prev.ranges.lin_vel_x,
                lin_vel_y=prev.ranges.lin_vel_y,
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
            lin_vel_x_resolution=0.05,
            lin_vel_y_resolution=0.05,
            ang_vel_z_resolution=0.2,
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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.9, 0.9)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 多様な yaw コマンドに頻繁に晒すためリサンプリング間隔を短めに固定する。
        self.commands.base_velocity.resampling_time_range = (1.0, 5.0)


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None