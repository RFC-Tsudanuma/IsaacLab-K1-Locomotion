# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ActuatorNetMLPCfg, DelayedPDActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .velocity_env_cfg import (
    JOINT_NAMES_K1,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    CurriculumCfg,
)

# K1専用のMDP関数 (位相報酬 + 位相観測)
# 注意: これらの関数が .mdp フォルダ内に存在することを確認してください
from .mdp import feet_phase, phase_obs
from .mdp.events import randomize_phase_freq_offset
from .mdp.rewards import (
    feet_close_penalty,
    feet_parallel_to_ground,
    minimum_height,
    zmp_support_center,
    action_smoothness_l2,
    action_rate_l2,
    compute_zmp_xy,
    feet_landing_impact,
    feet_landing_vel,
    feet_heel_strike,
    com_jerk_l2,
    both_feet_not_in_contact,
)
from .mdp.curriculums import (
    modify_command_resampling_time_range,
    lin_vel_command_curriculum,
    modify_push_robot,
)

##
# 基本設定
##
# 歩行周波数は速度コマンドのノルムに応じて線形遷移する (mdp.events.compute_cmd_phase_freq):
#   ||cmd_xy|| <= _PHASE_SPEED_LOW  → _PHASE_FREQ_LOW で固定
#   それ以上                        → _PHASE_SPEED_HIGH で _PHASE_FREQ_HIGH になる傾きで線形増加
# per-env の ±0.05 Hz ランダムオフセット (randomize_phase_freq_offset) が加算される。
_PHASE_FREQ_LOW: float = 1.5    # Hz (低速歩行の基本周波数)
_PHASE_FREQ_HIGH: float = 2.0   # Hz (_PHASE_SPEED_HIGH 時の周波数)
_PHASE_SPEED_LOW: float = 1.0   # m/s (この速度までは _PHASE_FREQ_LOW 固定)
_PHASE_SPEED_HIGH: float = 1.8  # m/s (この速度で _PHASE_FREQ_HIGH に到達)
_PHASE_FREQ_PARAMS: dict = {
    "low_speed": _PHASE_SPEED_LOW,
    "high_speed": _PHASE_SPEED_HIGH,
    "low_freq": _PHASE_FREQ_LOW,
    "high_freq": _PHASE_FREQ_HIGH,
}
_COMMAND_THRESHOLD: float = 0.05 # コマンド速度がこれ未満のときは停止とみなす
_STANCE_RATIO: float = 0.50 # 接地時間の割合
# 立ち止まり「かつ静止している」ときだけ action_smoothness / action_rate ペナルティを
# この倍率に増やし、recurrent ポリシーの停止時振動を抑える。push を受けて base が動いた
# 瞬間は倍率が 1.0 に戻るので push recovery は阻害しない (rewards._stand_still_boost 参照)。
_STAND_STILL_PENALTY_SCALE: float = 3.0
# True: LSTM/GRU (recurrent) 方策を使う前提。False: MLP 方策。
# recurrent では rsl_rl の mirror loss が構造的に動かないため、対称性は報酬
# (joint_mirror_symmetry) で担保する。MLP では mirror loss を使うのでこの報酬は不要。
# この1フラグで「env 側の joint_mirror_symmetry 報酬」と「runner 側の mirror loss」を
# 排他に切り替える (agents/rsl_rl_ppo_cfg.py も同じフラグを参照)。アーキを変えるときは
# ここを切り替えること。
_USE_RECURRENT_POLICY: bool = False

_K1_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../../assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf",
)

_LEG_NET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../../actuator_net_leg.pt",
)

##
# K1 robot asset configuration
##

actuatornet_leg = ActuatorNetMLPCfg(
    joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch"],
    effort_limit={".*_Hip_Pitch": 68.0, ".*_Hip_Roll": 76.0, ".*_Hip_Yaw": 38.3, ".*_Knee_Pitch": 112.0},
    velocity_limit={".*_Hip_Pitch": 14.66, ".*_Hip_Roll": 12.57, ".*_Hip_Yaw": 17.59, ".*_Knee_Pitch": 12.57},
    # ActuatorNetMLPCfg は DCMotorCfg を継承しており、ピーク(ストール)トルク saturation_effort が必須。
    # トルク-速度カーブの clipping にのみ使われるスカラ値 (グループ全関節に共通)。
    # 各関節の上限は effort_limit が別途キャップするため、ここでは最大の effort_limit (Knee=112) を
    # 上回る値にして低速域での不要なトルク制限を避ける。実機のモータ仕様が分かれば置き換えること。
    saturation_effort=120.0,
    stiffness={".*_Hip_.*": 140.0, ".*_Knee_Pitch": 140.0},
    damping={".*_Hip_.*": 3.5 , ".*_Knee_Pitch": 3.5},
    armature={".*_Hip_Pitch": 0.0478125,".*_Hip_Roll": 0.0339552 , ".*_Knee_Pitch": 0.095625, '.*_Hip_Yaw': 0.0282528},
    network_file=_LEG_NET_PATH,
    pos_scale=1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="vel_pos",
    input_idx=[2,1,0]
)

delayed_pd_leg = DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch"],
            effort_limit={".*_Hip_Pitch": 68.0, ".*_Hip_Roll": 76.0, ".*_Hip_Yaw": 38.3, ".*_Knee_Pitch": 112.0},
            velocity_limit={".*_Hip_Pitch": 14.66, ".*_Hip_Roll": 12.57, ".*_Hip_Yaw": 17.59, ".*_Knee_Pitch": 12.57},
            # stiffness={".*_Hip_Pitch": 30.20098947, ".*_Hip_Roll": 21.44796105, ".*_Hip_Yaw": 17.84601339, ".*_Knee_Pitch": 60.40197893},
            # damping={".*_Hip_Pitch": 90.6029684, ".*_Hip_Roll": 64.34388314, ".*_Hip_Yaw": 53.53804017, ".*_Knee_Pitch": 120.8039579},
            stiffness={".*_Hip_.*": 160.0, ".*_Knee_Pitch": 160.0},
            damping={".*_Hip_.*": 4.0 , ".*_Knee_Pitch": 4.0},
            armature={".*_Hip_Pitch": 0.0478125,".*_Hip_Roll": 0.0339552 , ".*_Knee_Pitch": 0.095625, '.*_Hip_Yaw': 0.0282528},
            min_delay=2,
            max_delay=7,
        )

K1_LOCOMOTION_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_K1_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_Hip_Pitch": -0.26,
            ".*_Hip_Roll": 0.0,
            ".*_Hip_Yaw": 0.0,
            ".*_Knee_Pitch": 0.52,
            ".*_Ankle_Pitch": -0.26,
            ".*_Ankle_Roll": 0.0,
            # "AAHead_yaw" : 0.0,
            # "Head_pitch" : 0.0,
            # "ALeft_Shoulder_Pitch": 0.0,
            # "ARight_Shoulder_Pitch": 0.0,
            # "Left_Shoulder_Roll": -0.7853981634 * 1.75,
            # "Left_Elbow_Pitch": 0.0,
            # "Left_Elbow_Yaw": 0.0,
            # "Right_Shoulder_Roll": 0.7853981634 * 1.75,
            # "Right_Elbow_Pitch": 0.0,
            # "Right_Elbow_Yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": delayed_pd_leg,
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit=38.3,
            velocity_limit=17.59,
            # stiffness=17.84601339,
            # damping=53.53804017,
            stiffness=50.0,
            damping=2.5,
            armature=0.0282528,
            min_delay=2,
            max_delay=7,
        ),
        # "arms": IdealPDActuatorCfg(
        #     joint_names_expr=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"],
        #     effort_limit=100.0,
        #     velocity_limit=50.0,
        #     stiffness=40.0,
        #     damping=10.0,
        # ),
    },
)

# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

##
# Observation Groups (Asymmetric)
##

@configclass
class K1PolicyCfg(ObsGroup):
    """Actor（ポリシー）用：実機で得られる情報のみ。線速度は含めない。"""
    # 線速度(base_lin_vel)は削除
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel,noise=Unoise(n_min=-1.5, n_max=1.5),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    actions = ObsTerm(func=mdp.last_action)

    # 整理した位相観測 (周波数はコマンド速度に応じて線形遷移)
    gait_phase = ObsTerm(func=phase_obs, params={"cmd_threshold": _COMMAND_THRESHOLD, **_PHASE_FREQ_PARAMS})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class K1CriticCfg(ObsGroup):
    """Critic（価値関数）用：特権情報（真の線速度など）を含める。"""
    # 基本情報はActorと同じ（ノイズなし）
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # Criticにはこれを入れる
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel,params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    actions = ObsTerm(func=mdp.last_action)
    gait_phase = ObsTerm(func=phase_obs, params={"cmd_threshold": _COMMAND_THRESHOLD, **_PHASE_FREQ_PARAMS})
    zmp_position = ObsTerm(func=compute_zmp_xy, params={"asset_cfg": SceneEntityCfg("robot")})

    def __post_init__(self):
        self.enable_corruption = False # Criticにノイズは不要
        self.concatenate_terms = True

@configclass
class K1ObservationsCfg(ObservationsCfg):
    policy: K1PolicyCfg = K1PolicyCfg()
    critic: K1CriticCfg = K1CriticCfg()

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

@configclass
class K1Rewards(RewardsCfg):
    """K1の報酬設定。位相ベースの歩行と各関節の制約を両立。"""

    # --- 基本報酬 ---
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )

    # --- 位相ベースの歩行報酬 (重要) ---
    # 空中時間報酬を0にし、位相報酬をメインにする
    feet_phase = RewTerm(
        func=feet_phase,
        weight=1.0, # 位相に合わせて足を動かすことへの報酬
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "command_name": "base_velocity",
            "stance_ratio": _STANCE_RATIO,
            "cmd_threshold": _COMMAND_THRESHOLD,
            **_PHASE_FREQ_PARAMS,
        },
    )

    # feet_height_bezier = RewTerm(
    #     func=feet_height_bezier,
    #     weight=1.5, # 足の高さが理想的なベジェ曲線に近い場合の報酬
    #     params={
    #         "swing_height": 0.12,
    #         "sigma": 0.005,
    #         "phase_freq": _PHASE_FREQ,
    #         "stance_ratio": _STANCE_RATIO,
    #     },
    # )


    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0, # 位相報酬を使う場合は通常0にするか微量にする
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # --- 制約・ペナルティ ---
    dof_pos_limits_ankle = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )
    # dof_pos_limits_arm = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_Pitch",".*_Shoulder_Roll",".*_Elbow_Pitch",".*_Elbow_Yaw"])},
    # )
    # dof_vel_limits = RewTerm(
    #     func=mdp.joint_vel_limits,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]),
    #         "soft_ratio": 0.95
    #     },
    # )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.10,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    # joint_deviation_arm = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_Pitch",".*_Shoulder_Roll",".*_Elbow_Pitch",".*_Elbow_Yaw"])},
    # )

    base_height_penalty = RewTerm(
        func=minimum_height,
        weight=-100.0,
        params={
            "min_height": 0.54,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": None,
        },
    )
    feet_close_penalty = RewTerm(
        func=feet_close_penalty,
        weight=-20.0,
        params={
            "feet_distance_threshold": 0.14,
        },
    )

    # weight は 30/45/60/90/120 の 5000iter 比較 (2026-07-17) で決定。追従調和平均は
    # 30:0.584 → 45:0.588 → 60:0.590 → 90:0.593 → 120:0.490 と 90 までは単調改善、
    # 120 で lin 追従が崩壊する。10000iter 継続でも 90 (0.616) > 60 (0.605) を確認し 90 を採用。
    feet_parallel_to_ground = RewTerm(
        func=feet_parallel_to_ground,
        weight=90.0,
        params={
            "sigma": 0.08
        },
    )

    action_smoothness_l2 = RewTerm(
        func=action_smoothness_l2,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": _COMMAND_THRESHOLD,
            "stand_still_scale": _STAND_STILL_PENALTY_SCALE,
        },
    )

    # dof_vel_l2 = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-5.0e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
    # )

    # 左右対称性を報酬で担保する (アーキ非依存)。recurrent (LSTM/GRU) 方策では rsl_rl の
    # mirror loss が構造的に動かない (rsl_rl_ppo_cfg.py のコメント参照) ため、その代替。
    # exp(-error/0.1) の [0,1] 報酬で、左右股関節・膝が鏡像対称なほど高い。weight は要調整。
    # MLP では mirror loss 側を使うので weight=0 にして二重がけを避ける (_USE_RECURRENT_POLICY)。
    # joint_mirror_symmetry = RewTerm(
    #     func=joint_mirror_symmetry,
    #     weight=0.5 if _USE_RECURRENT_POLICY else 0.0,
    # )

    zmp_stability = RewTerm(
        func=zmp_support_center,
        weight=0.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "force_threshold": 2.0,
            "ema_alpha": 0.6,
            "sigma": 0.05,
        },
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
class K1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: K1Rewards = K1Rewards()
    observations: K1ObservationsCfg = K1ObservationsCfg()
    curriculum: K1FlatCurriculumCfg = K1FlatCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # =====================================================================
        # 旧 K1RoughEnvCfg.__post_init__ から移植した設定。
        # (rough 環境は廃止し、flat でも有効だった設定のみをここに集約した。
        #  flat で直後に上書きされていた設定 —— lin_vel_z_l2=0.0, undesired_contacts,
        #  dof_acc_l2/dof_torques_l2 の重み・対象関節など —— は意味が無いので移植していない。)
        # =====================================================================
        # Scene
        self.scene.robot = K1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Randomization
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_com.params["asset_cfg"].body_names = ["Trunk"]

        # Rewards の微調整 (rough 由来で flat でも有効なもの)
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.action_rate_l2 = RewTerm(
            func=action_rate_l2,
            weight=-0.005,
            params={
                "command_name": "base_velocity",
                "cmd_threshold": _COMMAND_THRESHOLD,
                "stand_still_scale": _STAND_STILL_PENALTY_SCALE,
            },
        )
        # dof_acc_l2 の対象関節 (重みは後段の flat 設定で -1.0e-6 に上書きされる)
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.35}, # ペナルティより低くなったら終了
        )
        # =====================================================================
        # ここから下は flat 環境固有の設定
        # =====================================================================

        # 環境毎に歩行周波数オフセットを ±0.05 Hz の範囲でランダム化 (startup で1度だけ)。
        # 基本周波数はコマンド速度に応じて線形遷移し (_PHASE_FREQ_PARAMS 参照)、
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
        # ガニ股対策 (2026-07-13): 左右 Hip_Yaw が外向きに開いた歩容になったため、
        # Hip_Yaw の偏差を独立項に分離して強く罰する。joint_deviation_hip (Yaw+Roll 合算
        # weight=-0.10) は Roll のみに変更。旧挙動は
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

        # 速度コマンドは velocity_env_cfg の UniformVelocityCommandCfg (連続サンプリング) を
        # そのまま使う。以前は離散格子サンプリング (DiscreteVelocityCommandCfg) に差し替えて
        # いたが、速度追従が十分になったため連続版に戻した (2026-07-13)。
        # lin_vel_x / lin_vel_y の範囲は lin_vel_command カリキュラムが段階的に拡張する。

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
class K1FlatZeroGainJointCfg(K1FlatEnvCfg):
    """1 関節がフリー (脱力) になっても歩ける fault-tolerant 歩行を学習する環境。

    指定した 1 関節の P/D ゲインをエピソード開始時に 0 にし、その状態でも転倒せず
    ゆっくり歩けることを目標にする。素の Flat 環境からの差分は 3 点:

      1. 指定関節の P/D ゲインを reset 毎に 0 にするイベント (下記 zero_gain_joint)。
         0 ゲインだと DelayedPD アクチュエータの出力トルクが (位置偏差・速度偏差に
         よらず) 0 になるため、その関節は実質フリーになる。
         対象関節は Hydra のコマンドライン上書きで差し替える想定:
             env.events.zero_gain_joint.params.asset_cfg.joint_names=[Right_Knee_Pitch]
      2. コマンド速度カリキュラム (lin_vel_command) を撤去し、xy 速度コマンドを
         固定レンジ ±0.5 m/s の単一サンプリングにする。故障状態で高速追従を狙うと
         破綻しやすいため、まずは低速の安定歩行に絞る。
      3. 報酬を「速度追従 < ZMP 安定」に再設計する。関節が 1 つ失われると正確な速度
         追従は困難なので、追従より支持多角形内に ZMP を保つ (転ばない) ことを優先する。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # --- 1. 指定関節の P/D ゲインを 0 に設定するイベント ---
        # operation="abs" + 分布 (0.0, 0.0) で「加算/スケールではなく絶対値 0 を代入」する。
        # mode="reset" なので各エピソード開始時 (env リセット時) に発火する。
        # joint_names は Hydra で上書きする前提のデフォルト値 (単一関節)。
        self.events.zero_gain_joint = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["Right_Hip_Pitch"]),
                "stiffness_distribution_params": (0.0, 0.0),  # P ゲイン → 0
                "damping_distribution_params": (0.0, 0.0),    # D ゲイン → 0
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        # --- 2. コマンド速度カリキュラムを撤去し、固定レンジにする ---
        # lin_vel_command は追従誤差に応じて範囲を ±0.6→±1.8 と段階拡大するカリキュラム。
        # これを None にして段階拡大を止め、以下の固定レンジのみでサンプリングさせる。
        self.curriculum.lin_vel_command = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # xy 速度コマンドに集中させるため yaw コマンドは 0 に固定 (旋回させない)。
        # 旋回も学習させたい場合はこの範囲を (-1.0, 1.0) 等に変更する。
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # --- 3. 報酬の重みを「歩けるが ZMP 安定も重視」に設計 ---
        # track_lin / track_ang / zmp_stability はいずれも [0,1] スケールの exp 報酬なので、
        # weight がそのまま相対的な優先度になる。
        # 故障環境では高速歩行を考慮する必要がないため、広い std=0.5 の粗い追従項 (coarse) は
        # 削除する。コマンド範囲 ±0.5 は鋭い std=0.25 の項 (sharp) で十分カバーできるので、
        # coarse を消したぶん sharp の重みを少し引き上げる。
        # ZMP 安定の重み: 当初 5.0 にしたところ「その場で静止していれば ZMP が常に支持基準点に
        # 一致して報酬が最大化する」局所最適に落ち、立ち止まる歩容になった。5.0→1.0 に下げても
        # まだ立ち止まった。理由は「重み」ではなく「実際に得られる報酬」で見る必要があるため:
        # 1500iter 学習後の episode 報酬は zmp≈0.58 に対し track≈0.20 で、重みが同じ 1.0 でも
        # 静止が容易に稼げる ZMP のリターンが追従の約3倍あり、依然として静止が最適だった
        # (足上げ指標 feet_air_time もほぼ 0)。そこで achieved ベースで ZMP のリターンが追従を
        # 下回るよう、重みを 1.0→0.3 に下げる (静止時 ZMP リターン ≈ 0.58×0.3 ≈ 0.17 < 歩行時の
        # 追従リターン)。これで「歩いてコマンドに追従するほうが得」になり足上げを促す。
        # ZMP 項は残るので歩行中の支持基準点付近への ZMP 維持は引き続き弱く促される。
        self.rewards.track_lin_vel_xy_exp.weight = 1.0      # 旧 1.5 → FT では 0.8 から少し引き上げ (sharp, std=0.25)
        self.rewards.track_lin_vel_xy_coarse = None         # 粗い追従項 (std=0.5) を削除 (高速追従不要)
        self.rewards.track_ang_vel_z_exp.weight = 0.8       # 旧 4.2 (yaw=0 固定なので過度な旋回を軽く罰する程度)
        self.rewards.zmp_stability.weight = 0.05            # 5.0→1.0→0.3→0.05 (静止の魅力を更に下げる; 実験 §7)

        # 足上げ (ステッピング) を促す位相非依存の報酬を強める。
        # 本環境では位相報酬 feet_phase を削除している (上の "6." 参照) ため、足上げを駆動する
        # 報酬が feet_air_time だけになる。Flat の既定重み 0.2 では弱すぎて「足を上げずに滑って
        # コマンドに追従する」解に収束し、feet_air_time の achieved 報酬が ~0.0001 (足が閾値以上
        # 滞空していない) に留まった。そこで weight を 0.2→1.5 (追従項と同程度) に引き上げ、足を
        # 閾値 (0.4s) 以上滞空させる=足を上げて歩くことを主要な報酬源にする。feet_air_time は
        # 位相非依存なので、固定リズムを強制せずに (故障脚をかばう非対称歩容も許容しつつ) 足上げを促す。
        self.rewards.feet_air_time.weight = 1.5             # 0.2→1.5 (feet_phase 削除の代替の足上げ駆動)

        # --- 4. ジャンプ (両足が同時に地面から離れる滞空) を罰する ---
        # 故障脚をかばって「片脚ホッピング」歩容に陥るのを防ぐ。both_feet_not_in_contact は
        # 両足非接地のとき -1 を返すので、正の weight を掛けると滞空ステップへのペナルティになる。
        # 通常の二足歩行 (常にどちらかが接地) では 0 なので、正常歩容は罰しない。
        self.rewards.no_jump = RewTerm(
            func=both_feet_not_in_contact,
            weight=2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")},
        )

        # --- 5. 故障環境では外乱 push を小さくし、増強カリキュラムも止める ---
        # 1 関節が失われた状態で大きな push を受けると容易に転倒して学習が進まないため、
        # push の速度振幅を ±0.5 → ±0.2 m/s に縮小する。さらに push を強めるカリキュラム
        # (push_robot_stage1: 6000step で間隔短縮 + roll/pitch 付与) も無効化し、外乱を
        # 小さいまま固定する。
        self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
        self.curriculum.push_robot_stage1 = None

        # --- 6. 位相: 観測は削除するが、歩容駆動として feet_phase 報酬は低重みで残す ---
        # 当初は位相を完全に撤去した (観測・報酬・オフセットイベントすべて None) が、その結果
        # 足上げを駆動する報酬が feet_air_time (弱い) だけになり、足を上げずに滑って追従する解に
        # 収束して歩かなかった (feet_air_time を 1.5 まで上げても足上げの実測 air-time は変化せず)。
        # このコードベースでステッピングを駆動できる報酬は位相ベース (feet_phase / foot_clearance) のみ
        # なので、feet_phase 報酬を「低重み」で復活させ歩行リズムを促す。
        #   - gait_phase 観測 (policy / critic) は削除したまま → 観測 45 次元を維持 (方策は「時計」を
        #     直接与えられない)。feet_phase は報酬側で内部位相タイマーから計算するので観測不要。
        #   - feet_phase は「報酬」であって「拘束」ではないため、健全脚はリズムで踏み、ゲイン 0 の
        #     故障脚はリズムを崩しても構わない (その脚の位相報酬が下がるだけ)。→ 故障許容と両立。
        #   - 位相周波数の per-env ランダムオフセット (randomize_phase_freq) は無効のまま。
        #     compute_cmd_phase_freq / get_phase_freq はオフセット未設定でも基本周波数で動作する。
        # 重み: 素の Flat では 1.0 で主駆動だったが、ここでは追従・ZMP・feet_air_time と併用する
        # 「ソフトな歩行リズム事前分布」として 0.5 に抑える。
        self.observations.policy.gait_phase = None
        self.observations.critic.gait_phase = None
        self.rewards.feet_phase.weight = 0.5
        self.events.randomize_phase_freq = None

        # --- 7. 「歩き出し」を妨げる制約を一括で緩める実験 (2026-07) ---
        # 背景: これまで report 上の feet_phase / feet_air_time / track_lin 報酬は上がっていたが、
        # 実際のビデオでは全関節が歩行しておらず (Hip_Pitch でも片脚を上げる程度)、
        # 速度追従誤差 Metrics/base_velocity/error_vel_xy ≈ 1.0 (コマンド ±0.5 に対し過大) で
        # 「立ち止まり/その場足踏み」だった。track_lin ≈ 0.2 は歩行ではなく静止とみなすべき。
        # 原因は、動歩行に伴う動き (重心の上下動・体幹の傾き・鉛直速度・素早い action 変化・
        # 関節加速/トルク) をことごとく罰するペナルティ群が「立ったまま動かない」を最安全解に
        # していたため。歩行の可否は今後 error_vel_xy で判定する (小さいほど追従=歩行)。
        # そこで ZMP (§3 で 0.05 に低減) に加え、歩き出しを縛っていると思われる重みを全て大幅に
        # 下げ、動いてコマンドに追従するほうが得になるようにする。追従目的 (track_lin/ang)、
        # 歩容駆動 (feet_phase/feet_air_time)、転倒防止 (termination_penalty) は維持する。
        # ※これは制約緩和の実験。緩めすぎで暴れる/転ぶ場合は個別に戻して締め直す。
        # 重心の上下動を強く罰していた最大の抑制項。閾値 0.54 は初期高 0.6 に近く、僅かな沈み込みで
        # -100 が入り「背を伸ばして立つ」が最適になっていた。重み大幅減 + 閾値も歩行の沈み込みを
        # 許すよう下げる (転倒防止は base_height 終了条件 0.35 と termination_penalty が担う)。
        self.rewards.base_height_penalty.weight = -5.0
        self.rewards.base_height_penalty.params["min_height"] = 0.45
        self.rewards.feet_close_penalty.weight = -2.0        # 旧 -20 (足を寄せる歩容を過度に制限)
        self.rewards.flat_orientation_l2.weight = -2.0       # 旧 -20 (体幹の傾きを罰し動歩行を抑制)
        self.rewards.feet_parallel_to_ground.weight = 20.0   # 旧 90 (遊脚中の足の傾きを縛る)
        # 跳ね対策で -0.4 まで上げたが、歩行を保てる範囲では鉛直速度² (vz²≈0.017) が変わらず
        # (ペナルティは払うが方策が跳ねを減らさない)、error_vel_xy は 0.45→0.52 に悪化しただけだった。
        # no_jump と同じく、この環境の跳ねはペナルティ強化では歩行を崩さずに除去できないと判明。
        # 歩行最優先のため緩和水準 (-0.1) に戻す。
        self.rewards.lin_vel_z_l2.weight = -0.1              # 緩和水準 (跳ね対策の -0.4 は無効だったので戻す)
        self.rewards.ang_vel_xy_l2.weight = -0.05            # 旧 -0.25 (roll/pitch 角速度を罰する)
        # 「跳ねる」歩容の抑制 (2026-07): 制約緩和で歩けるようになったが動きが hopping 気味に
        # なったため、跳ねを罰する 3 項を引き上げる。ただし最優先は「速度を出す=歩く」ことなので、
        # 緩和前の元値までは戻さず、歩行 (error_vel_xy が walking band ~0.4-0.5 に留まる) を保てる
        # 範囲で可能な限り上げる。no_jump は両足滞空 (=跳ねの署名) のみを罰し、正常な片足支持歩行
        # では 0 なので、歩行を保ったまま跳ねだけを狙い撃ちできる → 3 項の中で最も強めに戻す。
        # 跳ね抑制のチューニング結果 (2026-07): action_rate/smoothness/no_jump を 0.5〜2.0倍で振ったが、
        # 歩行を保てる範囲 (no_jump ≤1.5) では両足滞空(跳ね)がほぼ変わらず、no_jump=2.0 で初めて
        # 跳ねが減るが error_vel_xy が 0.45→0.72 に悪化し歩行が崩れた。歩行最優先の方針から、跳ねが
        # わずかに減り歩行を保てる exp8b 水準 (下記) を採用。跳ねの本質的抑制には別レバー(lin_vel_z 等)が必要。
        self.rewards.action_rate_l2.weight = -0.1            # 緩和 -0.05 → -0.1 (歩行維持できる上限, error~0.45)
        self.rewards.action_smoothness_l2.weight = -0.05     # 緩和 -0.02 → -0.05 (同上)
        self.rewards.feet_slide.weight = -0.1                # 旧 -0.5 (足の踏み替えを過度に制限)
        self.rewards.joint_deviation_hip_yaw.weight = -0.1   # 旧 -1.0 (股関節 yaw の可動を制限)
        self.rewards.joint_deviation_hip.weight = -0.02      # 旧 -0.10 (股関節 roll の可動を制限)
        self.rewards.dof_pos_limits_ankle.weight = -0.2      # 旧 -1.0 (足首可動域; 安全のため一部残す)
        self.rewards.dof_acc_l2.weight = -1.0e-7             # 旧 -1.0e-6 (関節加速を罰し動きを抑制)
        self.rewards.dof_torques_l2.weight = -5.0e-6         # 旧 -5.0e-5 (トルクを罰し動きを抑制)
        self.rewards.com_jerk_l2.weight = -1.0e-7            # 旧 -1.0e-6 (重心躍度を罰する)
        self.rewards.no_jump.weight = 1.0                    # 緩和 0.5 → 1.0 (歩行維持できる上限; 2.0で跳ね減も歩行崩壊)


@configclass
class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 0.1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
