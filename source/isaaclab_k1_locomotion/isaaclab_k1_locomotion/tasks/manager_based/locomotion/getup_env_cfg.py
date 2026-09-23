# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 の「起き上がり (get-up)」ポリシー学習用の環境設定。

設計方針 (重要):
  この環境は歩行系 (velocity_env_cfg / rough_env_cfg) を **import して継承しない**。
  歩行側の設定変更が起き上がりに波及するのを避けるため、必要な設定は
  「コピーして持ち込む」方針で self-contained にしている。
    - ロボット設定 (モーター等): rough_env_cfg.py の K1_LOCOMOTION_CFG を複製
      (脚 delayed_pd / 足 actuator はそのまま、全身運動なので腕・頭の actuator を追加)。
    - events 設定: velocity_env_cfg.py の EventCfg を複製。
    - その他の scaffolding (scene / actions / observations / rewards / terminations /
      commands / curriculum): velocity_env_cfg.py を複製し K1 用に body 名を調整。

  歩行と異なり全身運動になるため、URDF は 22 自由度モデル
  (assets_soccer/booster_robotics_robots/K1/K1_22dof.urdf) を使う。
"""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# NOTE: upstream IsaacLab の共通 mdp 関数ライブラリ。歩行側の *_env_cfg とは無関係なので
#       import してよい (歩行側の設定変更の影響を受けない)。
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# 起き上がり専用の MDP 関数 (ローカル定義)。上記 `mdp` は upstream ライブラリなので、
# ローカル関数は必ず `.mdp.*` から明示 import する (歩行側 *_env_cfg には依存しない)。
from .mdp.getup_rewards import (
    base_height_increase,
    base_height,
    head_height,
    head_height_exp_reward,
    low_head_height_penalty,
    supported_head_height_exp,
    center_of_mass_height_reward,
    lowest_hip_height_reward,
    timeout_below_head_height,
    feet_ground_contact,
    one_foot_airborne_above_head_height,
    airborne_feet_count,
    feet_above_head_penalty,
    supine_bridge_progress,
    hip_clearance_with_foot_support_reward,
    hip_below_feet_exponential_penalty,
    feet_vertical_force,
    knee_above_trunk_and_hips_penalty,
    trunk_contact_under_foot_load_penalty,
    balanced_foot_load_progress,
    feet_flat_penalty,
    joint_torque_over_limit,
    joint_power_l1,
    action_smoothness_l2,
    upright_posture,
    stand_still_when_up,
    body_symmetry,
    body_symmetry_l1,
    elbow_deviation_exponential_penalty,
    hip_yaw_deviation_exponential_penalty,
    joint_deviation_l1_above_head_height,
    joint_deviation_l1_when_upright,
    staged_ground_contact_penalty,
    proximal_arm_contact_penalty,
    hand_support_by_head_and_hip_height,
    staged_knee_flexion_extension_reward,
    staged_hip_pitch_squat_to_stand_reward,
    backward_hip_pitch_fold_penalty,
    body_height_order,
    inverted_posture_penalty,
    lateral_roll_penalty,
    bilateral_limb_height_asymmetry_penalty,
    hip_roll_excess_penalty,
    vertical_alignment_to_support_reward,
    standing_support_geometry,
    hands_free_when_upright,
    standing_success_reward,
)
from .mdp.getup_terminations import (
    stable_standing,
    fall_above_head_height,
    foot_height_above_limit,
    head_high_while_hip_low,
)
from .mdp.events import reset_root_state_prone_supine
from .mdp.rewards import base_lin_vel_xy_l2
from .mdp.curriculums import (
    log_com_height,
    log_mean_body_z,
    log_min_body_z,
    log_max_body_z,
    log_foot_minus_head_height,
    log_hip_minus_foot_height,
    log_head_minus_hip_height,
    log_mean_joint_abs_deviation,
    log_mean_joint_position,
    log_trunk_tilt_deg,
    log_joint_speed_sq,
)

# ---------------------------------------------------------------------------
# 関節名リスト
# ---------------------------------------------------------------------------
# 全身運動なので 22 自由度すべてを制御対象にする。URDF (K1_22dof.urdf) の関節順に合わせる。
# obs の joint_pos/vel や action はこの順で指定する必要がある。
JOINT_NAMES_K1_22DOF = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

HIP_BODY_NAMES = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
]
HIP_GROUND_SENSOR_NAMES = [f"hip_ground_contact_{index}" for index in range(len(HIP_BODY_NAMES))]
FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]
FOOT_GROUND_SENSOR_NAMES = [f"foot_ground_contact_{index}" for index in range(len(FOOT_BODY_NAMES))]
HAND_BODY_NAMES = ["left_hand_link", "right_hand_link"]
HAND_GROUND_SENSOR_NAMES = [f"hand_ground_contact_{index}" for index in range(len(HAND_BODY_NAMES))]
PROXIMAL_ARM_BODY_NAMES = ["Left_Arm_3", "Right_Arm_3"]
PROXIMAL_ARM_GROUND_SENSOR_NAMES = [
    f"proximal_arm_ground_contact_{index}" for index in range(len(PROXIMAL_ARM_BODY_NAMES))
]
SHANK_BODY_NAMES = ["Left_Shank", "Right_Shank"]
SHANK_GROUND_SENSOR_NAMES = [f"shank_ground_contact_{index}" for index in range(len(SHANK_BODY_NAMES))]
TRUNK_GROUND_SENSOR_NAMES = ["trunk_ground_contact"]
HEAD_BODY_NAMES = ["Head_1", "Head_2"]
HEAD_GROUND_SENSOR_NAMES = [f"head_ground_contact_{index}" for index in range(len(HEAD_BODY_NAMES))]

HEAD_PHASE_HEIGHT = 0.35
HAND_RELEASE_HEAD_HEIGHT = 0.35
HAND_RELEASE_HIP_HEIGHT = 0.3
BRIDGE_HIP_HEIGHT = 0.45
CONTACT_THRESHOLD = 0.1
HEAD_CONTACT_THRESHOLD = 0.01


def _hip_ground_contact_sensor(body_name: str) -> ContactSensorCfg:
    """Create a single-body sensor filtered to contacts with the ground plane."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        history_length=3,
    )


def _foot_ground_contact_sensor(body_name: str) -> ContactSensorCfg:
    """Create a single-foot sensor filtered to contacts with the ground plane."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        history_length=3,
    )

# ---------------------------------------------------------------------------
# K1 robot asset configuration
# ---------------------------------------------------------------------------
# rough_env_cfg.py の K1_LOCOMOTION_CFG を複製 (import 継承しない)。
# 相違点:
#   - URDF を全身の 22 自由度モデルに変更 (K1_22dof.urdf)。
#   - 全身運動なので腕 (arms) と頭 (head) の actuator を追加。
#     (歩行側は URDF が脚のみで腕・頭 actuator はコメントアウトされていた)
_K1_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../../assets_soccer/booster_robotics_robots/K1/K1_22dof.urdf",
)

# 上半身 PD ゲインのスケール。rough に定義がある下半身には適用せず、設定を完全に一致させる。
# 環境変数 GETUP_PD_SCALE で腕・頭のゲインだけを実行時に上書きできる。
_PD_SCALE = float(os.environ.get("GETUP_PD_SCALE", "0.6"))

# 脚 actuator は rough_env_cfg の delayed_pd_leg と同一設定にする。
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

K1_GETUP_CFG = ArticulationCfg(
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
            # self collision を有効化 (2026-07-27)。今の起き上がりは自己衝突未考慮で腕/脚が
            # 体を貫通しうるため、実機で不可能な動きを排除する目的で ON にする。
            enabled_self_collisions=True,
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
            # 全身モデルの上半身の初期姿勢 (腕を体側に下ろした自然姿勢)。
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            "ALeft_Shoulder_Pitch": 0.0,
            "ARight_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": -0.7853981634 * 1.75,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": 0.0,
            "Right_Shoulder_Roll": 0.7853981634 * 1.75,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.0,
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
            stiffness=50.0,
            damping=2.5,
            armature=0.0282528,
            min_delay=2,
            max_delay=7,
        ),
        # 全身運動 (起き上がり) 用に腕・頭の actuator を追加する。
        # 歩行側では URDF が脚のみで、これらは未定義 (コメントアウト) だった。
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"],
            effort_limit=14.0,
            velocity_limit=33.51,
            armature=0.001,
            stiffness=40.0 * _PD_SCALE,
            damping=10.0 * _PD_SCALE,
            min_delay=2,
            max_delay=8,
        ),
        # 首 (AAHead_yaw) ・頭 (Head_pitch) は遅延 2~8 の DelayedPD にする。
        "head": DelayedPDActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit=6.0,
            velocity_limit=7.85,
            armature=0.001,
            stiffness=20.0 * _PD_SCALE,
            damping=5.0 * _PD_SCALE,
            min_delay=2,
            max_delay=8,
        ),
    },
)


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 足の報酬と、手・膝 (Shank) の段階的接地評価に必要な body だけを対象にする。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(.*_foot_link|.*_hand_link|.*_Shank)",
        history_length=3,
        track_air_time=True,
    )
    # filter付きContactSensorは単一bodyでのみ正しく働くため、Hipごとに地面接触を収集する。
    hip_ground_contact_0 = _hip_ground_contact_sensor(HIP_BODY_NAMES[0])
    hip_ground_contact_1 = _hip_ground_contact_sensor(HIP_BODY_NAMES[1])
    hip_ground_contact_2 = _hip_ground_contact_sensor(HIP_BODY_NAMES[2])
    hip_ground_contact_3 = _hip_ground_contact_sensor(HIP_BODY_NAMES[3])
    hip_ground_contact_4 = _hip_ground_contact_sensor(HIP_BODY_NAMES[4])
    hip_ground_contact_5 = _hip_ground_contact_sensor(HIP_BODY_NAMES[5])
    # feet_ground_contact は自己衝突を除き、左右足と地面の接触だけを現在フレームで判定する。
    foot_ground_contact_0 = _foot_ground_contact_sensor(FOOT_BODY_NAMES[0])
    foot_ground_contact_1 = _foot_ground_contact_sensor(FOOT_BODY_NAMES[1])
    # 手と頭などの自己衝突を除き、手支持報酬には手と地面の接触だけを使う。
    hand_ground_contact_0 = _foot_ground_contact_sensor(HAND_BODY_NAMES[0])
    hand_ground_contact_1 = _foot_ground_contact_sensor(HAND_BODY_NAMES[1])
    # Arm_3は肘側の前腕。ここでの支持をhand_link先端支持と区別する。
    proximal_arm_ground_contact_0 = _foot_ground_contact_sensor(PROXIMAL_ARM_BODY_NAMES[0])
    proximal_arm_ground_contact_1 = _foot_ground_contact_sensor(PROXIMAL_ARM_BODY_NAMES[1])
    # 膝と自己衝突を区別し、Shankと地面の接触だけを膝立ちペナルティへ渡す。
    shank_ground_contact_0 = _foot_ground_contact_sensor(SHANK_BODY_NAMES[0])
    shank_ground_contact_1 = _foot_ground_contact_sensor(SHANK_BODY_NAMES[1])
    # 初期仰向けを許容しつつ、上体を起こした後の背中・胴体支持を検出する。
    trunk_ground_contact = _hip_ground_contact_sensor("Trunk")
    # 初期仰向けでは許容し、Hip を持ち上げた後の頭支持を転倒判定で検出する。
    head_ground_contact_0 = _hip_ground_contact_sensor(HEAD_BODY_NAMES[0])
    head_ground_contact_1 = _hip_ground_contact_sensor(HEAD_BODY_NAMES[1])
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Zero velocity command kept for the existing observation layout."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-3.14, 3.14)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 全身運動なので 22 自由度すべてを行動対象にする。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=JOINT_NAMES_K1_22DOF, preserve_order=True, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 用: 実機で得られる情報のみ。base_lin_vel (真の線速度) は実機で取得困難なので
        actor には入れず、critic 側の特権情報にする (歩行 K1PolicyCfg と同じ非対称設計)。"""

        # observation terms (order preserved) — base_lin_vel を除いた 75 次元
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1_22DOF, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1_22DOF, preserve_order=True)},
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic 用: 特権情報 (真の base_lin_vel) を含む。順序は旧 policy obs と同じ
        (base_lin_vel 先頭) にして、旧 78 次元チェックポイントの critic をそのまま流用可能にする。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1_22DOF, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1_22DOF, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # critic にノイズは不要
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events.

    velocity_env_cfg.py の EventCfg を複製 (import 継承しない)。
    body 名 "base" は K1 には存在しないため、K1 の "Trunk" に __post_init__ 側で差し替える。
    """

    # startup
    # 地面 (足〜地面接触) の摩擦 DR。地面 (plane) の摩擦は固定 1.0 で combine_mode="multiply"
    # のため、ロボット材質の摩擦 = 実効接触摩擦になる。よってここを比較的広めに振ることで
    # 「滑る床〜よく効く床」まで様々な地面摩擦で起き上がれるようにする。起き上がりは全身が
    # 接地するので body_names=".*" で全身に適用する。バケット数も範囲拡大に合わせて増やす。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # 実は MuJoCo/実フィールドの実効摩擦は μ≈0.7-1.0 と高い (地面 friction 0.7 + 足
            # デフォルト 1.0)。MuJoCo で暴れていた真因は摩擦ではなく脚トルク上限の不一致だった
            # ため、摩擦は実フィールド μ を含む範囲 (0.3-1.0) に戻す。低摩擦側も残して DR の
            # ロバスト性は確保しつつ、実機の μ をカバーする。
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # 起き上がり: 仰向けの寝た姿勢からエピソードを開始する。
    # (velocity/rough の reset_root_state_uniform は立位からの reset なので差し替え)
    reset_base = EventTerm(
        func=reset_root_state_prone_supine,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14), "roll": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "lying_height": 0.2,
            # うつ伏せの確率を 0 にして、学習対象を仰向け起き上がりに限定する。
            "prone_prob": 0.0,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(7.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """起き上がり (get-up) 用の報酬設定。

    weight はすべて初期値であり要チューニング (reward logger で桁を合わせること)。
    地面高さ補正のため高さ系報酬には height_scanner を渡す (rough 地形対応)。
    """

    # ------------------------------------------------------------------
    # -- タスク報酬 (起き上がり)
    # ------------------------------------------------------------------
    # 明確な失敗終了には一回限りの大罰を与える。step_dt=0.02なので実効-400。
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-20000.0,
        params={"term_keys": "fall_failure"},
    )
    # foot_linkを0.15mより高く上げる動作は起き上がりとして不適切なので、専用終了時に
    # fall_failureより大きい一回限りのペナルティを与える。
    foot_height_failure_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-30000.0,
        params={"term_keys": "foot_height_failure"},
    )
    # 頭だけを0.25m以上へ上げ、最低Hipが0.20m以下に残る座位はブリッジ不成立として
    # 専用終了する。weight=-30000は終了ステップで実効-600の一回罰になる。
    head_hip_height_failure_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-30000.0,
        params={"term_keys": "head_hip_height_failure"},
    )
    # 成功せず時間切れになった場合は、頭高さにかかわらずタスク失敗として罰する。
    # 座位で頭だけ0.6m以上にして待つ抜け道を許さない。
    # 設定重み -500 は step_dt 適用後、時間切れステップで実効 -10.0 になる。
    timeout_failure_penalty = RewTerm(
        func=timeout_below_head_height,
        weight=-500.0,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "max_head_height": None,
            "timeout_term_name": "time_out",
            "success_term_name": "success",
        },
    )
    # 時間による強制終了は行わないが、30秒終了時にも頭またはTrunkが0.3m以下なら追加で強く罰する。
    # 通常の未成功timeoutペナルティと加算し、低姿勢で待ち続ける局所解を不利にする。
    timeout_low_pose_penalty = RewTerm(
        func=timeout_below_head_height,
        weight=-10000.0,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "max_head_height": 0.3,
            "max_trunk_height": 0.3,
            "timeout_term_name": "time_out",
            "success_term_name": "success",
        },
    )
    # base 高さが前ステップより高くなった分 (進捗報酬)。値は m/step と小さいので weight 大。
    base_height_increase = RewTerm(
        func=base_height_increase,
        weight=80.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "only_increase": True,
            "require_upright": True,
        },
    )
    # base 高さそのもの。0.01m から報酬を与え、高い領域ほど増分が大きい二次曲線にする。
    base_height = RewTerm(
        func=base_height,
        weight=15.0,
        params={
            "target_height": 0.6,
            "min_height": 0.01,
            "exponent": 2.0,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "require_upright": True,
        },
    )
    # 頭の高さによる初期誘導。座位での継続報酬を抑え、高価値部分は supported_head_height に移す。
    head_height = RewTerm(
        func=head_height,
        weight=8.0,
        params={
            "target_height": 0.9,
            "min_height": 0.2,
            "exponent": 3.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "require_upright": True,
        },
    )
    # 低高度でも勾配が消えない緩い指数曲線を大きなweightで使う。0.06m付近でも
    # 約+20/秒、0.35m到達時は約+276/秒となり、頭を上げる動作を明確に優先する。
    head_height_exp = RewTerm(
        func=head_height_exp_reward,
        weight=1500.0,
        params={
            "min_height": 0.0,
            "target_height": 0.9,
            "sharpness": 1.0,
            "double_scale_height": HEAD_PHASE_HEIGHT,
            "triple_scale_height": 0.7,
            "asset_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # 頭が第1フェーズ高さ0.35mより低い不足率を直接罰する。現在の頭高0.06mでは
    # 約-250/秒となり、HipやCoMだけを上げて頭を床へ残す局所解を成立させない。
    low_head_height = RewTerm(
        func=low_head_height_penalty,
        weight=-600.0,
        params={
            "minimum_height": HEAD_PHASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # 立位高さへ近づくほど指数的に増える主報酬。左右それぞれの足が体重を負担し、Trunkが
    # 直立している割合を掛けるため、上体だけ起こした座位や臀部支持では稼げない。
    supported_head_height = RewTerm(
        func=supported_head_height_exp,
        weight=250.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "min_height": 0.5,
            "target_height": 0.9,
            "sharpness": 4.0,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 頭だけでなく全リンクの質量加重CoMを上げる。全身上昇を頭単独より優先する。
    com_height = RewTerm(
        func=center_of_mass_height_reward,
        weight=180.0,
        params={"min_height": 0.12, "target_height": 0.5, "exponent": 1.0},
    )
    # 両足未接地ではhip_clearanceが0になるため、最低Hipの絶対高さを全フェーズで評価する。
    # 低姿勢の実測0.04m付近からも十分な線形勾配を与え、まず0.3mのブリッジへ誘導する。
    hip_height = RewTerm(
        func=lowest_hip_height_reward,
        weight=800.0,
        params={
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "min_height": 0.0,
            "target_height": 0.3,
            "exponent": 1.0,
        },
    )
    # 最低Hip > 最高足の高さ差を基礎報酬として常に評価し、骨盤を上げる探索を維持する。
    # 最高Headが最高Hipより0.15m上で2倍、0.30m上で3倍にし、ブリッジ後は上体上昇を優先する。
    body_height_order = RewTerm(
        func=body_height_order,
        weight=600.0,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "contact_threshold": CONTACT_THRESHOLD,
            "hip_foot_target": 0.45,
            "double_scale_head_hip_gap": 0.15,
            "triple_scale_head_hip_gap": 0.3,
        },
    )
    # 参照 standup 環境の orientation 正則化をK1向けに限定し、上下反転だけを連続的に罰する。
    # 横臥・ブリッジ中は0、地面を天に向けるほど1へ増えるため、頭支持の三点倒立を不利にする。
    inverted_posture = RewTerm(func=inverted_posture_penalty, weight=-150.0)
    # 仰向けのpitch回転は許容しつつ、片手片足を軸にした横倒れrollを連続的に罰する。
    lateral_roll = RewTerm(
        func=lateral_roll_penalty,
        weight=-100.0,
        params={"tolerance": 0.2},
    )
    # 左右どちらか一方の手足だけを天へ向ける姿勢を、左右末端高さ差で直接罰する。
    limb_height_asymmetry = RewTerm(
        func=bilateral_limb_height_asymmetry_penalty,
        weight=-80.0,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES, preserve_order=True),
            "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "tolerance": 0.05,
            "scale": 0.35,
        },
    )
    # 第1段階: 姿勢に関係なく両足を地面へ置く。ブリッジ前に直立度を要求すると勾配が消えるため、
    # uprightゲートを外す。両足接地で最大1。
    feet_ground_contact = RewTerm(
        func=feet_ground_contact,
        weight=250.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "threshold": CONTACT_THRESHOLD,
            "require_upright": False,
        },
    )
    # 専用終了に加え、離床中も両足を地面へ戻す強い連続勾配を与える。
    airborne_feet_penalty = RewTerm(
        func=airborne_feet_count,
        weight=-80.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 片足支持の横倒れを防ぐため、頭高さに関係なく片足だけの離床を強く罰する。
    one_foot_airborne_penalty = RewTerm(
        func=one_foot_airborne_above_head_height,
        weight=-250.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "min_head_height": 0.0,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 接触の有無だけでなく、各foot_linkが頭より0.1m上へ出た超過量を連続的に罰する。
    # 両足を天へ向けた現在の局所解にも発火し、足を下ろす方向へ直接勾配を与える。
    feet_above_head = RewTerm(
        func=feet_above_head_penalty,
        weight=-120.0,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "allowed_margin": 0.1,
            "scale": 0.3,
        },
    )
    # 第2段階: 足を床へ近づける連続勾配を与え、両足接地後だけ左右Hipを持ち上げる。
    # 両足を浮かせて肘・Hipで支える姿勢では、Hip高さ部分は必ず0になる。
    bridge_progress = RewTerm(
        func=supine_bridge_progress,
        weight=30.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "hip_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw"],
            ),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "hip_target_height": BRIDGE_HIP_HEIGHT,
            "foot_height_sigma": 0.08,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 両足支持を保ちながら、最も低いHipを最も高い足より持ち上げる直接的な連続報酬。
    hip_clearance = RewTerm(
        func=hip_clearance_with_foot_support_reward,
        weight=400.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "min_clearance": 0.02,
            "target_clearance": 0.35,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # lowest Hip が highest foot より低い不足量を指数的に罰する。既存hip_clearanceが
    # 0で飽和する領域にも勾配を与え、股関節を床すれすれに残す座位を直接排除する。
    hip_below_feet = RewTerm(
        func=hip_below_feet_exponential_penalty,
        weight=-150.0,
        params={
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "scale": 0.25,
            "sharpness": 4.0,
        },
    )
    # Hipと頭が第1段階へ達する高さから、後方へ倒れた上体を鉛直へ強く起こす。
    upright_posture = RewTerm(
        func=upright_posture,
        weight=600.0,
        params={"sigma": 0.25, "min_trunk_height": HAND_RELEASE_HIP_HEIGHT},
    )
    # 起き上がり判定 (CoM > 0.4m かつ 直立) 後、震えず静止しているほど高報酬。
    # 起き上がり途中は 0 なので動作は妨げない。
    # std は Σ(joint_vel²) の実スケールに合わせる。reward_manager の正規化から逆算すると立位時の
    # Σ(joint_vel²) は ~1400 (関節速度 ~8rad/s RMS = かなり震えている)。std~1000 で「震え(~1400)→
    # 静止(~300)」に勾配が出る (std=3/50 は小さすぎて exp≈0、勾配が死んでいた)。
    stand_still = RewTerm(
        func=stand_still_when_up,
        weight=20.0,
        params={"com_height_threshold": 0.4, "min_trunk_height": 0.55, "std": 1000.0},
    )
    # 全身の左右対称性 (mirror loss は使わずこの報酬で担保)。
    body_symmetry = RewTerm(
        func=body_symmetry,
        weight=0.25,
        params={"std": 0.5},
    )
    # 左右の動作が同一でない (非対称) ことへの L1 ペナルティ Σ|q_left - sign·q_right|。
    # body_symmetry (exp報酬) と相補的に、左右非対称な姿勢を直接罰する。
    # -0.5 では弱く getup が非対称化 (片側で押し上げ) したため強化 (-0.5 → -2.0)。
    body_symmetry_l1 = RewTerm(
        func=body_symmetry_l1,
        weight=-2.0,
    )
    # Hip rollの左右外向きは符号が逆なので、left+rightの鏡対称ずれと各側の過大開脚を罰する。
    # 0.35rad (~20deg) までは許容し、それ以上の大股を強く罰する。
    hip_roll_excess = RewTerm(
        func=hip_roll_excess_penalty,
        weight=-120.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Hip_Roll", "Right_Hip_Roll"],
                preserve_order=True,
            ),
            "max_abs_roll": 0.35,
            "symmetry_tolerance": 0.1,
        },
    )
    # 左右が逆符号なら相殺されるbody_symmetryとは別に、各Hip_Yawのdefault 0radからの
    # 偏差を直接評価する。0.15radまでは許容し、その後は指数的に強く罰する。
    hip_yaw_deviation = RewTerm(
        func=hip_yaw_deviation_exponential_penalty,
        weight=-80.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Hip_Yaw", "Right_Hip_Yaw"],
                preserve_order=True,
            ),
            "tolerance": 0.15,
            "scale": 0.85,
            "sharpness": 3.0,
        },
    )
    # 足裏を地面と平行 (水平) に保つペナルティ。「接地している足だけ」その水平を要求する
    # per-foot 接地ゲートに変更 (upright ゲートから変更)。接地中の足を確実に平らに踏ませ、
    # foot_linkの一部だけを接触させる踵/爪先/エッジ立ちを強く防ぐ。
    feet_flat = RewTerm(
        func=feet_flat_penalty,
        weight=-40.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "require_contact": True,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 下半身 (脚) 関節の applied torque が「設定最大トルクの10割」を超えた分へのペナルティ。
    torque_over_limit = RewTerm(
        func=joint_torque_over_limit,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]),
            "limit_ratio": 1.0,
        },
    )

    # ------------------------------------------------------------------
    # -- ペナルティ (第一弾: dof acc / dof vel / action rate / torque)
    # ------------------------------------------------------------------
    # 運動ペナルティ (acc/vel/action_rate/torque)。
    # from-scratch では強いと探索を妨げ立てなくなる (run1 で確認) ため、獲得済みの立位
    # ポリシーから resume して強め (gentle) に設定する。勢い/震え抑制は stand_still 報酬とも
    # 相補的に効く。sweep でこの強さと stand_still を調整する。
    # sweep (2026-07-27) 結果: PD 60% で resume 学習し、この運動ペナルティ群を「元の3倍 (S=3)」まで
    # 上げても起き上がり (CoM~0.48) は維持でき、末端の震え joint_speed_sq が ~1020→~33 (約97%減) に
    # なった (sim2real 向けの穏やかな起き上がり)。S=8 でも成立するが joint_speed 下限 (~33-40) は同等
    # なので S=3 を採用。checkpoint: logs/rsl_rl/k1_getup/2026-07-27_06-29-53/model_3496.pt
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-6.0e-8)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-3.0e-3)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    # 行動の二階差分 (ジャーク) ペナルティ。action_rate と独立に滑らかさを促す。
    action_smoothness_l2 = RewTerm(func=action_smoothness_l2, weight=-0.03)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-3.0e-5)
    # 機械的パワー Σ|torque × joint_vel| ペナルティ。大トルク×高速 (=勢いの良い/高エネルギー)
    # な動きをまとめて罰し、実機に優しい省エネな起き上がりを促す。値が大きいので重みは小さめ。
    joint_power = RewTerm(func=joint_power_l1, weight=-1.0e-4)

    # ------------------------------------------------------------------
    # -- ペナルティ (第二弾)
    # ------------------------------------------------------------------
    # dof pos error: rough と同じ立位姿勢をターゲットにする。K1_GETUP_CFG の脚の default 角は
    # rough (K1_LOCOMOTION_CFG) と同一なので、脚関節の default からの偏差 = rough 立位姿勢の誤差。
    # ただし上体が概ね垂直 (roll・pitch がともに 30° 以内) のときのみ適用し、寝姿勢からの
    # 起き上がり途中の大きな関節運動は罰しない。
    dof_pos_error = RewTerm(
        func=joint_deviation_l1_when_upright,
        weight=-0.5,
        params={
            "max_tilt_deg": 30.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]),
        },
    )
    # 腕の姿勢誤差: 立位時に腕を default (体側に下ろした姿勢) へ戻す。
    # com_height の可視化で、脚は default crouch のままなのに CoM が ~0.72m と高く、
    # dof_pos_error(脚)≈0 だったことから、over-extension は「腕を上げている」ことが原因と判明。
    # head_height 稼ぎで腕が上がり重心が不自然に高くなるのを抑える。
    # 起き上がり途中 (寝ている間) は腕で床を押せるよう upright ゲートで無効化する。
    arm_pos_error = RewTerm(
        func=joint_deviation_l1_when_upright,
        weight=-0.5,
        params={
            "max_tilt_deg": 30.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"],
            ),
        },
    )
    # 頭高0.6mで急に発火する罰を避けるため、以下3項は一時無効化する。
    # standing_shoulder_pitch_deviation = RewTerm(
    #     func=joint_deviation_l1_above_head_height,
    #     weight=-300.0,
    #     params={
    #         "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["ALeft_Shoulder_Pitch", "ARight_Shoulder_Pitch"],
    #             preserve_order=True,
    #         ),
    #         "minimum_head_height": 0.6,
    #     },
    # )
    # standing_hip_roll_deviation = RewTerm(
    #     func=joint_deviation_l1_above_head_height,
    #     weight=-300.0,
    #     params={
    #         "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["Left_Hip_Roll", "Right_Hip_Roll"],
    #             preserve_order=True,
    #         ),
    #         "minimum_head_height": 0.6,
    #     },
    # )
    # standing_hip_yaw_deviation = RewTerm(
    #     func=joint_deviation_l1_above_head_height,
    #     weight=-300.0,
    #     params={
    #         "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["Left_Hip_Yaw", "Right_Hip_Yaw"],
    #             preserve_order=True,
    #         ),
    #         "minimum_head_height": 0.6,
    #     },
    # )
    # arm_pos_errorは直立時のみなので、起き上がり途中にも肘4関節を既定角0rad付近へ保つ。
    # 0.35radまでは腕先支持に必要な動きを許容し、それ以上の逸脱だけを指数的に罰する。
    elbow_deviation = RewTerm(
        func=elbow_deviation_exponential_penalty,
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Elbow_Pitch", ".*_Elbow_Yaw"],
            ),
            "tolerance": 0.35,
            "scale": 1.5,
            "sharpness": 3.0,
        },
    )
    # 接地している足が滑ることへのペナルティ。低摩擦で滑って起き上がる動きを抑え、
    # 足を「踏ん張らずに置く」摩擦非依存の動きを促す。MuJoCo で暴れていたので更に強化 (-0.5 → -1.0)。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )
    # 足が地面を鉛直に押す力(法線反力)の「絶対値」を体重比で報酬 (0〜1.0)。増分報酬と違い、
    # 「足に荷重ゼロ = 暴れ」状態から「足で体重を支える」状態への明確な勾配を常時与えるので、
    # 摩擦に頼らず足で立つことを最優先させる。MuJoCo で暴れる問題への主対策。
    feet_vertical_force = RewTerm(
        func=feet_vertical_force,
        weight=50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "max_fraction": 1.0,
            "flatness_sigma": 0.25,
            "min_hip_clearance": 0.02,
            "target_hip_clearance": 0.3,
            # ブリッジ形成には仰向け段階から足荷重が必要。
            "require_upright": False,
        },
    )
    # 正常なブリッジ準備では膝がHipより高くなるため、最低Hip 0.30m以下では無効にする。
    # 0.30〜0.45mで段階的に有効化し、ブリッジ後も膝を高く残す姿勢だけを罰する。
    support_height_order = RewTerm(
        func=knee_above_trunk_and_hips_penalty,
        weight=-100.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "knee_cfg": SceneEntityCfg("robot", body_names=SHANK_BODY_NAMES),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "contact_threshold": CONTACT_THRESHOLD,
            "allowed_margin": 0.02,
            "scale": 0.25,
            "allow_below_hip_height": HAND_RELEASE_HIP_HEIGHT,
            "full_above_hip_height": BRIDGE_HIP_HEIGHT,
        },
    )
    # ブリッジ後、左右のうち荷重が弱い足へ体重が移った正味進捗を報酬化する。
    # 符号付き差分なので、荷重を抜いて踏み直す振動では累積報酬を稼げない。
    foot_load_progress = RewTerm(
        func=balanced_foot_load_progress,
        weight=40.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "hip_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw"],
            ),
            "min_hip_height": 0.3,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 頭0.35m・最低Hip 0.30mまでは、両足と手先を接地したまま両高さを上げるほど増益する。
    # 両閾値へ到達した後は、両手を離して頭0.9m・最低Hip 0.5mへ上げるほど増益する。
    hand_contact_phase = RewTerm(
        func=hand_support_by_head_and_hip_height,
        weight=1000.0,
        params={
            "hand_sensor_names": HAND_GROUND_SENSOR_NAMES,
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hand_cfg": SceneEntityCfg("robot", body_names=HAND_BODY_NAMES, preserve_order=True),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "head_height_threshold": HAND_RELEASE_HEAD_HEIGHT,
            "hip_height_threshold": HAND_RELEASE_HIP_HEIGHT,
            "target_head_height": 0.9,
            "target_hip_height": 0.5,
            "distal_tip_offset": 0.17,
            "distal_height_sigma": 0.04,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 手支持フェーズと同じ幾何条件で膝目標を切り替える。低姿勢ではπ/2 radまで屈曲して
    # 足を身体へ引き寄せ、頭高0.35mかつ最低Hip 0.30m到達後は
    # 立位既定角0.52radへ伸展する。左右の低い達成度を使い片膝だけの抜け道を防ぐ。
    knee_flexion_extension = RewTerm(
        func=staged_knee_flexion_extension_reward,
        weight=240.0,
        params={
            "knee_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Knee_Pitch", "Right_Knee_Pitch"],
                preserve_order=True,
            ),
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "flexion_target": 1.5707963268,
            "extension_target": 0.52,
            "flexion_tolerance": 1.3,
            "extension_tolerance": 0.7,
            "head_release_height": HAND_RELEASE_HEAD_HEIGHT,
            "hip_release_height": HAND_RELEASE_HIP_HEIGHT,
        },
    )
    # ブリッジ成立後はHip Pitchを前屈(-1.0rad)へ誘導して頭をHipより上へ運び、
    # 頭高0.35〜0.70mの上昇に合わせて立位既定角-0.26radへ連続的に戻す。
    hip_pitch_squat_to_stand = RewTerm(
        func=staged_hip_pitch_squat_to_stand_reward,
        weight=600.0,
        params={
            "hip_pitch_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Hip_Pitch", "Right_Hip_Pitch"],
                preserve_order=True,
            ),
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hip_body_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "minimum_hip_height": HAND_RELEASE_HIP_HEIGHT,
            "transition_head_height": HAND_RELEASE_HEAD_HEIGHT,
            "standing_head_height": 0.7,
            "squat_target": -1.0,
            "standing_target": -0.26,
            "tolerance": 1.5,
        },
    )
    # K1はTrunkが親、脚が子なので正のHip Pitchは、直立した脚に対して上体を後方へ倒す。
    # 頭とHipが第1段階高さへ達した後、片側でも正方向へ折れる姿勢を直接罰する。
    backward_hip_pitch_fold = RewTerm(
        func=backward_hip_pitch_fold_penalty,
        weight=-600.0,
        params={
            "hip_pitch_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Left_Hip_Pitch", "Right_Hip_Pitch"],
                preserve_order=True,
            ),
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hip_body_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "minimum_head_height": HAND_RELEASE_HEAD_HEIGHT,
            "minimum_hip_height": HAND_RELEASE_HIP_HEIGHT,
            "allowed_positive_pitch": 0.0,
            "scale": 1.2,
        },
    )
    # standup環境のreward_vertical_alignmentをK1向けに移植。両足中点の鉛直軸へ
    # Head・Trunk・左右Hip中点を揃え、横へ折れたまま高さだけ稼ぐ姿勢を不利にする。
    vertical_alignment = RewTerm(
        func=vertical_alignment_to_support_reward,
        weight=60.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "min_head_height": HEAD_PHASE_HEIGHT,
            "max_radius": 0.25,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 肘側Arm_3で地面を支える動作を罰する。hand_linkが未接地なら最大2倍にし、
    # 同じ腕支持でも接触点を前腕から末端リンクへ移すよう誘導する。
    proximal_arm_contact = RewTerm(
        func=proximal_arm_contact_penalty,
        weight=-80.0,
        params={
            "proximal_sensor_names": PROXIMAL_ARM_GROUND_SENSOR_NAMES,
            "hand_sensor_names": HAND_GROUND_SENSOR_NAMES,
            "contact_threshold": CONTACT_THRESHOLD,
            "contact_presence_scale": 0.5,
        },
    )
    # 参考実装の vertical_alignment と feet_spacing を統合。
    # 頭が上がり両足が接地した後、支持基底の中央へ頭・Trunkを揃え、自然な足幅を促す。
    standing_geometry = RewTerm(
        func=standing_support_geometry,
        weight=8.0,
        params={
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "foot_cfg": SceneEntityCfg("robot", body_names=".*_foot_link", preserve_order=True),
            "min_head_height": HEAD_PHASE_HEIGHT,
            "contact_threshold": CONTACT_THRESHOLD,
            "target_foot_spacing": 0.2,
            "alignment_sigma": 0.15,
            "spacing_sigma": 0.08,
        },
    )
    # 初期の寝姿勢では膝接地を許容するが、頭が0.25mを超えると段階的に有効化し、
    # 0.35m以上の膝立ちは体重比接触力で強く罰する。
    shank_contact_phase = RewTerm(
        func=staged_ground_contact_penalty,
        weight=-80.0,
        params={
            "sensor_names": SHANK_GROUND_SENSOR_NAMES,
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "allow_below_head_height": 0.25,
            "full_above_head_height": HEAD_PHASE_HEIGHT,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 足に荷重を移した後もTrunkを床へ残す姿勢を罰する。初期仰向けで足荷重がなければ0。
    trunk_ground_contact = RewTerm(
        func=trunk_contact_under_foot_load_penalty,
        weight=-100.0,
        params={
            "trunk_sensor_names": TRUNK_GROUND_SENSOR_NAMES,
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # 立位に近づいた後は、両足接地かつ両手離床を明示的に評価する。
    hands_free = RewTerm(
        func=hands_free_when_upright,
        weight=20.0,
        params={
            "hand_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hand_link"),
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "min_height": 0.55,
            "max_tilt_deg": 25.0,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    standing_success = RewTerm(
        func=standing_success_reward,
        weight=120.0,
        params={
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "hand_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hand_link"),
            "min_height": 0.6,
            "max_tilt_deg": 20.0,
            "max_lin_speed": 0.2,
            "max_ang_speed": 0.4,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )
    # base の角速度 (roll/pitch) ペナルティ。
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    # base の水平方向 (xy) 線速度ペナルティ (その場で起き上がり、横滑りを抑える)。
    base_lin_vel_xy_l2 = RewTerm(func=base_lin_vel_xy_l2, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    起き上がりでは開始時に胴体が接地しているため、歩行の base_contact (胴体接地で終了) は
    使えない (即終了してしまう)。時間切れまたは安定立位の成功で終了させる。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # reset直後の物理整定に0.25秒だけ猶予を与え、その後はいずれかの足が地面から
    # 0.15mを超えた時点で不適切な起き上がりとして終了する。
    foot_height_failure = DoneTerm(
        func=foot_height_above_limit,
        params={
            "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "maximum_height": 0.15,
            "grace_s": 0.25,
        },
    )
    # 頭が0.25mへ達した時点で最低Hipが0.20m以下なら、頭だけを起こした座位として終了する。
    head_hip_height_failure = DoneTerm(
        func=head_high_while_hip_low,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "hip_cfg": SceneEntityCfg("robot", body_names=".*_Hip_Pitch"),
            "minimum_head_height": 0.25,
            "minimum_hip_height": 0.2,
            "grace_s": 0.25,
        },
    )
    # 初期1秒を過ぎてTrunk離床後も頭が接地していれば終了。低姿勢の時間制限は設けない。
    fall_failure = DoneTerm(
        func=fall_above_head_height,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "head_sensor_names": HEAD_GROUND_SENSOR_NAMES,
            "hip_sensor_names": HIP_GROUND_SENSOR_NAMES,
            "foot_sensor_names": FOOT_GROUND_SENSOR_NAMES,
            "trunk_sensor_names": TRUNK_GROUND_SENSOR_NAMES,
            "hip_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_Hip_Pitch",
            ),
            "shoulder_cfg": SceneEntityCfg(
                "robot",
                body_names=["Left_Arm_1", "Right_Arm_1"],
            ),
            "min_head_height": HEAD_PHASE_HEIGHT,
            "trunk_contact_head_height": HEAD_PHASE_HEIGHT,
            "bridge_hip_height": BRIDGE_HIP_HEIGHT,
            "min_bridge_upper_body_height": 0.2,
            "head_support_grace_s": 1.0,
            "min_head_contact_hip_height": 0.3,
            "max_trunk_height": 0.3,
            "contact_threshold": CONTACT_THRESHOLD,
            "head_contact_threshold": HEAD_CONTACT_THRESHOLD,
        },
    )
    success = DoneTerm(
        func=stable_standing,
        params={
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "hand_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hand_link"),
            "stable_time_s": 0.5,
            "min_height": 0.6,
            "max_tilt_deg": 20.0,
            "max_lin_speed": 0.2,
            "max_ang_speed": 0.4,
            "contact_threshold": CONTACT_THRESHOLD,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # [ロギング専用] 全身 CoM (重心) 高さ [m] を Curriculum/com_height として TensorBoard に出す。
    # 立ち上がりの進捗を実寸で確認するための可視化。報酬・環境には影響しない。
    com_height = CurrTerm(func=log_com_height)
    # [ロギング専用/診断] 姿勢の実態把握用。すべて報酬・環境に影響しない。
    #   foot_height: 足リンクの平均 z [m] (~0.03=接地, 大きい=浮いている/爪先立ち)
    #   trunk_tilt_deg: Trunk の鉛直からの傾き [deg] (0=直立, 90=横倒れ)
    foot_height = CurrTerm(func=log_mean_body_z, params={"body_name": ".*_foot_link"})
    # fall_failureの幾何判定に使う実値。head_max_heightはhead_height_expの入力値でもある。
    hip_min_height = CurrTerm(func=log_min_body_z, params={"body_name": ".*_Hip_Pitch"})
    head_min_height = CurrTerm(func=log_min_body_z, params={"body_name": "Head.*"})
    head_max_height = CurrTerm(func=log_max_body_z, params={"body_name": "Head.*"})
    # 正値なら足が頭より高い。feet_above_headの無罰域は0.1m以下。
    foot_minus_head_height = CurrTerm(func=log_foot_minus_head_height)
    # 負値ならHipがfootより低く、hip_below_feetが発火する。
    hip_minus_foot_height = CurrTerm(func=log_hip_minus_foot_height)
    # 0付近なら頭とHipが同じ高さで上体が水平。body_height_orderの第1因子を直接確認する。
    head_minus_hip_height = CurrTerm(func=log_head_minus_hip_height)
    elbow_deviation_rad = CurrTerm(
        func=log_mean_joint_abs_deviation,
        params={"joint_name": ".*_Elbow_(Pitch|Yaw)"},
    )
    shoulder_pitch_deviation_rad = CurrTerm(
        func=log_mean_joint_abs_deviation,
        params={"joint_name": ".*_Shoulder_Pitch"},
    )
    hip_roll_deviation_rad = CurrTerm(
        func=log_mean_joint_abs_deviation,
        params={"joint_name": ".*_Hip_Roll"},
    )
    hip_yaw_deviation_rad = CurrTerm(
        func=log_mean_joint_abs_deviation,
        params={"joint_name": ".*_Hip_Yaw"},
    )
    knee_angle_rad = CurrTerm(
        func=log_mean_joint_position,
        params={"joint_name": ".*_Knee_Pitch"},
    )
    hip_pitch_angle_rad = CurrTerm(
        func=log_mean_joint_position,
        params={"joint_name": ".*_Hip_Pitch"},
    )
    shoulder_min_height = CurrTerm(func=log_min_body_z, params={"body_name": ".*_Arm_1"})
    trunk_tilt_deg = CurrTerm(func=log_trunk_tilt_deg)
    #   joint_speed_sq: Σ(joint_vel²) (震えの大きさ; stand_still 学習で下がるはず)
    joint_speed_sq = CurrTerm(func=log_joint_speed_sq)


##
# Environment configuration
##


@configclass
class K1GetupEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the K1 get-up environment (self-contained)."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        for sensor_name in HIP_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        for sensor_name in FOOT_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        for sensor_name in HAND_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        for sensor_name in PROXIMAL_ARM_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        for sensor_name in SHANK_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        for sensor_name in TRUNK_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        for sensor_name in HEAD_GROUND_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # ------------------------------------------------------------------
        # K1 固有の適合 (歩行側 rough_env_cfg.__post_init__ と同等の body 名調整)
        # ------------------------------------------------------------------
        # Scene: 22 自由度の全身ロボットを配置する。
        self.scene.robot = K1_GETUP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

        # events: velocity 由来の "base" 参照を K1 の "Trunk" に差し替える。
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_com.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]

        # ------------------------------------------------------------------
        # 起き上がり学習は平地から始めるのが定石 (rough + 段差では寝姿勢から立てず、
        # 報酬バランスも観測できない)。平地化し地形カリキュラムも無効にする。
        # NOTE: rough 地形で学習したくなったらこのブロックを外すだけで戻せる。
        # ------------------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # 平地なので地面高さは z=0。高さ系報酬は height_scanner 補正を使わない (None)。
        self.rewards.base_height_increase.params["sensor_cfg"] = None
        self.rewards.base_height.params["sensor_cfg"] = None
        self.rewards.head_height.params["sensor_cfg"] = None
        self.rewards.head_height_exp.params["sensor_cfg"] = None
        self.rewards.low_head_height.params["sensor_cfg"] = None


@configclass
class K1GetupEnvCfg_PLAY(K1GetupEnvCfg):
    """Play / 評価用 (少数 env・ドメインランダム化オフ)。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 起き上がり動作を繰り返し見せるため短めに (getup ~1.5s + 少し保持 → reset)。
        # calm ポリシーは立位で殆ど動かないので長い episode だと「停止」に見えてしまう。
        self.episode_length_s = 5.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
