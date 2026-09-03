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

TODO (まずは env を用意した段階):
  rewards / observations / terminations / commands は歩行由来のプレースホルダである。
  起き上がり課題に合わせて (寝た姿勢からの reset、姿勢直立の報酬、頭・腕の活用など)
  後で置き換えること。
"""

import math
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
    feet_ground_contact,
    feet_ground_reaction_increase,
    feet_vertical_force,
    feet_height_low,
    non_foot_contact_penalty,
    feet_flat_penalty,
    jump_penalty,
    joint_torque_over_limit,
    joint_power_l1,
    action_smoothness_l2,
    upright_posture,
    stand_still_when_up,
    body_symmetry,
    body_symmetry_l1,
    joint_deviation_l1_when_upright,
)
from .mdp.events import reset_root_state_prone_supine
from .mdp.rewards import base_lin_vel_xy_l2
from .mdp.curriculums import log_com_height, log_mean_body_z, log_trunk_tilt_deg, log_joint_speed_sq

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

# PD ゲインのスケール。元 (100%) の値に対する倍率。環境変数 GETUP_PD_SCALE で実行時に上書き可
# (既定 0.6 = 6割)。60% では起き上がれないことが判明したため、getup が成立する PD 下限を
# 探索できるよう env var 化した。学習コマンドで `GETUP_PD_SCALE=0.8 ...` のように指定する。
_PD_SCALE = float(os.environ.get("GETUP_PD_SCALE", "0.6"))

# 脚 actuator (rough_env_cfg の delayed_pd_leg を複製)。PD は 100% 基準値 × _PD_SCALE。
# effort_limit は実機/MuJoCo デプロイのトルク上限 (k1_constants.hpp TORQUE_LIMITS =
# K1_22dof_soccer_field.xml forcerange) に一致させる。歩行から流用した高い値 (膝112 等) で
# 学習すると、デプロイ側が膝40Nm でクランプされるため MuJoCo で持ち上げ切れず暴れていた。
# 実機のトルク予算内で起き上がる動きを学習させるのが目的。
delayed_pd_leg = DelayedPDActuatorCfg(
    joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch"],
    effort_limit={".*_Hip_Pitch": 30.0, ".*_Hip_Roll": 35.0, ".*_Hip_Yaw": 20.0, ".*_Knee_Pitch": 40.0},
    velocity_limit={".*_Hip_Pitch": 14.66, ".*_Hip_Roll": 12.57, ".*_Hip_Yaw": 17.59, ".*_Knee_Pitch": 12.57},
    stiffness={".*_Hip_.*": 160.0 * _PD_SCALE, ".*_Knee_Pitch": 160.0 * _PD_SCALE},
    damping={".*_Hip_.*": 4.0 * _PD_SCALE, ".*_Knee_Pitch": 4.0 * _PD_SCALE},
    armature={".*_Hip_Pitch": 0.0478125, ".*_Hip_Roll": 0.0339552, ".*_Knee_Pitch": 0.095625, ".*_Hip_Yaw": 0.0282528},
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
            effort_limit=20.0,  # 実機/MuJoCo デプロイ (Ankle 20Nm) に一致
            velocity_limit=17.59,
            stiffness=50.0 * _PD_SCALE,
            damping=2.5 * _PD_SCALE,
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
    # 報酬で参照するのは足 (feet_ground_contact / feet_slide) と胴体 (将来用) のみ。
    # 全 body を張ると収集が CPU 律速で遅くなるので Trunk + 足だけに絞る (純粋な高速化)。
    # 足 (報酬) + 胴 + 「足以外の接地を罰する (足だけで起き上がる誘導)」用に手・膝(Shank)も張る。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(Trunk|.*_foot_link|.*_hand_link|.*_Shank)",
        history_length=3,
        track_air_time=True,
    )
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
    """Command specifications for the MDP.

    TODO: 起き上がり課題では速度コマンドは本来不要。プレースホルダとして velocity_env の
          定義を複製している。起き上がり用に置き換えること。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)
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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.3, n_max=0.3))
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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
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

    # 起き上がり: うつ伏せ or 仰向けの寝た姿勢からエピソードを開始する。
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
            "prone_prob": 0.5,
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
    # base 高さそのもの (立つほど高い、target で 1 に飽和)。
    base_height = RewTerm(
        func=base_height,
        weight=15.0,
        params={
            "target_height": 0.6,
            "min_height": 0.2,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "require_upright": True,
        },
    )
    # 頭の高さ (持ち上がるほど高い)。
    head_height = RewTerm(
        func=head_height,
        weight=25.0,
        params={
            "target_height": 0.9,
            "min_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names="Head.*"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "require_upright": True,
        },
    )
    # 足裏が接地していること (両足で 1)。完全接地を促すため強化 (1.0 → 3.0)。
    feet_ground_contact = RewTerm(
        func=feet_ground_contact,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 1.0,
            "require_upright": True,
        },
    )
    # 上体 (Trunk) が鉛直にまっすぐ。
    upright_posture = RewTerm(
        func=upright_posture,
        weight=3.0,
        params={"sigma": 0.25},
    )
    # 起き上がり判定 (CoM > 0.4m かつ 直立) 後、震えず静止しているほど高報酬。
    # 起き上がり途中は 0 なので動作は妨げない。
    # std は Σ(joint_vel²) の実スケールに合わせる。reward_manager の正規化から逆算すると立位時の
    # Σ(joint_vel²) は ~1400 (関節速度 ~8rad/s RMS = かなり震えている)。std~1000 で「震え(~1400)→
    # 静止(~300)」に勾配が出る (std=3/50 は小さすぎて exp≈0、勾配が死んでいた)。
    stand_still = RewTerm(
        func=stand_still_when_up,
        weight=4.0,
        params={"com_height_threshold": 0.4, "std": 1000.0},
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
    # 足が低い (地面に近い) ほど大きい報酬 Σexp(-10·h_foot)。足を無駄に高く上げず地面近くに
    # 保つことを促す。小さめの weight。
    feet_low = RewTerm(
        func=feet_height_low,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "scale": 10.0,
        },
    )
    # 足裏を地面と平行 (水平) に保つペナルティ。「接地している足だけ」その水平を要求する
    # per-foot 接地ゲートに変更 (upright ゲートから変更)。接地中の足を確実に平らに踏ませ、
    # 踵/爪先/エッジ立ちを防ぐ。重みも強化 (-5.0 → -8.0)。
    feet_flat = RewTerm(
        func=feet_flat_penalty,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "require_contact": True,
            "contact_threshold": 1.0,
        },
    )
    # ジャンプ (両足が同時に地面から離れる) への強いペナルティ。立位時のみ。
    jump = RewTerm(
        func=jump_penalty,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 1.0,
            "com_height_threshold": 0.4,
        },
    )
    # 下半身 (脚) 関節の applied torque が「設定最大トルクの7割」を超えた分へのペナルティ。
    torque_over_limit = RewTerm(
        func=joint_torque_over_limit,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]),
            "limit_ratio": 0.7,
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
    # 足が地面から受ける垂直反力が前ステップより増加した分への報酬。摩擦(水平)ではなく
    # 法線方向(垂直)の押し込みで体を持ち上げる動きを促す。優先度を上げるため強化 (5.0 → 10.0)。
    feet_reaction_increase = RewTerm(
        func=feet_ground_reaction_increase,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 足が地面を鉛直に押す力(法線反力)の「絶対値」を体重比で報酬 (0〜1.0)。増分報酬と違い、
    # 「足に荷重ゼロ = 暴れ」状態から「足で体重を支える」状態への明確な勾配を常時与えるので、
    # 摩擦に頼らず足で立つことを最優先させる。MuJoCo で暴れる問題への主対策。
    feet_vertical_force = RewTerm(
        func=feet_vertical_force,
        weight=20.0,  # 接地反力を重視 (10.0 → 20.0)。足で体重を支える完全接地を促す。
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "max_fraction": 1.0,
            # 上体が起きた後だけ足裏押しを要求 (寝たまま farm するのを防ぐ)。
            "require_upright": True,
        },
    )
    # 足以外 (手・膝) で地面を押して起き上がるのを抑え、「足だけで起き上がる」動きを誘導。
    # sim2real で接触モデル差が出やすい非足部の接地依存を減らす (Stage1 の主誘導)。
    non_foot_contact = RewTerm(
        func=non_foot_contact_penalty,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hand_link", ".*_Shank"]),
            "asset_cfg": SceneEntityCfg("robot"),
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
    使えない (即終了してしまう)。時間切れのみで終了させる。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


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
        self.episode_length_s = 6.0
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
