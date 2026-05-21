# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg, DelayedPDActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .velocity_env_cfg import (
    JOINT_NAMES_K1,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

# K1専用のMDP関数 (位相報酬 + 位相観測)
# 注意: これらの関数が .mdp フォルダ内に存在することを確認してください
from .mdp import feet_phase, phase_obs, feet_height_bezier
from .mdp.rewards import feet_close_penalty, feet_parallel_to_ground, minimum_height, foot_clearance_ji, foot_clearance_ji_pen, feet_stride_length

##
# 基本設定
##
_PHASE_FREQ: float = 1.4  # Hz (歩行周期)
_COMMAND_THRESHOLD: float = 0.05 # コマンド速度がこれ未満のときは停止とみなす
_STANCE_RATIO: float = 0.55 # 接地時間の割合

_K1_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../../../assets_soccer/booster_robotics_robots/K1/K1_locomotion.urdf",
)

##
# K1 robot asset configuration
##

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
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Knee_Pitch"],
            effort_limit={".*_Hip_Pitch": 68.0, ".*_Hip_Roll": 76.0, ".*_Hip_Yaw": 38.3, ".*_Knee_Pitch": 112.0},
            velocity_limit={".*_Hip_Pitch": 14.66, ".*_Hip_Roll": 12.57, ".*_Hip_Yaw": 17.59, ".*_Knee_Pitch": 12.57},
            # stiffness={".*_Hip_Pitch": 30.20098947, ".*_Hip_Roll": 21.44796105, ".*_Hip_Yaw": 17.84601339, ".*_Knee_Pitch": 60.40197893},
            # damping={".*_Hip_Pitch": 90.6029684, ".*_Hip_Roll": 64.34388314, ".*_Hip_Yaw": 53.53804017, ".*_Knee_Pitch": 120.8039579},
            stiffness={".*_Hip_.*": 200.0, ".*_Knee_Pitch": 200.0},
            damping={".*_Hip_.*": 5.0 , ".*_Knee_Pitch": 5.0},
            armature={".*_Hip_Pitch": 0.0478125,".*_Hip_Roll": 0.0339552 , ".*_Knee_Pitch": 0.095625, '.*_Hip_Yaw': 0.0282528},
            min_delay=1,
            max_delay=3,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit=38.3,
            velocity_limit=17.59,
            # stiffness=17.84601339,
            # damping=53.53804017,
            stiffness=50.84601339,
            damping=3.0,
            armature=0.0282528,
            min_delay=1,
            max_delay=3,
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
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel,noise=Unoise(n_min=-1.5, n_max=1.5),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    actions = ObsTerm(func=mdp.last_action)
    
    # 整理した位相観測
    gait_phase = ObsTerm(func=phase_obs, params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD}) # 頻度は適宜調整

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
    gait_phase = ObsTerm(func=phase_obs, params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD})

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
        weight=1.2,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # --- 位相ベースの歩行報酬 (重要) ---
    # 空中時間報酬を0にし、位相報酬をメインにする
    feet_phase = RewTerm(
        func=feet_phase,
        weight=2.0, # 位相に合わせて足を動かすことへの報酬
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "command_name": "base_velocity",
            "phase_freq": _PHASE_FREQ,
            "stance_ratio": _STANCE_RATIO,
            "cmd_threshold": _COMMAND_THRESHOLD,
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
        weight=-1.0,
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
    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.09,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    # joint_deviation_arm = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_Pitch",".*_Shoulder_Roll",".*_Elbow_Pitch",".*_Elbow_Yaw"])},
    # )

    base_height_penalty = RewTerm(
        func=minimum_height,
        weight=-1.0,
        params={
            "min_height": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": None, 
        },
    )
    feet_close_penalty = RewTerm(
        func=feet_close_penalty,
        weight=-15.0,
        params={
            "feet_distance_threshold": 0.06,
        },
    )

    feet_parallel_to_ground = RewTerm(
        func=feet_parallel_to_ground,
        weight=30.0,
        params={
            "sigma": 0.08
        },
    )

    foot_clearance_ji_pen = RewTerm(
        func=foot_clearance_ji_pen,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "target_clearance": 0.09,
            "phase_freq": _PHASE_FREQ,
            "stance_ratio": _STANCE_RATIO,
            "cmd_threshold": _COMMAND_THRESHOLD,
        },
    )

# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------

@configclass
class K1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: K1Rewards = K1Rewards()
    observations: K1ObservationsCfg = K1ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.robot = K1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

        # Randomization
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_com.params["asset_cfg"].body_names = ["Trunk"]


        # Rewards の微調整
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw", ".*_Shank"],
                ),
                "threshold": 1.0,
            },
        )
        self.rewards.flat_orientation_l2.weight = -20.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Ankle_.*"]
        )
        self.rewards.dof_torques_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Ankle_.*"]
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


@configclass
class K1RoughEnvCfg_PLAY(K1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        