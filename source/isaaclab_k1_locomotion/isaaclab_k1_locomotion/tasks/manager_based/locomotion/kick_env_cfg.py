# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import  DelayedPDActuatorCfg
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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg,RigidObjectCfg 
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg,MassPropertiesCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .velocity_env_cfg import (
    JOINT_NAMES_K1,
)

# K1専用のMDP関数 (ローカル .mdp にのみ定義されているもの)
from .mdp.observations import ball_pos_rel, ball_vel
from .mdp.rewards import minimum_height, touch_ball, feet_close_penalty ,ball_velocity_toward_target, force_touch_ball_downhalf, fix_stance_foot_pos , base_lin_vel_xy_l2, base_ang_vel_z_l2, foot_height_penalty, ball_distance
from .mdp.terminations import time_after_ball_contact
from .mdp.curriculums import ball_max_speed_curriculum, kick_direction_command_curriculum


##
# 基本設定
##

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
        pos=(0.0, 0.0, 0.53),
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

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
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
    robot: ArticulationCfg = K1_LOCOMOTION_CFG

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=0.11,  # 11cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
            ),
            mass_props=MassPropertiesCfg(
                mass=0.45,  # 450g
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.7,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30,0.0,0.055)),
    )
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    contact_balls_right = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_foot_link", # 足に当たった場合を検知する
                                        update_period=0.0,
                                        history_length=1,
                                        track_air_time=True,
                                        filter_prim_paths_expr=[
                                        "{ENV_REGEX_NS}/SoccerBall",
                                    ])
    contact_balls_left = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_foot_link", # 足に当たった場合を検知する
                                        update_period=0.0,
                                        history_length=1,
                                        track_air_time=True,
                                        filter_prim_paths_expr=[
                                        "{ENV_REGEX_NS}/SoccerBall",
                                    ])
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # なんか3~10に設定しているはずなのに12とかが引かれるんだが...。
    target_pos = mdp.commands.commands_cfg.UniformPose2dCommandCfg(
                        ranges=mdp.commands.commands_cfg.UniformPose2dCommandCfg.Ranges((0.2,1.0),
                                                                                        (-1.0,1.0),
                                                                                        (-math.pi/3, math.pi/3)),
                        simple_heading=False,
                        debug_vis=True,
                        resampling_time_range=(12.,12.), 
                        asset_name="robot"
                    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINT_NAMES_K1,preserve_order=True, scale=0.5, use_default_offset=True)

@configclass
class K1PolicyCfg(ObsGroup):
    """Actor（ポリシー）用：実機で得られる情報のみ。線速度は含めない。"""
    # 線速度(base_lin_vel)は削除
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel,noise=Unoise(n_min=-1.5, n_max=1.5),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    actions = ObsTerm(func=mdp.last_action)
    ball_pos_rel = ObsTerm(func=ball_pos_rel, noise=Unoise(n_min=-0.08, n_max=0.08))


    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class K1CriticCfg(ObsGroup):
    """Critic（価値関数）用：特権情報（真の線速度など）を含める。"""
    # 基本情報はActorと同じ（ノイズなし）
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel,params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_K1, preserve_order=True)})
    actions = ObsTerm(func=mdp.last_action)
    ball_pos_rel = ObsTerm(func=ball_pos_rel)
    ball_vel = ObsTerm(func=ball_vel)  # これだけ特権情報


    def __post_init__(self):
        self.enable_corruption = False # Criticにノイズは不要
        self.concatenate_terms = True

@configclass
class K1ObservationsCfg:
    policy: K1PolicyCfg = K1PolicyCfg()
    critic: K1CriticCfg = K1CriticCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height_below_minimum = DoneTerm(mdp.root_height_below_minimum,params={"minimum_height": 0.40})
    bad_orientation = DoneTerm(func=mdp.bad_orientation,params={"limit_angle": 0.4})
    # ボールに最初に触れてから3秒経過したらエピソード終了
    after_ball_contact_timeout = DoneTerm(
        func=time_after_ball_contact,
        time_out=True,
        params={"time_after_contact": 5.0},
    )

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 0.8),
            "dynamic_friction_range": (0.7, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.5, 1.5),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
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
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.7, 1.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.), "yaw": (0.0, 0.0)},
            "pose_range": {
                "x": (0., 0.),
                "y": (0.0, 0.0),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.), "yaw": (0.0, 0.0)},
            "pose_range": {
                "x": (-0.15, 0.05),   # ここのx,y,zはdefault_posに追加される。
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "asset_cfg" :  SceneEntityCfg("soccer_ball")
        },
    )


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    # 上半身の傾き (projected_gravity の xy 成分の L2 二乗) にペナルティ。0:完全直立, 1:横倒し。
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.0)
    base_lin_vel_xy = RewTerm(func=base_lin_vel_xy_l2, weight=-5.0)  # 動かないで蹴って欲しいので線速度にペナルティを課す
    base_ang_vel_z = RewTerm(func=base_ang_vel_z_l2, weight=-5.0)  # yaw 回転を抑えて身体の向きを保つ
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    foot_close_penalty = RewTerm(
        func=feet_close_penalty,
        weight=-15.5,
        params={
            "feet_distance_threshold": 0.08,
        }
    )
    base_height_penalty = RewTerm(
        func=minimum_height,
        weight=-1.0,
        params={
            "min_height": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": None,
        },
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.09,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )
    # タスク報酬
    touch_ball=RewTerm(func=touch_ball,weight=10.0)
    ball_distance=RewTerm(func=ball_distance, weight=5.0)
    # ball_speed と ball_command_tracking の加算は局所解 (速いが方向無視 / 方向正解だが遅い) を生むため、
    # 目標方向への速度成分一本に統合した。
    # max_speed は学習初期にペナルティに負けないよう低めに設定 (デフォルト 6.0 → 2.0)。
    ball_velocity_toward_target = RewTerm(
        func=ball_velocity_toward_target,
        weight=35.0,
        params={"max_speed": 2.0},
    )
    force_touch_ball_downhalf_rew = RewTerm(func=force_touch_ball_downhalf,
                                      weight= -touch_ball.weight,   # 触るのを帳消しにする
                                      params = {
                                        "z_upper": 0.19,
                                        "z_lower": 0.04,
                                      }
                                    )
    fix_stance_foot_pos=RewTerm(func=fix_stance_foot_pos,
                                params={"asset_cfg":SceneEntityCfg("robot",body_names=["right_foot_link"])},
                                weight=-1.5)
    # 蹴った後に蹴り足を地面に戻させる。base config では蹴り足=左 (stance=右)。
    # subclass で body_names を上書きする。
    kick_foot_height = RewTerm(
        func=foot_height_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link"]),
            "threshold": 0.08,
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum for the kick environment."""

    # 目標方向のボール速度 EMA が threshold_ratio * max_speed を超えたら、次の max_speed に進む。
    ball_max_speed = CurrTerm(
        func=ball_max_speed_curriculum,
        params={
            "stages": [1.0, 3.0, 4.5, 6.0],
            "threshold_ratio": 0.5,
            "reward_term_name": "ball_velocity_toward_target",
            "command_name": "target_pos",
            "ball_asset": "soccer_ball",
            "ema_alpha": 0.02,
            "min_updates": 50,
        },
    )

    # 蹴り方向 (target_pos の pos_y 範囲) を段階的に拡げる。
    # Stage 0: 正面のみ → 誤差 EMA (1 - cos_sim) が閾値未満で次へ。
    kick_direction = CurrTerm(
        func=kick_direction_command_curriculum,
        params={
            "stages": [(0.0, 0.0), (-0.3, 0.3), (-0.6, 0.6), (-1.0, 1.0)],
            "error_threshold": 0.2,  # cos_sim > 0.8 (≒ 37° 以内) で進む
            "command_name": "target_pos",
            "ball_asset": "soccer_ball",
            "min_speed": 0.5,
            "ema_alpha": 0.02,
            "min_updates": 50,
        },
    )


@configclass
class KickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the kick environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: K1ObservationsCfg = K1ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # Scene: robot の prim_path を設定 (K1_LOCOMOTION_CFG は prim_path 未設定)
        self.scene.robot = K1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

@configclass
class RightKickEnvCfg(KickEnvCfg):
    """Configuration for the kick environment with right foot contact penalty."""

    def __post_init__(self):
        super().__post_init__()
        # 右脚で蹴るため、支持脚は左脚にする
        self.rewards.fix_stance_foot_pos.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["left_foot_link"])
        # 蹴り足は右なので、kick_foot_height も右に切り替える
        self.rewards.kick_foot_height.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["right_foot_link"])
        self.events.reset_ball.params["pose_range"]["x"] = (-0.20, 0.07)  # ボールの初期位置を右寄りにする

@configclass
class LeftKickEnvCfg(KickEnvCfg):
    """Configuration for the kick environment with left foot contact penalty."""

    def __post_init__(self):
        super().__post_init__()
        # 左脚で蹴るため、支持脚は右脚にする
        self.rewards.fix_stance_foot_pos.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["right_foot_link"])
        self.events.reset_ball.params["pose_range"]["x"] = (-0.07, 0.20)  # ボールの初期位置を左寄りにする