# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass

from ..rough_env_cfg import JOINT_NAMES_K1, K1CriticCfg, K1_LOCOMOTION_CFG, K1PolicyCfg
from ..mdp.rewards import stand_still_joint_deviation_l1
from . import mdp as kick_mdp


BALL_RADIUS = 0.11
BALL_SPAWN_POS = (0.20, 0.0, BALL_RADIUS)
GOAL_POS = (4.0, 0.0, BALL_RADIUS)
KICK_SUCCESS_DISTANCE = 0.15
KICK_SUCCESS_SPEED = 0.40
BALL_TRAVEL_DISTANCE_THRESHOLD = 1.5
POST_KICK_RECOVERY_TIME_S = 3.0
RECOVERY_TIMEOUT_S = 2.75
POST_RECOVERY_OBSERVE_TIME_S = 0.4
LEG_SELF_COLLISION_DISTANCE_THRESHOLD = 0.15
LEG_SELF_COLLISION_CONTACT_FORCE_THRESHOLD = 0.1
LEG_SELF_COLLISION_DEBUG_PAIR_METRICS = False


@configclass
class K1KickSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = K1_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    soccer_ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                metallic=0.0,
                roughness=0.6,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_SPAWN_POS),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    ball_contact_left_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    ball_contact_right_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/SoccerBall"],
    )
    leg_self_contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(Left_Hip_Yaw|Left_Shank|Left_Ankle_Cross|left_foot_link)",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Right_Hip_Yaw",
            "{ENV_REGEX_NS}/Robot/Right_Shank",
            "{ENV_REGEX_NS}/Robot/Right_Ankle_Cross",
            "{ENV_REGEX_NS}/Robot/right_foot_link",
        ],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=600.0,
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_K1,
        preserve_order=True,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    policy: K1PolicyCfg = K1PolicyCfg()
    critic: K1CriticCfg = K1CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball = EventTerm(
        func=kick_mdp.reset_ball_position,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    first_ball_contact_bonus = RewTerm(func=kick_mdp.first_ball_contact_bonus, weight=60.0)
    reward_ball_contact = RewTerm(func=kick_mdp.reward_ball_contact, weight=20.0)
    reward_ball_speed_increase = RewTerm(func=kick_mdp.reward_ball_speed_increase, weight=10.0)
    reward_ball_forward_velocity = RewTerm(func=kick_mdp.reward_ball_forward_velocity, weight=8.0)
    reward_kick_distance_progress = RewTerm(
        func=kick_mdp.reward_kick_distance_progress,
        weight=2.5,
        params={"ball_spawn_pos": BALL_SPAWN_POS, "success_distance": KICK_SUCCESS_DISTANCE},
    )
    reward_kick_speed_progress = RewTerm(
        func=kick_mdp.reward_kick_speed_progress,
        weight=1.5,
        params={"success_speed": KICK_SUCCESS_SPEED},
    )
    reward_kick_success = RewTerm(
        func=kick_mdp.reward_kick_success,
        weight=96.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    reward_post_kick_stability = RewTerm(
        func=kick_mdp.reward_post_kick_stability,
        weight=2.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    reward_recover_to_stand = RewTerm(
        func=kick_mdp.reward_recover_to_stand,
        weight=2.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
            "required_hold_time_s": 0.25,
            "recovery_time_scale_s": 2.0,
        },
    )
    reward_symmetric_posture = RewTerm(
        func=kick_mdp.reward_symmetric_posture,
        weight=3.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_joint_symmetry = RewTerm(
        func=kick_mdp.reward_joint_symmetry,
        weight=5.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_double_support = RewTerm(
        func=kick_mdp.reward_double_support,
        weight=3.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
        },
    )
    reward_feet_alignment = RewTerm(
        func=kick_mdp.reward_feet_alignment,
        weight=2.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
        },
    )
    reward_post_kick_balance = RewTerm(
        func=kick_mdp.reward_post_kick_balance,
        weight=4.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
        },
    )
    reward_post_kick_settling = RewTerm(
        func=kick_mdp.reward_post_kick_settling,
        weight=3.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_step_recovery = RewTerm(
        func=kick_mdp.reward_step_recovery,
        weight=1.4,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
        },
    )
    reward_forward_step_stability = RewTerm(
        func=kick_mdp.reward_forward_step_stability,
        weight=1.2,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
        },
    )
    reward_opposite_foot_recovery = RewTerm(
        func=kick_mdp.reward_opposite_foot_recovery,
        weight=3.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.1,
        },
    )
    reward_com_stability = RewTerm(
        func=kick_mdp.reward_com_stability,
        weight=3.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_stable_double_support = RewTerm(
        func=kick_mdp.reward_stable_double_support,
        weight=3.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_stop_after_recovery = RewTerm(
        func=kick_mdp.reward_stop_after_recovery,
        weight=5.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "required_hold_time_s": 0.25,
            "min_post_recovery_success_s": 0.0,
            "max_post_recovery_success_s": POST_RECOVERY_OBSERVE_TIME_S,
        },
    )
    reward_zero_velocity_after_settle = RewTerm(
        func=kick_mdp.reward_zero_velocity_after_settle,
        weight=5.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "required_hold_time_s": 0.25,
            "min_post_recovery_success_s": 0.0,
            "max_post_recovery_success_s": POST_RECOVERY_OBSERVE_TIME_S,
        },
    )
    reward_final_double_support = RewTerm(
        func=kick_mdp.reward_final_double_support,
        weight=5.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_feet_under_com_x = RewTerm(
        func=kick_mdp.reward_feet_under_com_x,
        weight=6.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_stance_x_alignment = RewTerm(
        func=kick_mdp.reward_stance_x_alignment,
        weight=7.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "target_max_x_separation": 0.05,
        },
    )
    reward_stance_width_y = RewTerm(
        func=kick_mdp.reward_stance_width_y,
        weight=5.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "target_stance_width_y": 0.195,
            "stance_width_std": 0.025,
        },
    )
    reward_nominal_stance_width = RewTerm(
        func=kick_mdp.reward_nominal_stance_width,
        weight=4.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    reward_yaw_stabilization = RewTerm(
        func=kick_mdp.reward_yaw_stabilization,
        weight=2.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    reward_heading_recovery = RewTerm(
        func=kick_mdp.reward_heading_recovery,
        weight=2.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    reward_stand_still = RewTerm(
        func=kick_mdp.reward_stand_still,
        weight=0.6,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    reward_base_position_hold = RewTerm(
        func=kick_mdp.reward_base_position_hold,
        weight=0.15,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    reward_posture_hold = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    reward_stand_pose = RewTerm(
        func=stand_still_joint_deviation_l1,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "cmd_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch",
                    ".*_Knee_Pitch",
                    ".*_Ankle_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                ],
            ),
        },
    )
    penalty_base_position_drift = RewTerm(func=kick_mdp.penalty_base_position_drift, weight=-0.35)
    penalty_unnecessary_walking = RewTerm(func=kick_mdp.penalty_unnecessary_walking, weight=-0.5)
    penalty_yaw_rate = RewTerm(
        func=kick_mdp.penalty_yaw_rate,
        weight=-3.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    penalty_post_kick_yaw = RewTerm(
        func=kick_mdp.penalty_post_kick_yaw,
        weight=-2.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    penalty_torso_pitch = RewTerm(
        func=kick_mdp.penalty_torso_pitch,
        weight=-1.0,
        params={"pitch_tolerance": 0.10},
    )
    penalty_hip_roll_deviation = RewTerm(
        func=kick_mdp.penalty_hip_roll_deviation,
        weight=-0.25,
    )
    penalty_hip_yaw_deviation = RewTerm(
        func=kick_mdp.penalty_hip_yaw_deviation,
        weight=-0.15,
    )
    penalty_post_kick_walking = RewTerm(
        func=kick_mdp.penalty_post_kick_walking,
        weight=-5.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    penalty_support_foot_drift = RewTerm(
        func=kick_mdp.penalty_support_foot_drift,
        weight=-1.5,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            "no_penalty_support_foot_x_drift": 0.05,
            "clear_penalty_support_foot_x_drift": 0.10,
        },
    )
    penalty_split_stance = RewTerm(
        func=kick_mdp.penalty_split_stance,
        weight=-8.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "tolerated_split_x": 0.10,
            "penalty_scale_x": 0.05,
        },
    )
    penalty_narrow_stance_width = RewTerm(
        func=kick_mdp.penalty_narrow_stance_width,
        weight=-3.0,
        params={
            "target_stance_width_y": 0.19,
        },
    )
    penalty_wide_stance_width = RewTerm(
        func=kick_mdp.penalty_wide_stance_width,
        weight=-1.0,
        params={
            "target_stance_width_y": 0.25,
        },
    )
    penalty_min_leg_link_distance = RewTerm(
        func=kick_mdp.penalty_min_leg_link_distance,
        weight=-2.5,
        params={
            "no_penalty_distance": 0.15,
            "strong_penalty_distance": 0.10,
        },
    )
    penalty_foot_overlap = RewTerm(
        func=kick_mdp.penalty_foot_overlap,
        weight=-4.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "minimum_foot_distance_xy": 0.15,
        },
    )
    penalty_final_pose_hip_deviation = RewTerm(
        func=kick_mdp.penalty_final_pose_hip_deviation,
        weight=-0.35,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
        },
    )
    penalty_single_leg_freeze = RewTerm(
        func=kick_mdp.penalty_single_leg_freeze,
        weight=-4.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.25,
            "single_leg_grace_time_s": 0.45,
        },
    )
    penalty_raised_kick_leg = RewTerm(
        func=kick_mdp.penalty_raised_kick_leg,
        weight=-6.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_recovery_delay_s": 0.20,
            "kick_leg_height_threshold": 0.14,
        },
    )
    penalty_no_kick_timeout = RewTerm(func=kick_mdp.penalty_no_kick_timeout, weight=-120.0)
    penalty_recovery_timeout = RewTerm(
        func=kick_mdp.penalty_recovery_timeout,
        weight=-60.0,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    penalty_fall = RewTerm(func=mdp.root_height_below_minimum, weight=-80.0, params={"minimum_height": 0.40})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=kick_mdp.terminate_time_out, time_out=True)
    no_kick_timeout = DoneTerm(func=kick_mdp.terminate_no_kick_timeout, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.40})
    leg_self_collision = DoneTerm(
        func=kick_mdp.terminate_leg_self_collision,
        params={
            "minimum_leg_link_distance": LEG_SELF_COLLISION_DISTANCE_THRESHOLD,
            "contact_force_threshold": LEG_SELF_COLLISION_CONTACT_FORCE_THRESHOLD,
            "log_detailed_pair_metrics": LEG_SELF_COLLISION_DEBUG_PAIR_METRICS,
        },
    )
    trunk_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )
    ball_travel_distance = DoneTerm(
        func=kick_mdp.terminate_ball_travel_distance,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "distance_threshold": BALL_TRAVEL_DISTANCE_THRESHOLD,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "min_post_success_time_s": POST_KICK_RECOVERY_TIME_S,
        },
    )
    post_kick_settle_time = DoneTerm(
        func=kick_mdp.terminate_post_kick_settle_time,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "settle_time_s": POST_RECOVERY_OBSERVE_TIME_S,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )
    recovery_timeout = DoneTerm(
        func=kick_mdp.terminate_recovery_timeout,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
        },
    )


@configclass
class CurriculumCfg:
    kick_stage = CurrTerm(
        func=kick_mdp.kick_success_rate_curriculum,
        params={
            "ball_spawn_pos": BALL_SPAWN_POS,
            "success_distance": KICK_SUCCESS_DISTANCE,
            "success_speed": KICK_SUCCESS_SPEED,
            "window_size": 128,
            "contact_rate_threshold": 0.8,
            "kick_rate_threshold": 0.7,
            "avg_ball_speed_threshold": 1.2,
            "recovery_rate_threshold": 0.8,
        },
    )


@configclass
class K1KickEnvCfg(ManagerBasedRLEnvCfg):
    scene: K1KickSceneCfg = K1KickSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 7.0
        self.viewer.eye = (6.0, 0.0, 3.0)
        self.viewer.lookat = (0.3, 0.0, 0.7)

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.ball_contact_left_foot is not None:
            self.scene.ball_contact_left_foot.update_period = self.sim.dt
        if self.scene.ball_contact_right_foot is not None:
            self.scene.ball_contact_right_foot.update_period = self.sim.dt
        if self.scene.leg_self_contact_left is not None:
            self.scene.leg_self_contact_left.update_period = self.sim.dt


@configclass
class K1KickEnvCfg_PLAY(K1KickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 8.0
