"""K1 direct-kicking task with a deployment-oriented ball belief."""

import math

import numpy as np
import torch
from .math_xyzw import get_euler_xyz, quat_rotate, quat_rotate_inverse

from .kicking_logic import KickingLogic
from .action_smoothness import action_second_difference_l2
from .ball_trajectory import build_ball_trajectory
from .vision_filter import VisionFilter
from .sensor_noise import distance_scaled_measurement_std
from .ball_visibility import horizontal_fov_mask
from .direct_kicking_observation import (
    BELIEF_STATUS_SIZE,
    HORIZON_TOKEN_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    TARGET_OBSERVATION_SIZE,
    build_horizon_tokens,
    expected_direct_kicking_observation_size,
)
from .direct_kicking_outcome import (
    latch_first_kick_step,
    normalize_xy_direction,
    one_shot_direction_reward,
    physical_kick_event,
    post_kick_termination_mask,
    post_kick_walking_pose_reward,
    sigmoid_velocity_change_scale,
    update_post_kick_phase_target,
)
from .action_delay import (
    action_delay_step_range,
    delayed_targets_for_substep,
    sample_action_delay_steps,
)
from .disturbance_schedule import sample_interval_steps
from .ego_motion import (
    forecast_constant_body_twist,
    forecast_ego_state_covariance,
    relative_state_and_covariance,
)
from .kick_foot_symmetry import (
    large_stationary_kick_attempt_candidate,
    max_foot_height_excess_penalty,
    mirrored_strike_position_progress,
    nearest_foot_distance_progress,
    nonparallel_step_penalty,
    per_foot_kick_candidates,
    select_kicking_foot_mask,
    update_edge_only_support_penalty,
    update_kick_attempt_outcome,
)
from .k1_foot_kinematics import k1_foot_contact_points_root
from .utils import apply_randomization


class DirectKickingLogic(KickingLogic):
    locomotion_observation_size = LOCOMOTION_OBSERVATION_SIZE
    belief_features_per_horizon = HORIZON_TOKEN_SIZE
    belief_status_size = BELIEF_STATUS_SIZE
    target_observation_size = TARGET_OBSERVATION_SIZE

    def _configure_ball_motion(self, cfg):
        motion_cfg = self.direct_cfg["ball_motion_randomization"]
        self.ball_speed_range = self._validated_range(
            motion_cfg["speed_range_mps"],
            "speed_range_mps",
            non_negative=True,
        )
        self.incoming_time_to_closest_range = self._validated_range(
            motion_cfg["incoming_time_to_closest_range_s"],
            "incoming_time_to_closest_range_s",
            positive=True,
        )
        self.minimum_spawn_distance = float(motion_cfg["minimum_spawn_distance_m"])
        if self.minimum_spawn_distance <= 0.0:
            raise ValueError("minimum_spawn_distance_m must be positive")
        self.stationary_spawn_distance_range = self._validated_range(
            motion_cfg["stationary_spawn_distance_range_m"],
            "stationary_spawn_distance_range_m",
            positive=True,
        )
        self.outgoing_spawn_distance_range = self._validated_range(
            motion_cfg["outgoing_spawn_distance_range_m"],
            "outgoing_spawn_distance_range_m",
            positive=True,
        )
        self.closest_approach_offset_range = self._validated_range(
            motion_cfg["closest_approach_offset_range_m"],
            "closest_approach_offset_range_m",
        )
        self.spawn_bearing_range = self._validated_range(
            motion_cfg["spawn_bearing_range_rad"],
            "spawn_bearing_range_rad",
        )
        self.incoming_probability = float(motion_cfg["incoming_probability"])
        self.stationary_probability = float(motion_cfg["stationary_probability"])

        if not 0.0 <= self.incoming_probability <= 1.0:
            raise ValueError("incoming_probability must be in [0, 1]")
        if not 0.0 <= self.stationary_probability <= 1.0:
            raise ValueError("stationary_probability must be in [0, 1]")

        minimum_spawn_distance = min(
            self.minimum_spawn_distance,
            self.outgoing_spawn_distance_range[0],
            self.stationary_spawn_distance_range[0],
        )
        maximum_offset = max(abs(value) for value in self.closest_approach_offset_range)
        if maximum_offset >= minimum_spawn_distance:
            raise ValueError(
                "closest_approach_offset_range_m must stay inside the minimum "
                "spawn distance"
            )

        vision_cfg = cfg.get("vision", {})
        half_fov = 0.5 * float(vision_cfg["fov_yaw"])
        if (
            self.spawn_bearing_range[0] <= -half_fov
            or self.spawn_bearing_range[1] >= half_fov
        ):
            raise ValueError("spawn_bearing_range_rad must stay inside vision fov_yaw")
        maximum_spawn_distance = max(
            self.minimum_spawn_distance,
            math.hypot(
                self.ball_speed_range[1] * self.incoming_time_to_closest_range[1],
                maximum_offset,
            ),
            self.outgoing_spawn_distance_range[1],
            self.stationary_spawn_distance_range[1],
        )
        minimum_visible_distance = float(vision_cfg["min_distance"])
        maximum_visible_distance = float(vision_cfg["max_distance"])
        if minimum_spawn_distance <= minimum_visible_distance:
            raise ValueError(
                "spawn distance must be greater than vision min_distance"
            )
        if maximum_spawn_distance >= maximum_visible_distance:
            raise ValueError(
                "spawn distance must be less than vision max_distance"
            )

    def _randomize_ground_material(self, cfg):
        physics_cfg = self.direct_cfg.get("physics_randomization", {})
        enabled = physics_cfg.get("enabled", True)
        friction_range = self._validated_range(
            physics_cfg.get("ground_friction_scale_range", [1.0, 1.0]),
            "ground_friction_scale_range",
            positive=True,
        )
        friction_scale = np.random.uniform(*friction_range) if enabled else 1.0
        cfg["terrain"]["static_friction"] *= float(friction_scale)
        cfg["terrain"]["dynamic_friction"] *= float(friction_scale)
        self.sampled_ground_friction_scale = float(friction_scale)
        self.sampled_ground_static_friction = cfg["terrain"]["static_friction"]
        self.sampled_ground_dynamic_friction = cfg["terrain"]["dynamic_friction"]

    def _configure_action_delay(self, cfg):
        self.action_delay_range_s = self._validated_range(
            cfg["randomization"]["action_delay_range_s"],
            "action_delay_range_s",
            positive=True,
        )
        self.action_delay_step_range = action_delay_step_range(
            self.action_delay_range_s,
            cfg["sim"]["dt"],
        )
        maximum_delay_steps = self.action_delay_step_range[1]
        decimation = int(cfg["control"]["decimation"])
        if decimation <= 0:
            raise ValueError("control.decimation must be positive")
        self.action_target_history_length = (
            math.ceil(maximum_delay_steps / decimation) + 1
        )

    def _configure_external_disturbances(self, cfg):
        randomization_cfg = cfg["randomization"]
        self.force_push_interval_range = self._validated_range(
            randomization_cfg["push_interval_s"],
            "push_interval_s",
            positive=True,
        )
        self.force_push_duration_s = float(randomization_cfg["push_duration_s"])
        if self.force_push_duration_s <= 0.0:
            raise ValueError("push_duration_s must be positive")
        if self.force_push_duration_s >= self.force_push_interval_range[0]:
            raise ValueError("push_duration_s must be shorter than push_interval_s")

        self.velocity_push_interval_range = self._validated_range(
            randomization_cfg["velocity_push_interval_s"],
            "velocity_push_interval_s",
            positive=True,
        )
        velocity_push_cfg = randomization_cfg["velocity_push_xy"]
        self.velocity_push_xy_range = self._validated_range(
            velocity_push_cfg["range"],
            "velocity_push_xy.range",
        )
        if not (
            self.velocity_push_xy_range[0] <= 0.0
            and self.velocity_push_xy_range[1] >= 0.0
        ):
            raise ValueError("velocity_push_xy.range must contain zero")
        if velocity_push_cfg["operation"] != "additive":
            raise ValueError("velocity_push_xy.operation must be additive")
        if velocity_push_cfg["distribution"] != "uniform":
            raise ValueError("velocity_push_xy.distribution must be uniform")

    def _randomize_ball_restitution(self, env_ids):
        """Resample and apply ball restitution for the selected environments."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return

        if self.ball_physics_randomization_enabled:
            restitution = self._sample_uniform(self.ball_restitution_range, count)
        else:
            restitution = torch.full(
                (count,),
                float(self.task_cfg["ball"]["restitution"]),
                device=self.device,
            )
        self.sampled_ball_restitution[env_ids] = restitution

        self._write_ball_restitution(env_ids, restitution)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_action_delay_buffers()
        self.previous_previous_actions = torch.zeros_like(self.actions)
        self.direct_previous_feet_ankle_pos_world = torch.zeros_like(self.feet_pos)
        self._init_external_disturbance_buffers()
        self._init_ankle_dof_indices()
        self._init_walking_policy_initial_state()
        self._init_contact_aware_spawn()
        self._init_edge_only_support()
        self._init_direct_perception_buffers()
        self._init_direct_outcome_buffers()

    def _init_edge_only_support(self):
        edge_positions = torch.as_tensor(
            self.task_cfg["asset"]["feet_edge_pos"],
            device=self.device,
            dtype=self.feet_pos.dtype,
        )
        toe_edges = edge_positions[:, 0] > 0.0
        heel_edges = edge_positions[:, 0] < 0.0
        if not torch.any(toe_edges).item() or not torch.any(heel_edges).item():
            raise ValueError("feet_edge_pos must contain toe and heel contact points")

        self.edge_only_support_edge_positions = edge_positions
        self.edge_only_support_toe_edges = toe_edges
        self.edge_only_support_heel_edges = heel_edges
        self.edge_only_support_steps_buf = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            device=self.device,
            dtype=torch.long,
        )

    def _init_action_delay_buffers(self):
        self.action_target_history_cursor = 0
        self.action_target_history = self.dof_pos.unsqueeze(1).repeat(
            1,
            self.action_target_history_length,
            1,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_action_delay(env_ids)

    def _reset_action_delay(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return

        self.delay_steps[env_ids] = sample_action_delay_steps(
            self.action_delay_step_range,
            count,
            self.device,
        )
        reset_targets = self.dof_pos[env_ids].unsqueeze(1)
        self.action_target_history[env_ids] = reset_targets

    def _update_delayed_dof_targets(self, dof_targets, substep_index):
        if substep_index == 0:
            self.action_target_history_cursor = (
                self.action_target_history_cursor + 1
            ) % self.action_target_history_length
            self.action_target_history[:, self.action_target_history_cursor] = (
                dof_targets
            )

        delayed_targets = delayed_targets_for_substep(
            self.action_target_history,
            self.action_target_history_cursor,
            self.delay_steps,
            substep_index,
            self.task_cfg["control"]["decimation"],
        )
        self.last_dof_targets.copy_(delayed_targets)

    def _init_external_disturbance_buffers(self):
        self.next_force_push_step = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.force_push_end_step = torch.full_like(self.next_force_push_step, -1)
        self.next_velocity_push_step = torch.zeros_like(self.next_force_push_step)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_external_disturbance_schedule(env_ids)

    def _reset_external_disturbance_schedule(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return

        current_step = int(self.common_step_counter)
        self.next_force_push_step[env_ids] = current_step + sample_interval_steps(
            self.force_push_interval_range,
            self.dt,
            count,
            self.device,
        )
        self.next_velocity_push_step[env_ids] = current_step + sample_interval_steps(
            self.velocity_push_interval_range,
            self.dt,
            count,
            self.device,
        )
        self.force_push_end_step[env_ids] = -1
        self.pushing_forces[env_ids, self.base_indice, :].zero_()
        self.pushing_torques[env_ids, self.base_indice, :].zero_()

    def _kick_robots(self):
        """Apply independently scheduled world-frame XY velocity disturbances."""
        current_step = int(self.common_step_counter)
        due_ids = (current_step >= self.next_velocity_push_step).nonzero(
            as_tuple=False
        ).flatten()
        count = due_ids.numel()
        if count == 0:
            return

        velocity_delta_xy = apply_randomization(
            torch.zeros(count, 2, device=self.device),
            self.task_cfg["randomization"]["velocity_push_xy"],
        )
        self.root_states[due_ids, 0, 7:9] += velocity_delta_xy
        self.base_lin_vel[due_ids] = quat_rotate_inverse(
            self.base_quat[due_ids],
            self.root_states[due_ids, 0, 7:10],
        )

        self._write_root_states(due_ids, robot=True)
        self.next_velocity_push_step[due_ids] = current_step + sample_interval_steps(
            self.velocity_push_interval_range,
            self.dt,
            count,
            self.device,
        )

    def _push_robots(self):
        """Apply independently scheduled finite-duration force disturbances."""
        current_step = int(self.common_step_counter)
        ended_ids = (
            (self.force_push_end_step >= 0)
            & (current_step >= self.force_push_end_step)
        ).nonzero(as_tuple=False).flatten()
        if ended_ids.numel() > 0:
            self.pushing_forces[ended_ids, self.base_indice, :].zero_()
            self.pushing_torques[ended_ids, self.base_indice, :].zero_()
            self.force_push_end_step[ended_ids] = -1

        due_ids = (current_step >= self.next_force_push_step).nonzero(
            as_tuple=False
        ).flatten()
        count = due_ids.numel()
        if count > 0:
            self.pushing_forces[due_ids, self.base_indice, :] = apply_randomization(
                torch.zeros(count, 3, device=self.device),
                self.task_cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[due_ids, self.base_indice, :] = apply_randomization(
                torch.zeros(count, 3, device=self.device),
                self.task_cfg["randomization"].get("push_torque"),
            )
            self.force_push_end_step[due_ids] = current_step + int(
                math.ceil(self.force_push_duration_s / self.dt)
            )
            self.next_force_push_step[due_ids] = current_step + sample_interval_steps(
                self.force_push_interval_range,
                self.dt,
                count,
                self.device,
            )

        self._write_external_forces()

    def _init_ankle_dof_indices(self):
        ankle_dof_names = (
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        )
        name_to_index = {name: index for index, name in enumerate(self.dof_names)}
        missing_names = [name for name in ankle_dof_names if name not in name_to_index]
        if missing_names:
            raise ValueError(
                "DirectKicking requires missing ankle joints: "
                + ", ".join(missing_names)
            )
        self.ankle_dof_indices = torch.as_tensor(
            [name_to_index[name] for name in ankle_dof_names],
            device=self.device,
            dtype=torch.long,
        )

    def _init_walking_policy_initial_state(self):
        initial_state_cfg = self.direct_cfg.get(
            "walking_policy_initial_state",
            {},
        )
        self.walking_policy_initial_state_enabled = bool(
            initial_state_cfg.get("enabled", False)
        )
        self.walking_policy_initial_state_probability = float(
            initial_state_cfg.get("probability", 0.0)
        )
        self.walking_policy_pose_blend_range = self._validated_range(
            initial_state_cfg.get("pose_blend_range", [1.0, 1.0]),
            "walking_policy_initial_state.pose_blend_range",
            non_negative=True,
        )
        if not 0.0 <= self.walking_policy_initial_state_probability <= 1.0:
            raise ValueError(
                "walking_policy_initial_state.probability must be in [0, 1]"
            )
        if self.walking_policy_pose_blend_range[1] > 1.0:
            raise ValueError(
                "walking_policy_initial_state.pose_blend_range must be in [0, 1]"
            )

        joint_angles = initial_state_cfg.get("joint_angles", {})
        if self.walking_policy_initial_state_enabled and not joint_angles:
            raise ValueError(
                "walking_policy_initial_state.joint_angles must not be empty "
                "when enabled"
            )

        default_angle = float(joint_angles.get("default", 0.0))
        self.walking_policy_default_dof_pos = torch.full_like(
            self.default_dof_pos,
            default_angle,
        )
        for dof_index, dof_name in enumerate(self.dof_names):
            matches = [
                float(angle)
                for name, angle in joint_angles.items()
                if name != "default" and name in dof_name
            ]
            if len(matches) > 1:
                raise ValueError(
                    "walking_policy_initial_state.joint_angles contains "
                    f"multiple matches for {dof_name}"
                )
            if matches:
                self.walking_policy_default_dof_pos[:, dof_index] = matches[0]

    def _init_contact_aware_spawn(self):
        spawn_cfg = self.direct_cfg.get("contact_aware_spawn", {})
        self.contact_aware_spawn_enabled = bool(
            spawn_cfg.get("enabled", False)
        )
        self.spawn_ground_clearance = float(
            spawn_cfg.get("ground_clearance_m", 0.003)
        )
        if self.spawn_ground_clearance < 0.0:
            raise ValueError(
                "contact_aware_spawn.ground_clearance_m must be non-negative"
            )

        contact_points = spawn_cfg.get("foot_contact_points")
        if self.contact_aware_spawn_enabled and not contact_points:
            raise ValueError(
                "contact_aware_spawn.foot_contact_points must not be empty "
                "when enabled"
            )
        self.spawn_foot_contact_points = torch.as_tensor(
            contact_points or [[0.0, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float,
        )
        if (
            self.spawn_foot_contact_points.ndim != 2
            or self.spawn_foot_contact_points.shape[1] != 3
        ):
            raise ValueError(
                "contact_aware_spawn.foot_contact_points must have shape [N, 3]"
            )

        leg_joint_names = (
            (
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
            ),
            (
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ),
        )
        name_to_index = {name: index for index, name in enumerate(self.dof_names)}
        missing_names = [
            name
            for leg_names in leg_joint_names
            for name in leg_names
            if name not in name_to_index
        ]
        if self.contact_aware_spawn_enabled and missing_names:
            raise ValueError(
                "contact_aware_spawn requires missing joints: "
                + ", ".join(missing_names)
            )
        self.spawn_leg_joint_indices = torch.as_tensor(
            [
                [name_to_index.get(name, 0) for name in leg_names]
                for leg_names in leg_joint_names
            ],
            device=self.device,
            dtype=torch.long,
        )

    def _reset_dofs(self, env_ids):
        count = env_ids.numel()
        nominal_pose = self.default_dof_pos.expand(count, -1).clone()
        if self.walking_policy_initial_state_enabled:
            use_walking_pose = (
                torch.rand(count, device=self.device)
                < self.walking_policy_initial_state_probability
            )
            pose_blend = self._sample_uniform(
                self.walking_policy_pose_blend_range,
                count,
            )
            pose_blend *= use_walking_pose.float()
            nominal_pose += pose_blend.unsqueeze(-1) * (
                self.walking_policy_default_dof_pos - self.default_dof_pos
            )

        self.dof_pos[env_ids] = apply_randomization(
            nominal_pose,
            self.task_cfg["randomization"].get("init_dof_pos"),
        )
        self.dof_pos[env_ids] = torch.clamp(
            self.dof_pos[env_ids],
            min=self.dof_pos_limits[:, 0],
            max=self.dof_pos_limits[:, 1],
        )
        self.dof_vel[env_ids] = apply_randomization(
            torch.zeros_like(nominal_pose),
            self.task_cfg["randomization"].get("init_dof_vel"),
        )

        self._write_dof_state(env_ids)

    def _initial_root_height(self, env_ids):
        if not self.contact_aware_spawn_enabled:
            return super()._initial_root_height(env_ids)

        foot_points_root = k1_foot_contact_points_root(
            self.dof_pos[env_ids],
            self.spawn_leg_joint_indices,
            self.spawn_foot_contact_points,
        )
        base_quat = self.root_states[env_ids, 0, 3:7]
        expanded_quat = base_quat[:, None, None, :].expand(
            -1,
            foot_points_root.shape[1],
            foot_points_root.shape[2],
            -1,
        )
        rotated_points = quat_rotate(
            expanded_quat.reshape(-1, 4),
            foot_points_root.reshape(-1, 3),
        ).reshape_as(foot_points_root)
        world_points = rotated_points.clone()
        world_points[..., :2] += self.root_states[env_ids, 0, None, None, :2]
        terrain_height = self._terrain_heights(
            world_points.reshape(-1, 3)
        ).reshape(world_points.shape[:-1])
        required_root_height = (
            terrain_height
            + self.spawn_ground_clearance
            - rotated_points[..., 2]
        )
        return required_root_height.amax(dim=(1, 2))

    def _init_direct_outcome_buffers(self):
        outcome_cfg = self.direct_cfg["outcome"]
        post_kick_termination_s = float(outcome_cfg["post_kick_termination_s"])
        self.max_kick_direction_reward = float(
            outcome_cfg["max_kick_direction_reward"]
        )
        self.kick_direction_reward_sharpness = float(
            outcome_cfg["direction_reward_sharpness"]
        )
        self.kick_velocity_change_reward_center = float(
            outcome_cfg["velocity_change_reward_center_mps"]
        )
        self.kick_velocity_change_reward_sharpness = float(
            outcome_cfg["velocity_change_reward_sharpness"]
        )
        self.walking_pose_reward_error_scale = float(
            outcome_cfg["walking_pose_reward_error_scale_rad"]
        )
        self.kick_attempt_minimum_forward_position = float(
            outcome_cfg["kick_attempt_minimum_forward_position_m"]
        )
        self.kick_attempt_minimum_forward_speed = float(
            outcome_cfg["kick_attempt_minimum_forward_speed_mps"]
        )
        self.kick_attempt_maximum_base_speed = float(
            outcome_cfg["kick_attempt_maximum_base_speed_mps"]
        )
        kick_attempt_outcome_window_s = float(
            outcome_cfg["kick_attempt_outcome_window_s"]
        )
        if post_kick_termination_s <= 0.0:
            raise ValueError("post_kick_termination_s must be positive")
        if self.max_kick_direction_reward < 0.0:
            raise ValueError("max_kick_direction_reward must be non-negative")
        if self.kick_direction_reward_sharpness <= 0.0:
            raise ValueError("direction_reward_sharpness must be positive")
        if self.kick_velocity_change_reward_center < 0.0:
            raise ValueError(
                "velocity_change_reward_center_mps must be non-negative"
            )
        if self.kick_velocity_change_reward_sharpness <= 0.0:
            raise ValueError(
                "velocity_change_reward_sharpness must be positive"
            )
        if self.walking_pose_reward_error_scale <= 0.0:
            raise ValueError(
                "walking_pose_reward_error_scale_rad must be positive"
            )
        if self.kick_attempt_minimum_forward_position < 0.0:
            raise ValueError(
                "kick_attempt_minimum_forward_position_m must be non-negative"
            )
        if self.kick_attempt_minimum_forward_speed < 0.0:
            raise ValueError(
                "kick_attempt_minimum_forward_speed_mps must be non-negative"
            )
        if self.kick_attempt_maximum_base_speed < 0.0:
            raise ValueError(
                "kick_attempt_maximum_base_speed_mps must be non-negative"
            )
        if kick_attempt_outcome_window_s <= 0.0:
            raise ValueError("kick_attempt_outcome_window_s must be positive")

        self.post_kick_duration_steps = int(
            math.ceil(post_kick_termination_s / self.dt)
        )
        self.kick_attempt_outcome_window_steps = int(
            math.ceil(kick_attempt_outcome_window_s / self.dt)
        )
        self.first_valid_kick_step = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self.post_kick_terminal_buf = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self.kick_attempt_pending_buf = torch.zeros_like(self.valid_kick_buf)
        self.kick_attempt_candidate_buf = torch.zeros_like(self.valid_kick_buf)
        self.failed_kick_attempt_buf = torch.zeros_like(self.valid_kick_buf)
        self.kick_attempt_deadline_step = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.kicking_foot_mask_buf = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            device=self.device,
            dtype=torch.bool,
        )
        self.kicking_foot_airborne_buf = torch.zeros_like(
            self.kicking_foot_mask_buf
        )
        self.previous_feet_contact_buf = torch.zeros_like(
            self.kicking_foot_mask_buf
        )
        self.post_kick_phase_target_buf = torch.zeros_like(
            self.valid_kick_buf
        )
        self._direct_kick_detection_processed = False

    def _init_direct_perception_buffers(self):
        filter_cfg = self.direct_cfg["filter"]
        perception_cfg = self.direct_cfg["perception_randomization"]
        observation_cfg = self.direct_cfg["observation"]

        self.prediction_horizons = tuple(
            float(value) for value in observation_cfg["prediction_horizons_s"]
        )
        if not self.prediction_horizons or self.prediction_horizons[0] != 0.0:
            raise ValueError("prediction_horizons_s must start with 0.0")
        if any(
            next_value <= value
            for value, next_value in zip(
                self.prediction_horizons,
                self.prediction_horizons[1:],
            )
        ):
            raise ValueError("prediction horizons must be strictly increasing")
        self.std_reference = float(observation_cfg["std_reference_m"])
        self.log_std_clip = tuple(
            float(value) for value in observation_cfg["log_std_clip"]
        )
        self.measurement_age_normalizer = float(
            observation_cfg["measurement_age_normalizer_s"]
        )
        if self.std_reference <= 0.0 or self.measurement_age_normalizer <= 0.0:
            raise ValueError(
                "std_reference_m and measurement_age_normalizer_s must be positive"
            )
        if len(self.log_std_clip) != 2 or self.log_std_clip[0] > self.log_std_clip[1]:
            raise ValueError("log_std_clip must contain two ordered values")

        expected_observations = expected_direct_kicking_observation_size(
            len(self.prediction_horizons)
        )
        if self.num_obs != expected_observations:
            raise ValueError(
                "DirectKicking num_observations must be {}, got {}".format(
                    expected_observations,
                    self.num_obs,
                )
            )

        self.measurement_noise_std_range = self._validated_range(
            perception_cfg["measurement_noise_std_range_m"],
            "measurement_noise_std_range_m",
            positive=True,
        )
        self.measurement_noise_reference_distance = float(
            perception_cfg.get("measurement_noise_reference_distance_m", 1.0)
        )
        self.measurement_noise_max_scale = float(
            perception_cfg.get("measurement_noise_max_scale", 8.0)
        )
        if self.measurement_noise_reference_distance <= 0.0:
            raise ValueError(
                "measurement_noise_reference_distance_m must be positive"
            )
        if self.measurement_noise_max_scale < 1.0:
            raise ValueError("measurement_noise_max_scale must be at least one")
        self.camera_fps_range = self._validated_range(
            perception_cfg["camera_fps_range"],
            "camera_fps_range",
            positive=True,
        )
        self.latency_range = self._validated_range(
            perception_cfg["latency_range_s"],
            "latency_range_s",
            non_negative=True,
        )
        self.q_scale_range = self._validated_range(
            filter_cfg["process_noise_scale_range"],
            "process_noise_scale_range",
            positive=True,
        )
        self.r_scale_range = self._validated_range(
            filter_cfg["measurement_noise_scale_range"],
            "measurement_noise_scale_range",
            positive=True,
        )

        self.camera_fps_jitter = float(perception_cfg.get("camera_fps_jitter", 0.0))
        if not 0.0 <= self.camera_fps_jitter < 1.0:
            raise ValueError("camera_fps_jitter must be in [0, 1)")
        self.dropout_burst_probability = float(
            perception_cfg.get("dropout_burst_probability", 0.0)
        )
        if not 0.0 <= self.dropout_burst_probability <= 1.0:
            raise ValueError("dropout_burst_probability must be in [0, 1]")
        burst_range = perception_cfg.get("dropout_burst_frames", [1, 1])
        self.dropout_burst_min = int(burst_range[0])
        self.dropout_burst_max = int(burst_range[1])
        if self.dropout_burst_min < 1 or self.dropout_burst_max < self.dropout_burst_min:
            raise ValueError("dropout_burst_frames must be positive and ordered")
        self.outlier_probability = float(perception_cfg.get("outlier_probability", 0.0))
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must be in [0, 1]")
        self.outlier_distance_range = self._validated_range(
            perception_cfg.get("outlier_distance_range_m", [0.0, 0.0]),
            "outlier_distance_range_m",
            non_negative=True,
        )

        self.base_process_acceleration_std = float(
            filter_cfg["process_acceleration_std_mps2"]
        )
        self.initial_velocity_std = float(filter_cfg["initial_velocity_std_mps"])
        self.nis_threshold = float(filter_cfg["nis_threshold"])
        self.max_missing_time = float(filter_cfg["max_missing_time_s"])
        if self.base_process_acceleration_std < 0.0:
            raise ValueError("process_acceleration_std_mps2 must be non-negative")
        if self.initial_velocity_std <= 0.0 or self.max_missing_time <= 0.0:
            raise ValueError("initial_velocity_std_mps and max_missing_time_s must be positive")

        self.ball_filter = VisionFilter(self.num_envs, self.device)
        self.measurement_noise_std = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        ego_noise_cfg = self.direct_cfg.get("ego_motion_noise", {})
        self.ego_velocity_bias_std = float(
            ego_noise_cfg.get("velocity_bias_std_mps", 0.03)
        )
        self.ego_velocity_drift_std = float(
            ego_noise_cfg.get("velocity_drift_std_mps_per_sqrt_s", 0.01)
        )
        self.ego_yaw_rate_bias_std = float(
            ego_noise_cfg.get("yaw_rate_bias_std_rps", 0.05)
        )
        self.ego_yaw_rate_drift_std = float(
            ego_noise_cfg.get("yaw_rate_drift_std_rps_per_sqrt_s", 0.02)
        )
        self.ego_position_noise_std = float(
            ego_noise_cfg.get("position_noise_std_m", 0.005)
        )
        self.ego_position_bias_std = float(
            ego_noise_cfg.get("position_bias_std_m", 0.01)
        )
        self.ego_yaw_noise_std = float(
            ego_noise_cfg.get("yaw_noise_std_rad", 0.005)
        )
        self.ego_yaw_bias_std = float(
            ego_noise_cfg.get("yaw_bias_std_rad", 0.01)
        )
        noise_cfg = self.task_cfg.get("noise", {})
        self.ego_velocity_noise_std = float(
            noise_cfg.get("lin_vel", {}).get("range", [0.0, 0.0])[1]
        )
        self.ego_yaw_rate_noise_std = float(
            noise_cfg.get("ang_vel", {}).get("range", [0.0, 0.0])[1]
        )
        for value, name in (
            (self.ego_velocity_bias_std, "velocity_bias_std_mps"),
            (
                self.ego_velocity_drift_std,
                "velocity_drift_std_mps_per_sqrt_s",
            ),
            (self.ego_yaw_rate_bias_std, "yaw_rate_bias_std_rps"),
            (
                self.ego_yaw_rate_drift_std,
                "yaw_rate_drift_std_rps_per_sqrt_s",
            ),
            (self.ego_position_noise_std, "position_noise_std_m"),
            (self.ego_position_bias_std, "position_bias_std_m"),
            (self.ego_yaw_noise_std, "yaw_noise_std_rad"),
            (self.ego_yaw_bias_std, "yaw_bias_std_rad"),
            (self.ego_velocity_noise_std, "lin_vel_noise_std_mps"),
            (self.ego_yaw_rate_noise_std, "ang_vel_noise_std_rps"),
        ):
            if value < 0.0:
                raise ValueError("{} must be non-negative".format(name))
        self.ego_velocity_bias = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.ego_velocity_drift = torch.zeros_like(self.ego_velocity_bias)
        self.ego_yaw_rate_bias = torch.zeros_like(self.measurement_noise_std)
        self.ego_yaw_rate_drift = torch.zeros_like(self.measurement_noise_std)
        self.ego_position_bias = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.ego_yaw_bias = torch.zeros_like(self.measurement_noise_std)
        self.observed_base_position = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.observed_base_yaw = torch.zeros_like(self.measurement_noise_std)
        self.filter_measurement_std = torch.zeros_like(self.measurement_noise_std)
        self.process_acceleration_std = torch.zeros_like(self.measurement_noise_std)
        self.camera_fps = torch.zeros_like(self.measurement_noise_std)
        self.camera_timer = torch.zeros_like(self.measurement_noise_std)
        self.last_measurement_age = torch.zeros_like(self.measurement_noise_std)
        self.measurement_updated = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self.belief_valid = torch.zeros_like(self.measurement_updated)
        self.perception_just_reset = torch.zeros_like(self.measurement_updated)
        self.dropout_remaining = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.perception_latency_steps = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.perception_latency = torch.zeros_like(self.measurement_noise_std)
        self.perception_history_valid_steps = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.kick_detection_block_until_step = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.direct_previous_ball_pos_world = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        self.direct_previous_ball_lin_vel_world = torch.zeros_like(
            self.direct_previous_ball_pos_world
        )
        self.direct_previous_base_pos_world = torch.zeros_like(
            self.direct_previous_ball_pos_world
        )
        self.direct_previous_base_quat_world = torch.zeros(
            self.num_envs,
            4,
            device=self.device,
        )

        max_latency_steps = int(math.ceil(self.latency_range[1] / self.dt))
        self.perception_history_length = max_latency_steps + 2
        self.ball_position_history = torch.zeros(
            self.num_envs,
            self.perception_history_length,
            2,
            device=self.device,
        )
        self.base_position_history = torch.zeros_like(self.ball_position_history)
        self.base_yaw_history = torch.zeros(
            self.num_envs,
            self.perception_history_length,
            device=self.device,
        )
        self.perception_history_cursor = 0

    def _reset_ball_at_robot_front(self, env_ids):
        """Spawn balls inside the camera FOV with a robot-relative trajectory."""
        if len(env_ids) == 0:
            return

        self._randomize_ball_restitution(env_ids)
        self.valid_kick_buf[env_ids] = False
        if hasattr(self, "first_valid_kick_step"):
            self.first_valid_kick_step[env_ids] = -1
            self.post_kick_terminal_buf[env_ids] = False
            self.kick_attempt_pending_buf[env_ids] = False
            self.kick_attempt_candidate_buf[env_ids] = False
            self.failed_kick_attempt_buf[env_ids] = False
            self.kick_attempt_deadline_step[env_ids] = 0
            self.kicking_foot_mask_buf[env_ids] = False
            self.kicking_foot_airborne_buf[env_ids] = False
            self.previous_feet_contact_buf[env_ids] = False
            self.post_kick_phase_target_buf[env_ids] = False
        count = len(env_ids)
        incoming = (
            torch.rand(count, device=self.device) < self.incoming_probability
        )
        stationary = torch.rand(count, device=self.device) < self.stationary_probability
        base_speed = self._sample_ball_speed(count)
        base_speed = torch.where(stationary, 0.0, base_speed)
        spawn_bearing = self._sample_uniform(self.spawn_bearing_range, count)
        closest_approach_offset = self._sample_uniform(
            self.closest_approach_offset_range,
            count,
        )
        time_to_closest = self._sample_uniform(self.incoming_time_to_closest_range, count)
        # For a stationary robot and a constant-speed ball, the path to closest
        # approach has length v*T. Low speeds keep the approved distance floor.
        incoming_distance = torch.hypot(
            base_speed * time_to_closest, closest_approach_offset,
        ).clamp_min(self.minimum_spawn_distance)
        outgoing_distance = self._sample_uniform(
            self.outgoing_spawn_distance_range,
            count,
        )
        spawn_distance = torch.where(
            incoming,
            incoming_distance,
            outgoing_distance,
        )
        stationary_distance = self._sample_uniform(self.stationary_spawn_distance_range, count)
        spawn_distance = torch.where(stationary, stationary_distance, spawn_distance)
        local_spawn_xy, local_velocity_xy = build_ball_trajectory(
            spawn_distance,
            spawn_bearing,
            closest_approach_offset,
            base_speed,
            incoming,
        )

        robot_pos = self.root_states[env_ids, 0, 0:3]
        _, _, robot_yaw = get_euler_xyz(self.root_states[env_ids, 0, 3:7])
        local_to_world = self._world_to_local_rotation(robot_yaw).transpose(-1, -2)
        world_spawn_offset = torch.matmul(
            local_to_world,
            local_spawn_xy.unsqueeze(-1),
        ).squeeze(-1)
        world_velocity_xy = torch.matmul(
            local_to_world,
            local_velocity_xy.unsqueeze(-1),
        ).squeeze(-1)
        ball_target_xy = robot_pos[:, :2] + world_spawn_offset
        ball_target_z = (
            self._terrain_heights(ball_target_xy) + self.ball_radius
        )

        self.root_states[env_ids, 1, 0:2] = ball_target_xy
        self.root_states[env_ids, 1, 2] = ball_target_z
        self.root_states[env_ids, 1, 3:7] = 0.0
        self.root_states[env_ids, 1, 6] = 1.0
        self.root_states[env_ids, 1, 7:13] = 0.0
        self.root_states[env_ids, 1, 7:9] = world_velocity_xy
        self.root_states[env_ids, 1, 10] = -world_velocity_xy[:, 1] / self.ball_radius
        self.root_states[env_ids, 1, 11] = world_velocity_xy[:, 0] / self.ball_radius

        self._write_root_states(env_ids, ball=True)
        if hasattr(self, "ball_filter"):
            self.kick_detection_block_until_step[env_ids] = (
                self.episode_length_buf[env_ids] + self._kick_warmup_steps()
            )
            self._reset_direct_perception(env_ids)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, "action_target_history"):
            self._reset_action_delay(env_ids)
        if len(env_ids) > 0 and hasattr(self, "next_force_push_step"):
            self._reset_external_disturbance_schedule(env_ids)
        if len(env_ids) > 0 and hasattr(self, "kick_detection_block_until_step"):
            self.kick_detection_block_until_step[env_ids] = (
                self.episode_length_buf[env_ids] + self._kick_warmup_steps()
            )
        if len(env_ids) > 0 and hasattr(self, "post_kick_phase_target_buf"):
            self.kicking_foot_mask_buf[env_ids] = False
            self.kicking_foot_airborne_buf[env_ids] = False
            self.previous_feet_contact_buf[env_ids] = False
            self.post_kick_phase_target_buf[env_ids] = False
        if len(env_ids) > 0 and hasattr(self, "edge_only_support_steps_buf"):
            self.edge_only_support_steps_buf[env_ids] = 0

    def _reset_direct_perception(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return

        self.ego_velocity_bias[env_ids] = (
            torch.randn(count, 2, device=self.device)
            * self.ego_velocity_bias_std
        )
        self.ego_velocity_drift[env_ids] = 0.0
        self.ego_yaw_rate_bias[env_ids] = (
            torch.randn(count, device=self.device)
            * self.ego_yaw_rate_bias_std
        )
        self.ego_yaw_rate_drift[env_ids] = 0.0
        self.ego_position_bias[env_ids] = (
            torch.randn(count, 2, device=self.device)
            * self.ego_position_bias_std
        )
        self.ego_yaw_bias[env_ids] = (
            torch.randn(count, device=self.device) * self.ego_yaw_bias_std
        )

        self.measurement_noise_std[env_ids] = self._sample_uniform(
            self.measurement_noise_std_range,
            count,
        )
        self.filter_measurement_std[env_ids] = 0.0  # R comes from VisionFilter calibration.
        self.process_acceleration_std[env_ids] = 0.8
        self.camera_fps[env_ids] = self._sample_uniform(self.camera_fps_range, count)
        sampled_latency = self._sample_uniform(self.latency_range, count)
        latency_steps = torch.round(sampled_latency / self.dt).to(torch.long)
        latency_steps.clamp_(max=self.perception_history_length - 2)
        self.perception_latency_steps[env_ids] = latency_steps
        self.perception_latency[env_ids] = latency_steps.to(torch.float) * self.dt
        self.camera_timer[env_ids] = self.perception_latency[env_ids] + (
            torch.rand(count, device=self.device) / self.camera_fps[env_ids]
        )
        self.last_measurement_age[env_ids] = self.max_missing_time
        self.measurement_updated[env_ids] = False
        self.belief_valid[env_ids] = False
        self.perception_just_reset[env_ids] = True
        self.dropout_remaining[env_ids] = 0
        self.perception_history_valid_steps[env_ids] = 0

        ball_xy = self.root_states[env_ids, 1, :2]
        base_xy = self.root_states[env_ids, 0, :2]
        _, _, base_yaw = get_euler_xyz(self.root_states[env_ids, 0, 3:7])
        self.observed_base_position[env_ids] = (
            base_xy
            + self.ego_position_bias[env_ids]
            + torch.randn(count, 2, device=self.device)
            * self.ego_position_noise_std
        )
        self.observed_base_yaw[env_ids] = (
            base_yaw
            + self.ego_yaw_bias[env_ids]
            + torch.randn(count, device=self.device) * self.ego_yaw_noise_std
        )
        self.ball_position_history[env_ids] = ball_xy.unsqueeze(1)
        self.base_position_history[env_ids] = self.observed_base_position[env_ids].unsqueeze(1)
        self.base_yaw_history[env_ids] = self.observed_base_yaw[env_ids].unsqueeze(1)

        self.ball_filter.invalidate(env_ids)

    def _advance_direct_perception(self):
        just_reset = self.perception_just_reset.clone()
        perception_active = ~just_reset
        filter_active = self.ball_filter.initialized & perception_active

        self.perception_history_cursor = (
            self.perception_history_cursor + 1
        ) % self.perception_history_length
        cursor = self.perception_history_cursor
        self.ball_position_history[:, cursor] = self.root_states[:, 1, :2]
        self.base_position_history[:, cursor] = self.observed_base_position
        self.base_yaw_history[:, cursor] = self.observed_base_yaw
        self.perception_history_valid_steps[perception_active] += 1

        # Each environment keeps its CVKF on a fixed-lag timeline.  The lag is
        # sampled once per episode, so standard (non-out-of-sequence) KF
        # corrections remain covariance-consistent.
        self.last_measurement_age[perception_active] += self.dt
        self.camera_timer[perception_active] -= self.dt
        history_ready = (
            self.perception_history_valid_steps >= self.perception_latency_steps
        )
        due = perception_active & history_ready & (self.camera_timer <= 0.0)
        due_ids = due.nonzero(as_tuple=False).flatten()
        self.measurement_updated.zero_()

        if due_ids.numel() > 0:
            jitter = 1.0 + self.camera_fps_jitter * (
                2.0 * torch.rand(due_ids.numel(), device=self.device) - 1.0
            )
            next_period = jitter / self.camera_fps[due_ids]
            self.camera_timer[due_ids] += next_period
            self.camera_timer[due_ids] = torch.clamp(
                self.camera_timer[due_ids],
                min=0.25 * self.dt,
            )

            dropout = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            continuing_dropout = self.dropout_remaining[due_ids] > 0
            continuing_ids = due_ids[continuing_dropout]
            dropout[continuing_ids] = True
            self.dropout_remaining[continuing_ids] -= 1

            eligible_ids = due_ids[~continuing_dropout]
            if eligible_ids.numel() > 0:
                starts_dropout = (
                    torch.rand(eligible_ids.numel(), device=self.device)
                    < self.dropout_burst_probability
                )
                start_ids = eligible_ids[starts_dropout]
                if start_ids.numel() > 0:
                    burst_lengths = torch.randint(
                        self.dropout_burst_min,
                        self.dropout_burst_max + 1,
                        (start_ids.numel(),),
                        device=self.device,
                    )
                    dropout[start_ids] = True
                    self.dropout_remaining[start_ids] = burst_lengths - 1

            latency_steps = self.perception_latency_steps[due_ids]
            measurement_age = self.perception_latency[due_ids]
            history_index = (cursor - latency_steps) % self.perception_history_length
            captured_ball = self.ball_position_history[due_ids, history_index]
            captured_base = self.base_position_history[due_ids, history_index]
            captured_yaw = self.base_yaw_history[due_ids, history_index]
            captured_local = self._world_point_to_local(
                captured_ball,
                captured_base,
                captured_yaw,
            )
            captured_distance = torch.norm(captured_local, dim=-1)

            vision_cfg = self.task_cfg.get("vision", {})
            if vision_cfg.get("enabled", True):
                visible_local = (
                    horizontal_fov_mask(
                        captured_local,
                        vision_cfg.get("fov_yaw", 2.0 * math.pi),
                    )
                    & (captured_distance > vision_cfg.get("min_distance", 0.0))
                    & (captured_distance < vision_cfg.get("max_distance", float("inf")))
                )
            else:
                visible_local = torch.ones(
                    due_ids.numel(),
                    device=self.device,
                    dtype=torch.bool,
                )

            available_local = visible_local & ~dropout[due_ids]
            available_ids = due_ids[available_local]
            measurements = torch.zeros(self.num_envs, 2, device=self.device)
            update_mask = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            ages = torch.zeros(self.num_envs, device=self.device)
            measured_local_all = torch.zeros_like(measurements)
            if available_ids.numel() > 0:
                measured_local = captured_local[available_local].clone()
                sensor_measurement_std = distance_scaled_measurement_std(
                    self.measurement_noise_std[available_ids],
                    captured_distance[available_local],
                    self.measurement_noise_reference_distance,
                    self.measurement_noise_max_scale,
                )
                measured_local += (
                    torch.randn_like(measured_local)
                    * sensor_measurement_std.unsqueeze(-1)
                )
                outlier_candidate = (
                    torch.rand(available_ids.numel(), device=self.device)
                    < self.outlier_probability
                )
                initialized_track = self.ball_filter.initialized[available_ids]
                # Without a prior belief there is no meaningful NIS gate.  A
                # false first detection is therefore treated as a missed frame
                # rather than silently converted into a clean measurement.
                rejected_initial_outlier = outlier_candidate & ~initialized_track
                outlier = outlier_candidate & initialized_track
                outlier_distance = self._sample_uniform(
                    self.outlier_distance_range,
                    available_ids.numel(),
                )
                outlier_angle = 2.0 * math.pi * torch.rand(
                    available_ids.numel(),
                    device=self.device,
                )
                outlier_scale = outlier.float() * outlier_distance
                measured_local[:, 0] += outlier_scale * torch.cos(outlier_angle)
                measured_local[:, 1] += outlier_scale * torch.sin(outlier_angle)
                measurements[available_ids] = self._local_point_to_world(
                    measured_local,
                    captured_base[available_local],
                    captured_yaw[available_local],
                )
                measured_local_all[available_ids] = measured_local
                usable_ids = available_ids[~rejected_initial_outlier]
                update_mask[usable_ids] = True
                ages[usable_ids] = measurement_age[available_local][
                    ~rejected_initial_outlier
                ]

            capture_ns = (
                self.common_step_counter * int(round(self.dt * 1e9))
                - self.perception_latency_steps[due_ids] * int(round(self.dt * 1e9))
            )
            self.ball_filter.process(due_ids, capture_ns, measurements[due_ids],
                                     measured_local_all[due_ids], update_mask[due_ids])
            self.measurement_updated[:] = self.ball_filter.updated
            accepted = self.measurement_updated
            self.last_measurement_age[accepted] = self.perception_latency[accepted]

        self.belief_valid[:] = self.ball_filter.status != 0
        self.perception_just_reset.zero_()

    def _update_ego_motion_sensor_noise(self):
        """Advance slowly varying odometry and IMU errors by one control step."""
        drift_scale = math.sqrt(self.dt)
        self.ego_velocity_drift += (
            torch.randn_like(self.ego_velocity_drift)
            * self.ego_velocity_drift_std
            * drift_scale
        )
        self.ego_yaw_rate_drift += (
            torch.randn_like(self.ego_yaw_rate_drift)
            * self.ego_yaw_rate_drift_std
            * drift_scale
        )

    def _update_observed_base_pose(self):
        """Create the noisy base pose used by the perception pipeline."""
        true_base_position = self.root_states[:, 0, :2]
        _, _, true_base_yaw = get_euler_xyz(self.root_states[:, 0, 3:7])
        self.observed_base_position[:] = (
            true_base_position
            + self.ego_position_bias
            + torch.randn_like(true_base_position) * self.ego_position_noise_std
        )
        self.observed_base_yaw[:] = (
            true_base_yaw
            + self.ego_yaw_bias
            + torch.randn_like(true_base_yaw) * self.ego_yaw_noise_std
        )

    def _belief_observation(self, observed_base_lin_vel_yaw, observed_base_ang_vel):
        horizons = torch.as_tensor(
            self.prediction_horizons,
            device=self.device,
            dtype=self.ball_filter.state.dtype,
        )
        snapshot_age = (self.common_step_counter * int(round(self.dt * 1e9))
                        - self.ball_filter.stamp_ns).to(horizons.dtype) * 1e-9
        forecast_offsets = snapshot_age.unsqueeze(1) + horizons.unsqueeze(0)
        means_world, covariance_world = self.ball_filter.forecast_offsets(
            forecast_offsets,
            self.process_acceleration_std,
        )
        # The CVKF belief is on the delayed measurement timeline, so the ball
        # must be forecast by snapshot age + horizon.  The current ego pose is
        # already at the present time, so its motion is forecast by horizon
        # only; including latency here would double-count robot motion.
        ego_forecast_offsets = horizons.unsqueeze(0).expand(self.num_envs, -1)
        base_position = self.observed_base_position
        base_yaw = self.observed_base_yaw
        base_position_forecast, base_yaw_forecast = forecast_constant_body_twist(
            base_position,
            base_yaw,
            observed_base_lin_vel_yaw[:, :2],
            observed_base_ang_vel[:, 2],
            ego_forecast_offsets,
        )
        ego_covariance = forecast_ego_state_covariance(
            ego_forecast_offsets,
            self.ego_position_noise_std,
            self.ego_position_bias_std,
            self.ego_velocity_noise_std,
            self.ego_velocity_bias_std,
            self.ego_velocity_drift_std,
            self.ego_yaw_noise_std,
            self.ego_yaw_bias_std,
            self.ego_yaw_rate_noise_std,
            self.ego_yaw_rate_bias_std,
            self.ego_yaw_rate_drift_std,
        )
        relative_state, local_covariance = relative_state_and_covariance(
            means_world, covariance_world, base_position_forecast, base_yaw_forecast,
            observed_base_lin_vel_yaw[:, None, :2].expand(-1, len(horizons), -1),
            observed_base_ang_vel[:, None, 2].expand(-1, len(horizons)),
            ego_covariance,
        )
        position_scale = float(self.task_cfg["normalization"]["ball_pos"])
        velocity_scale = float(self.task_cfg["normalization"]["ball_vel"])
        scale = relative_state.new_tensor((position_scale, position_scale, velocity_scale, velocity_scale))
        relative_state = relative_state * scale
        # Scale the full covariance consistently with the state: D P D^T.
        # No log transform or clipping: all position/velocity cross terms remain.
        local_covariance = local_covariance * scale[:, None] * scale[None, :]
        # Invalid tokens are placeholders, not a probability distribution.
        # Retain the old position-uncertainty sentinel in covariance units;
        # new velocity/cross entries are zero, as is the invalid public filter P.
        invalid_position_variance = (self.std_reference * math.exp(self.log_std_clip[1]))**2
        invalid_covariance = torch.diag(relative_state.new_tensor(
            (invalid_position_variance, invalid_position_variance, 0., 0.)
        ) * scale.square())
        horizon_features = build_horizon_tokens(
            relative_state, local_covariance, self.prediction_horizons,
            self.belief_valid, invalid_covariance,
        ).reshape(self.num_envs, -1)
        valid = self.belief_valid

        normalized_age = torch.clamp(
            self.last_measurement_age / self.measurement_age_normalizer,
            min=0.0,
            max=1.0,
        )
        status = torch.stack(
            (
                normalized_age,
                self.measurement_updated.float(),
                valid.float(),
            ),
            dim=-1,
        )
        return torch.cat((horizon_features, status), dim=-1)

    def _compute_observations(self):
        if self._perception_step_requested:
            self._update_ego_motion_sensor_noise()
            self._update_observed_base_pose()
            self._advance_direct_perception()
        else:
            self.perception_just_reset.zero_()

        commands_scale = torch.tensor(
            [
                self.task_cfg["normalization"]["lin_vel"],
                self.task_cfg["normalization"]["lin_vel"],
                self.task_cfg["normalization"]["ang_vel"],
            ],
            device=self.device,
        )
        target_direction_world = self._unit_kick_target_direction_xy()
        base_yaw = self.observed_base_yaw
        target_observation = torch.matmul(
            self._world_to_local_rotation(base_yaw),
            target_direction_world.unsqueeze(-1),
        ).squeeze(-1)
        observed_base_ang_vel = apply_randomization(
            self.base_ang_vel,
            self.task_cfg["noise"].get("ang_vel"),
        )
        observed_base_ang_vel[:, 2] += (
            self.ego_yaw_rate_bias + self.ego_yaw_rate_drift
        )
        base_velocity_yaw_xy = torch.matmul(
            self._world_to_local_rotation(base_yaw),
            self.root_states[:, 0, 7:9].unsqueeze(-1),
        ).squeeze(-1)
        base_velocity_yaw = torch.cat(
            (base_velocity_yaw_xy, self.root_states[:, 0, 9:10]),
            dim=-1,
        )
        observed_base_lin_vel_yaw = apply_randomization(
            base_velocity_yaw,
            self.task_cfg["noise"].get("lin_vel"),
        )
        observed_base_lin_vel_yaw[:, :2] += (
            self.ego_velocity_bias + self.ego_velocity_drift
        )
        belief_observation = self._belief_observation(
            observed_base_lin_vel_yaw,
            observed_base_ang_vel,
        )

        self.obs_buf = torch.cat(
            (
                apply_randomization(
                    self.projected_gravity,
                    self.task_cfg["noise"].get("gravity"),
                )
                * self.task_cfg["normalization"]["gravity"],
                observed_base_ang_vel * self.task_cfg["normalization"]["ang_vel"],
                self.commands[:, :3] * commands_scale,
                (
                    torch.cos(2.0 * math.pi * self.gait_process)
                    * (self.gait_frequency > 1.0e-8).float()
                ).unsqueeze(-1),
                (
                    torch.sin(2.0 * math.pi * self.gait_process)
                    * (self.gait_frequency > 1.0e-8).float()
                ).unsqueeze(-1),
                apply_randomization(
                    self.dof_pos - self.default_dof_pos,
                    self.task_cfg["noise"].get("dof_pos"),
                )
                * self.task_cfg["normalization"]["dof_pos"],
                apply_randomization(
                    self.dof_vel,
                    self.task_cfg["noise"].get("dof_vel"),
                )
                * self.task_cfg["normalization"]["dof_vel"],
                self.actions,
                belief_observation,
                target_observation,
            ),
            dim=-1,
        )
        if self.obs_buf.shape[1] != self.num_obs:
            raise RuntimeError(
                "DirectKicking produced {} observations, expected {}".format(
                    self.obs_buf.shape[1],
                    self.num_obs,
                )
            )

        privileged_ball_velocity = self.ball_lin_vel[:, 0:2]
        privileged_feet_position = torch.cat(
            (self.feet_pos[:, 0, 0:2], self.feet_pos[:, 1, 0:2]),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(
                    self.base_lin_vel,
                    self.task_cfg["noise"].get("lin_vel"),
                )
                * self.task_cfg["normalization"]["lin_vel"],
                apply_randomization(
                    self.base_pos[:, 2]
                    - self._terrain_heights(self.base_pos),
                    self.task_cfg["noise"].get("height"),
                ).unsqueeze(-1),
                self.pushing_forces[:, 0, :]
                * self.task_cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :]
                * self.task_cfg["normalization"]["push_torque"],
                privileged_ball_velocity,
                privileged_feet_position,
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf
        self.extras["post_kick_phase_target"] = (
            self.post_kick_phase_target_buf.float().clone()
        )

    def _update_valid_kick(
        self,
        previous_ball_pos_world,
        previous_ball_lin_vel_world,
    ):
        if self._direct_kick_detection_processed:
            return torch.zeros_like(self.valid_kick_buf)
        self._direct_kick_detection_processed = True
        if not self._reward_value("use_valid_kick_gating", False):
            return torch.zeros_like(self.valid_kick_buf)

        per_foot_candidate = per_foot_kick_candidates(
            self.feet_pos,
            self.last_feet_pos,
            self.ball_pos,
            previous_ball_pos_world,
            self.dt,
            float(
                self._reward_value(
                    "kick_detection_foot_distance_threshold",
                    0.23,
                )
            ),
            float(
                self._reward_value(
                    "kick_detection_min_foot_speed_towards_ball",
                    0.2,
                )
            ),
        )
        foot_candidate = torch.any(per_foot_candidate, dim=-1)
        min_velocity_change = float(
            self._reward_value("kick_detection_velocity_change_threshold", 0.5)
        )
        valid_kick = physical_kick_event(
            foot_candidate,
            self.root_states[:, 1, 7:9],
            previous_ball_lin_vel_world[:, :2],
            min_velocity_change,
        )
        was_valid_kick = self.valid_kick_buf.clone()
        self.valid_kick_buf |= valid_kick

        warmup_finished = (
            self.episode_length_buf >= self.kick_detection_block_until_step
        )
        self.valid_kick_buf &= warmup_finished
        new_valid_kick = self.valid_kick_buf & ~was_valid_kick
        kicking_foot_mask = select_kicking_foot_mask(
            per_foot_candidate,
            self.feet_pos,
            self.last_feet_pos,
            self.ball_pos,
            previous_ball_pos_world,
        )
        self.kicking_foot_mask_buf[new_valid_kick] = kicking_foot_mask[
            new_valid_kick
        ]
        self.first_valid_kick_step[:] = latch_first_kick_step(
            self.first_valid_kick_step,
            self.episode_length_buf,
            new_valid_kick,
        )
        return new_valid_kick

    def _update_post_kick_phase_target(self):
        (
            self.kicking_foot_airborne_buf[:],
            self.post_kick_phase_target_buf[:],
        ) = update_post_kick_phase_target(
            self.kicking_foot_mask_buf,
            self.previous_feet_contact_buf,
            self.feet_contact,
            self.kicking_foot_airborne_buf,
            self.post_kick_phase_target_buf,
        )

    def _compute_reward(self):
        # The parent updates valid_kick after rewards. DirectKicking detects the
        # physical event first so the first-kick timer and one-shot direction
        # reward are recorded on the same control step.
        new_valid_kick = self._update_valid_kick(
            self.direct_previous_ball_pos_world,
            self.direct_previous_ball_lin_vel_world,
        )
        self._update_post_kick_phase_target()
        self._update_kick_attempt_outcome(new_valid_kick)
        kick_direction_reward = one_shot_direction_reward(
            self.root_states[:, 1, 7:9],
            self._unit_kick_target_direction_xy(),
            new_valid_kick,
            self.max_kick_direction_reward,
            sharpness=self.kick_direction_reward_sharpness,
        )
        velocity_change_scale = sigmoid_velocity_change_scale(
            self.root_states[:, 1, 7:9]
            - self.direct_previous_ball_lin_vel_world[:, :2],
            center_mps=self.kick_velocity_change_reward_center,
            sharpness=self.kick_velocity_change_reward_sharpness,
        )
        kick_direction_reward *= velocity_change_scale
        kick_direction_reward *= (~self.reset_buf).float()
        super()._compute_reward()
        self.rew_buf += kick_direction_reward
        self.extras["rew_terms"]["kick_direction"] = kick_direction_reward

    def _reward_ankle_torques(self):
        """Penalize commanded torques for the four ankle joints."""
        ankle_torques = self.torques[:, self.ankle_dof_indices]
        return torch.sum(torch.square(ankle_torques), dim=-1)

    def _reward_action_second_difference(self):
        """Penalize the squared second difference of consecutive actions."""
        penalty = action_second_difference_l2(
            self.actions,
            self.last_actions,
            self.previous_previous_actions,
        )
        self.previous_previous_actions.copy_(self.last_actions)
        return penalty

    def _reward_parallel_step(self):
        """Penalize a forward swing foot moving laterally in the base frame."""
        foot_velocity_world = (
            (self.feet_pos - self.last_feet_pos)[:, :, :2] / self.dt
        )
        foot_offset_world = (
            self.feet_pos[:, :, :2] - self.base_pos[:, None, :2]
        )
        base_yaw_rate = self.root_states[:, 0, 12].unsqueeze(-1)
        base_rotation_velocity = base_yaw_rate.unsqueeze(-1) * torch.stack(
            (-foot_offset_world[:, :, 1], foot_offset_world[:, :, 0]),
            dim=-1,
        )
        foot_velocity_relative = (
            foot_velocity_world
            - self.root_states[:, 0, 7:9].unsqueeze(1)
            - base_rotation_velocity
        )
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        world_to_base = self._world_to_local_rotation(base_yaw).unsqueeze(1)
        foot_velocity_local = torch.matmul(
            world_to_base,
            foot_velocity_relative.unsqueeze(-1),
        ).squeeze(-1)
        penalty = nonparallel_step_penalty(
            foot_velocity_local,
            self.feet_contact,
            lateral_velocity_tolerance=float(
                self._reward_value(
                    "parallel_step_lateral_velocity_tolerance_mps",
                    0.05,
                )
            ),
            lateral_velocity_scale=float(
                self._reward_value(
                    "parallel_step_lateral_velocity_scale_mps",
                    0.25,
                )
            ),
        )
        active = (self.episode_length_buf > 1) & ~self.valid_kick_buf
        return penalty * active.float()

    def _large_stationary_kick_attempt_candidate(self):
        foot_position_world = (
            self.feet_pos[:, :, :2] - self.base_pos[:, None, :2]
        )
        foot_velocity_world = (
            (self.feet_pos - self.last_feet_pos)[:, :, :2] / self.dt
        )
        foot_velocity_relative = (
            foot_velocity_world
            - self.root_states[:, 0, 7:9].unsqueeze(1)
        )
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        world_to_base = self._world_to_local_rotation(base_yaw).unsqueeze(1)
        foot_position_local = torch.matmul(
            world_to_base,
            foot_position_world.unsqueeze(-1),
        ).squeeze(-1)
        foot_velocity_local = torch.matmul(
            world_to_base,
            foot_velocity_relative.unsqueeze(-1),
        ).squeeze(-1)
        base_planar_speed = torch.norm(
            self.root_states[:, 0, 7:9],
            dim=-1,
        )
        candidate = large_stationary_kick_attempt_candidate(
            foot_position_local[:, :, 0],
            foot_velocity_local[:, :, 0],
            self.feet_contact,
            base_planar_speed,
            self.kick_attempt_minimum_forward_position,
            self.kick_attempt_minimum_forward_speed,
            self.kick_attempt_maximum_base_speed,
        )
        return candidate & (self.episode_length_buf > 1)

    def _update_kick_attempt_outcome(self, new_valid_kick):
        """Latch large swings and emit one failure after the outcome window."""
        candidate = (
            self._large_stationary_kick_attempt_candidate()
            & ~self.valid_kick_buf
        )
        pending, previous_candidate, deadline, failed = (
            update_kick_attempt_outcome(
                candidate,
                new_valid_kick,
                self.kick_attempt_pending_buf,
                self.kick_attempt_candidate_buf,
                self.kick_attempt_deadline_step,
                self.episode_length_buf,
                self.kick_attempt_outcome_window_steps,
            )
        )
        self.kick_attempt_pending_buf[:] = pending
        self.kick_attempt_candidate_buf[:] = previous_candidate
        self.kick_attempt_deadline_step[:] = deadline
        self.failed_kick_attempt_buf[:] = failed

    def _reward_failed_kick_attempt(self):
        """Return a one-step event; its scale is chosen for an effective -0.2."""
        return self.failed_kick_attempt_buf.float()

    def _reward_post_kick_walking_pose(self):
        """Reward recovery to the deployed walking policy's nominal posture."""
        walking_pose = self.walking_policy_default_dof_pos.expand_as(
            self.dof_pos
        )
        return post_kick_walking_pose_reward(
            self.dof_pos,
            walking_pose,
            self.valid_kick_buf,
            self.walking_pose_reward_error_scale,
        )

    def _check_termination(self):
        # Parent ball-rest/motion timeouts and success accounting are post-kick
        # concepts.  Initial motion or stillness must not consume those budgets
        # before a valid contact-like kick event has been detected.
        waiting_for_kick = ~self.valid_kick_buf
        self.min_ball_vel_buf[waiting_for_kick] = 0.0
        self.time_since_ball_is_still_buf[waiting_for_kick] = 0.0
        self.time_since_ball_is_moving_buf[waiting_for_kick] = 0.0
        super()._check_termination()
        # The parent also marks command-resample instants as timeouts without
        # resetting the environment. DirectKicking has fixed commands, so only
        # the actual episode time limit is a PPO timeout.
        episode_length_s = float(self._reward_value("episode_length_s"))
        self.time_out_buf[:] = self.episode_length_buf > np.ceil(
            episode_length_s / self.dt
        )
        already_terminated = self.reset_buf.clone()
        post_kick_done = post_kick_termination_mask(
            self.valid_kick_buf,
            self.first_valid_kick_step,
            self.episode_length_buf,
            duration_steps=self.post_kick_duration_steps,
        )
        self.post_kick_terminal_buf[:] = post_kick_done & ~already_terminated
        self.reset_buf |= self.post_kick_terminal_buf
        self.time_out_buf[self.post_kick_terminal_buf] = False

    def _reward_ball_velocity_target_direction(self):
        return (
            super()._reward_ball_velocity_target_direction()
            * self.valid_kick_buf.float()
        )

    def _reward_ball_acceleration(self):
        return super()._reward_ball_acceleration() * self.valid_kick_buf.float()

    def _reward_body_alignment_for_kick(self):
        robot_forward_local = torch.tensor(
            [1.0, 0.0, 0.0],
            device=self.device,
        ).unsqueeze(0).expand(self.num_envs, -1)
        robot_forward_world = quat_rotate(self.base_quat, robot_forward_local)
        robot_forward_xy = robot_forward_world[:, :2]
        robot_forward_xy = robot_forward_xy / (
            torch.clamp(
                torch.norm(robot_forward_xy, dim=-1, keepdim=True),
                min=1.0e-6,
            )
        )
        target_direction = self._unit_kick_target_direction_xy()
        alignment = torch.sum(
            robot_forward_xy * target_direction,
            dim=-1,
        )
        sigma = max(
            float(self._reward_value("alignment_to_target_sigma", 0.5)),
            1.0e-6,
        )
        max_reward = float(self._reward_value("max_alignment_reward", 1.0))
        return torch.clamp(
            torch.exp((alignment - 1.0) / sigma),
            min=0.0,
            max=max_reward,
        )

    def _unit_kick_target_direction_xy(self):
        return normalize_xy_direction(self._ball_to_kick_target_dir_xy())

    def _reward_body_approach_ball(self):
        """Reward moving the base toward either mirrored kicking pose."""
        _, _, current_base_yaw = get_euler_xyz(self.base_quat)
        _, _, previous_base_yaw = get_euler_xyz(
            self.direct_previous_base_quat_world
        )
        ball_pos_world = self.body_states[:, -1, 0:2]
        current_ball_pos_local = self._world_point_to_local(
            ball_pos_world,
            self.base_pos[:, :2],
            current_base_yaw,
        )
        previous_ball_pos_local = self._world_point_to_local(
            ball_pos_world,
            self.direct_previous_base_pos_world[:, :2],
            previous_base_yaw,
        )
        nominal_strike_point = self.direct_cfg.get(
            "mirror_consistency",
            {},
        ).get(
            "nominal_strike_point_m",
            (0.185, 0.096),
        )
        distance_scale = max(
            float(
                self._reward_value(
                    "body_approach_distance_scale",
                    0.05,
                )
            ),
            1.0e-6,
        )
        progress = mirrored_strike_position_progress(
            current_ball_pos_local,
            previous_ball_pos_local,
            nominal_strike_point,
            distance_scale,
        )
        valid_history = self.episode_length_buf > 1
        return (
            progress
            * valid_history.float()
            * (~self.valid_kick_buf).float()
        )

    def _reward_kicking_foot_approach_ball_stationary(self):
        if self._reward_value("kicking_foot_approach_mode", "proximity") != "progress":
            return super()._reward_kicking_foot_approach_ball_stationary()

        distance_scale = max(
            float(
                self._reward_value(
                    "kicking_foot_approach_distance_scale",
                    0.05,
                )
            ),
            1.0e-6,
        )
        ball_pos_world_xy = self.body_states[:, -1, 0:2]
        feet_ankle_pos_world_xy = self._feet_ankle_positions_world()[..., :2]
        progress = nearest_foot_distance_progress(
            feet_ankle_pos_world_xy,
            self.direct_previous_feet_ankle_pos_world[..., :2],
            ball_pos_world_xy,
            distance_scale,
        )
        valid_history = self.episode_length_buf > 1
        return (
            progress
            * valid_history.float()
            * (~self.valid_kick_buf).float()
        )

    def _reward_body_heading_to_ball(self):
        """Penalize pre-kick body headings that face away from the ball."""
        robot_forward_world = quat_rotate(
            self.base_quat,
            torch.tensor(
                [1.0, 0.0, 0.0],
                device=self.device,
            ).unsqueeze(0).expand(self.num_envs, -1),
        )[:, :2]
        robot_to_ball_xy = self.ball_pos[:, :2] - self.base_pos[:, :2]
        forward_norm = torch.norm(robot_forward_world, dim=-1, keepdim=True)
        target_norm = torch.norm(robot_to_ball_xy, dim=-1, keepdim=True)
        alignment = torch.sum(
            robot_forward_world * robot_to_ball_xy,
            dim=-1,
        ) / (forward_norm.squeeze(-1) * target_norm.squeeze(-1) + 1.0e-6)
        heading_error = torch.square(1.0 - torch.clamp(alignment, -1.0, 1.0))
        return heading_error * (~self.valid_kick_buf).float()

    def _reward_kicking_foot_height(self):
        flattened_feet_pos = self.feet_pos.reshape(-1, 3)
        terrain_heights = self._terrain_heights(flattened_feet_pos).reshape(
            self.num_envs,
            self.feet_pos.shape[1],
        )
        foot_heights = self.feet_pos[:, :, 2] - terrain_heights
        height_limit = float(
            self._reward_value("kicking_foot_height_limit", 0.18)
        )
        excess_scale = max(
            float(
                self._reward_value(
                    "kicking_foot_height_excess_scale",
                    0.10,
                )
            ),
            1.0e-6,
        )
        penalty = max_foot_height_excess_penalty(
            foot_heights,
            height_limit,
            excess_scale,
        )
        return penalty * (~self.valid_kick_buf).float()

    def _reward_feet_airborne(self):
        """Penalize losing both support feet before the kick."""
        both_feet_airborne = ~torch.any(self.feet_contact, dim=-1)
        return both_feet_airborne.float() * (~self.valid_kick_buf).float()

    def _toe_and_heel_ground_contact(self):
        edge_positions = self.edge_only_support_edge_positions
        edge_count = edge_positions.shape[0]
        foot_count = len(self.feet_indices)
        local_edges = edge_positions.view(1, 1, edge_count, 3).expand(
            self.num_envs,
            foot_count,
            -1,
            -1,
        )
        feet_positions = self.feet_pos.unsqueeze(2).expand_as(local_edges)
        feet_quaternions = self.feet_quat.unsqueeze(2).expand(
            self.num_envs,
            foot_count,
            edge_count,
            4,
        )
        world_edges = feet_positions + quat_rotate(
            feet_quaternions.reshape(-1, 4),
            local_edges.reshape(-1, 3),
        ).reshape_as(local_edges)
        terrain_heights = self._terrain_heights(
            world_edges.reshape(-1, 3)
        ).reshape(self.num_envs, foot_count, edge_count)
        clearance = float(
            self._reward_value("edge_only_support_contact_clearance_m", 0.01)
        )
        edge_contact = world_edges[..., 2] - terrain_heights < clearance
        toe_contact = torch.any(
            edge_contact[..., self.edge_only_support_toe_edges],
            dim=-1,
        )
        heel_contact = torch.any(
            edge_contact[..., self.edge_only_support_heel_edges],
            dim=-1,
        )
        return toe_contact, heel_contact

    def _reward_edge_only_support(self):
        """Strongly penalize persistent loaded support on only one foot edge."""
        toe_contact, heel_contact = self._toe_and_heel_ground_contact()
        vertical_force = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2],
            min=0.0,
        )
        load_fraction = float(
            self._reward_value("edge_only_support_load_fraction", 0.10)
        )
        gravity_z = abs(float(self.task_cfg["sim"]["gravity"][2]))
        robot_weight = (
            self.robot_body_masses.sum(dim=1, keepdim=True) * gravity_z
        )
        loaded = vertical_force >= robot_weight * load_fraction

        grace_s = max(
            float(self._reward_value("edge_only_support_grace_s", 0.08)),
            0.0,
        )
        required_steps = max(1, int(math.ceil(grace_s / self.dt)))
        active = ~self.valid_kick_buf & ~self.kick_attempt_pending_buf
        self.edge_only_support_steps_buf[:], penalty = (
            update_edge_only_support_penalty(
                toe_contact,
                heel_contact,
                loaded,
                active,
                self.edge_only_support_steps_buf,
                required_steps,
            )
        )
        return penalty

    def _feet_ankle_positions_world(self):
        """Return pitch-invariant ankle positions reconstructed from foot COM poses."""
        feet_local_com = self.robot_body_local_com[:, self.feet_indices]
        feet_com_offsets_world = quat_rotate(
            self.feet_quat.reshape(-1, 4),
            feet_local_com.reshape(-1, 3),
        ).reshape_as(self.feet_pos)
        return self.feet_pos - feet_com_offsets_world

    def _robot_center_of_mass_xy(self):
        """Return the mass-weighted XY center from rigid-body COM states."""
        body_com_positions_xy = self.body_states[:, : self.num_bodies, 0:2]
        total_mass = self.robot_body_masses.sum(dim=1, keepdim=True)
        return (
            self.robot_body_masses.unsqueeze(-1)
            * body_com_positions_xy
        ).sum(dim=1) / total_mass

    def _reward_swing_feet_pitch(self):
        """Discourage excessive pitch of feet ahead of the CoM while approaching."""
        _, feet_pitch, _ = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        feet_pitch = feet_pitch.reshape(self.num_envs, len(self.feet_indices))
        feet_pitch = (feet_pitch + torch.pi) % (2 * torch.pi) - torch.pi
        deadband = float(
            self._reward_value("swing_feet_pitch_deadband_rad", math.radians(10.0))
        )
        pitch_excess = torch.clamp(torch.abs(feet_pitch) - deadband, min=0.0)

        robot_com_xy = self._robot_center_of_mass_xy()
        feet_offset_xy = self.feet_pos[..., :2] - robot_com_xy.unsqueeze(1)
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_forward_of_com = (
            torch.cos(base_yaw).unsqueeze(1) * feet_offset_xy[..., 0]
            + torch.sin(base_yaw).unsqueeze(1) * feet_offset_xy[..., 1]
        ) >= 0.0
        approach_active = (
            ~self.valid_kick_buf & ~self.kick_attempt_pending_buf
        ).unsqueeze(1)
        evaluated_feet = feet_forward_of_com & approach_active
        penalty = torch.sum(
            torch.square(pitch_excess) * evaluated_feet.float(),
            dim=-1,
        )
        return penalty

    def _kick_warmup_steps(self):
        warmup_s = float(
            self.direct_cfg["ball_motion_randomization"].get(
                "kick_detection_warmup_s",
                0.1,
            )
        )
        if warmup_s < 0.0:
            raise ValueError("kick_detection_warmup_s must be non-negative")
        return int(math.ceil(warmup_s / self.dt))

    @staticmethod
    def _validated_range(values, name, positive=False, non_negative=False):
        if len(values) != 2:
            raise ValueError("{} must contain two values".format(name))
        lower = float(values[0])
        upper = float(values[1])
        if lower > upper:
            raise ValueError("{} must be ordered".format(name))
        if positive and lower <= 0.0:
            raise ValueError("{} must be positive".format(name))
        if non_negative and lower < 0.0:
            raise ValueError("{} must be non-negative".format(name))
        return lower, upper

    def _sample_ball_speed(self, count):
        """Symmetric triangular speed, with its mode at the range midpoint."""
        return 0.5 * (
            self._sample_uniform(self.ball_speed_range, count)
            + self._sample_uniform(self.ball_speed_range, count)
        )

    def _sample_uniform(self, value_range, count):
        lower, upper = value_range
        return lower + (upper - lower) * torch.rand(count, device=self.device)

    @staticmethod
    def _world_to_local_rotation(yaw):
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        rotation = torch.zeros(*yaw.shape, 2, 2, device=yaw.device, dtype=yaw.dtype)
        rotation[..., 0, 0] = cosine
        rotation[..., 0, 1] = sine
        rotation[..., 1, 0] = -sine
        rotation[..., 1, 1] = cosine
        return rotation

    @classmethod
    def _world_point_to_local(cls, point, origin, yaw):
        rotation = cls._world_to_local_rotation(yaw)
        return torch.matmul(rotation, (point - origin).unsqueeze(-1)).squeeze(-1)

    @classmethod
    def _local_point_to_world(cls, point, origin, yaw):
        rotation = cls._world_to_local_rotation(yaw)
        return origin + torch.matmul(
            rotation.transpose(-1, -2),
            point.unsqueeze(-1),
        ).squeeze(-1)
