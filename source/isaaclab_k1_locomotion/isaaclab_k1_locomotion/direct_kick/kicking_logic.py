"""Source-preserving task math from rl_humanoid_htwk 3af2acc9.

Only simulator boundaries/config ownership are adapted. See migration.md.
"""
import os
import csv
import numpy as np
import torch
from .math_xyzw import get_axis_params, to_torch, quat_rotate_inverse, quat_from_euler_xyz, torch_rand_float, get_euler_xyz, quat_rotate
from .utils import apply_randomization

class KickingLogic:
    def _init_buffers(self):
        self.num_obs = self.task_cfg["env"]["num_observations"]
        self.num_privileged_obs = self.task_cfg["env"]["num_privileged_obs"]
        self.num_actions = self.task_cfg["env"]["num_actions"]
        self.dt = self.task_cfg["control"]["decimation"] * self.task_cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_ball_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Buffer for ball-only resets
        self.fall_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.min_ball_vel_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.time_since_ball_is_still_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_since_ball_is_moving_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.valid_kick_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        self.root_states = torch.zeros(self.num_envs, 2, 13, device=self.device)
        self.root_states[..., 6] = 1.0
        self.dof_state = torch.zeros(self.num_envs, self.num_dofs, 2, device=self.device)
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies + 1, 3, device=self.device)
        self.body_states = torch.zeros(self.num_envs, self.num_bodies + 1, 13, device=self.device)
        self.body_states[..., 6] = 1.0
        # Get robot states (index 0) and ball states (index 1)
        self.base_pos = self.root_states[:, 0, 0:3]  # Robot position
        self.base_quat = self.root_states[:, 0, 3:7]  # Robot quaternion
        self.ball_pos = self.root_states[:, 1, 0:3]  # Ball position
        self.ball_rot = self.root_states[:, 1, 3:7]  # Ball quaternion
        self.ball_lin_vel = self.body_states[:, -1, 7:10]  # Ball linear velocity
        self.ball_ang_vel = self.body_states[:, -1, 10:13]  # Ball angular velocity
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.kick_target_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.kick_target_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.kick_target_dir_world = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.kick_target_dir_world[:, 0] = 1.0
        self.kick_start_ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.kick_target_pos_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        # Only apply gravity to robot's state
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.task_cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.task_cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.task_cfg["commands"]["lin_vel_levels"], self.task_cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.task_cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.task_cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.task_cfg["init_state"]["default_joint_angles"]["default"]

        self.last_ball_lin_vel_world = torch.zeros_like(self.body_states[:, -1, 7:10]) # World frame

    def _init_stage_curriculum(self):
        self.stage_curriculum_cfg = self.task_cfg.get("stage_curriculum", {})
        self.stage_curriculum_enabled = self.stage_curriculum_cfg.get("enabled", False)
        self.stage_curriculum_stages = self.stage_curriculum_cfg.get("stages", [])
        self.curriculum_global_step = 0
        self.curriculum_total_env_steps = 0
        self.curriculum_iteration = 0
        self.active_stage_idx = -1
        self.active_stage = {}

    def set_curriculum_progress(self, global_step, iteration=None, total_env_steps=None):
        self.curriculum_global_step = int(global_step)
        self.curriculum_total_env_steps = int(total_env_steps if total_env_steps is not None else global_step)
        self.curriculum_iteration = int(iteration or 0)
        if not self.stage_curriculum_enabled or not self.stage_curriculum_stages:
            return False

        new_stage_idx = len(self.stage_curriculum_stages) - 1
        for idx, stage in enumerate(self.stage_curriculum_stages):
            end_iteration = stage.get("end_iteration")
            if end_iteration is not None:
                if self.curriculum_iteration < int(end_iteration):
                    new_stage_idx = idx
                    break
                continue
            end_step = stage.get("end_step")
            if end_step is None or self.curriculum_global_step < int(end_step):
                new_stage_idx = idx
                break

        if new_stage_idx == self.active_stage_idx:
            return False

        self._apply_stage(new_stage_idx)
        return True

    def _apply_stage(self, stage_idx):
        self.active_stage_idx = stage_idx
        self.active_stage = self.stage_curriculum_stages[stage_idx] if stage_idx >= 0 else {}
        stage_name = self.active_stage.get("name", "disabled")

        base_scales = {} if self.active_stage.get("replace_reward_scales", False) else self.reward_scales_base.copy()
        base_scales.update(self.active_stage.get("reward_scales", {}))
        self.reward_scales = {
            key: value * self.dt
            for key, value in base_scales.items()
            if value != 0
        }

        ball_rolling_scales = {} if self.active_stage.get("replace_ball_rolling_scale", False) else self.reward_scales_ball_rolling_base.copy()
        ball_rolling_scales.update(self.active_stage.get("ball_rolling_scale", {}))
        self.reward_scales_ball_rolling = {
            key: value * self.dt
            for key, value in ball_rolling_scales.items()
        }
        print(
            "Stage curriculum: iteration={}, per_env_step={}, total_env_steps={}, "
            "stage={}, name={}".format(
                self.curriculum_iteration,
                self.curriculum_global_step,
                self.curriculum_total_env_steps,
                stage_idx,
                stage_name,
            )
        )

    def _stage_value(self, section, key, default=None):
        stage_section = self._sampling_stage().get(section, {})
        if key in stage_section:
            return stage_section[key]
        return self.task_cfg.get(section, {}).get(key, default)

    def _sampling_stage(self):
        return self.active_stage

    def _reward_value(self, key, default=None):
        reward_params = self.active_stage.get("reward_params", {})
        if key in reward_params:
            return reward_params[key]
        return self.task_cfg["rewards"].get(key, default)

    def _kicking_foot_index(self):
        kicking_foot_index = int(self._reward_value("kicking_foot_index", 0))
        if not 0 <= kicking_foot_index < self.feet_pos.shape[1]:
            raise ValueError(
                f"kicking_foot_index must be in [0, {self.feet_pos.shape[1] - 1}], got {kicking_foot_index}"
            )
        return kicking_foot_index

    def _update_valid_kick(self, previous_ball_pos_world, previous_ball_lin_vel_world):
        if not self._reward_value("use_valid_kick_gating", False):
            return

        kicking_foot_index = self._kicking_foot_index()
        current_kicking_foot_pos = self.feet_pos[:, kicking_foot_index, :]
        previous_kicking_foot_pos = self.last_feet_pos[:, kicking_foot_index, :]
        current_foot_ball_distance = torch.norm(
            current_kicking_foot_pos - self.ball_pos,
            dim=-1,
        )
        previous_foot_ball_distance = torch.norm(
            previous_kicking_foot_pos - previous_ball_pos_world,
            dim=-1,
        )
        foot_ball_distance = torch.minimum(current_foot_ball_distance, previous_foot_ball_distance)
        max_foot_ball_distance = float(self._reward_value("kick_detection_foot_distance_threshold", 0.23))

        previous_foot_to_ball = previous_ball_pos_world - previous_kicking_foot_pos
        previous_foot_to_ball_direction = previous_foot_to_ball / (
            torch.norm(previous_foot_to_ball, dim=-1, keepdim=True) + 1.0e-6
        )
        kicking_foot_velocity = (current_kicking_foot_pos - previous_kicking_foot_pos) / self.dt
        foot_speed_towards_ball = torch.sum(kicking_foot_velocity * previous_foot_to_ball_direction, dim=-1)
        min_foot_speed = float(self._reward_value("kick_detection_min_foot_speed_towards_ball", 0.2))

        target_direction = self._ball_to_kick_target_dir_xy()
        previous_target_velocity = torch.sum(previous_ball_lin_vel_world[:, :2] * target_direction, dim=-1)
        current_target_velocity = torch.sum(self.root_states[:, 1, 7:9] * target_direction, dim=-1)
        target_velocity_increase = current_target_velocity - previous_target_velocity
        min_velocity_increase = float(self._reward_value("kick_detection_speed_increase_threshold", 0.5))

        valid_kick = (
            (foot_ball_distance <= max_foot_ball_distance)
            & (foot_speed_towards_ball >= min_foot_speed)
            & (target_velocity_increase >= min_velocity_increase)
            & (current_target_velocity >= min_velocity_increase)
        )
        self.valid_kick_buf |= valid_kick

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self.reward_scales_base = self.task_cfg["rewards"]["scales"].copy()
        self.reward_scales_ball_rolling_base = self.task_cfg["rewards"].get("ball_rolling_scale", {}).copy()

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        reward_names = set(self.reward_scales_base.keys())
        for stage in self.stage_curriculum_stages:
            reward_names.update(stage.get("reward_scales", {}).keys())
        for name in sorted(reward_names):
            if not hasattr(self, "_reward_" + name):
                print(f"Warning: reward function _reward_{name} is not defined")
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

        if self.stage_curriculum_enabled and self.stage_curriculum_stages:
            self._apply_stage(0)
        else:
            self.active_stage_idx = -1
            self.active_stage = {}
            self.reward_scales = {
                key: value * self.dt
                for key, value in self.reward_scales_base.items()
                if value != 0
            }
            self.reward_scales_ball_rolling = {
                key: value * self.dt
                for key, value in self.reward_scales_ball_rolling_base.items()
            }

    def _init_csv_logging(self):
        """Initialize CSV logging for reward values"""
        # Check if CSV logging is enabled in config
        self.csv_logging_enabled = self.task_cfg.get("basic", {}).get("enable_csv_logging", True)
        
        if not self.csv_logging_enabled:
            print("CSV logging disabled in configuration")
            return
            
        # Only log for environment 0 (single environment setup)
        self.log_env_id = 0
        
        # Create logs directory if it doesn't exist
        self.log_dir = self.task_cfg["basic"].get("log_dir", "logs/direct_kick")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create CSV file with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.log_dir, f"debug_csv/reward_log_{timestamp}.csv")
        os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
        self.csv_episode_id = 0
        
        # Prepare CSV headers
        self.csv_headers = [
            "episode_id", "episode_step", "total_reward",
            "ball_pos_x", "ball_pos_y", "ball_pos_z",
            "ball_vel_x", "ball_vel_y", "ball_vel_z",
            "robot_pos_x", "robot_pos_y", "robot_pos_z",
            "robot_lin_vel_x", "robot_lin_vel_y", "robot_lin_vel_z",
            "base_height", "base_pitch", "base_roll",
            "left_foot_x", "left_foot_y", "left_foot_z",
            "right_foot_x", "right_foot_y", "right_foot_z",
            "left_foot_contact", "right_foot_contact",
            "left_foot_ball_distance", "right_foot_ball_distance",
            "action_norm", "ball_speed", "ball_distance_to_robot",
            "valid_kick", "fall"
        ]
        reward_names = ["reward_" + name for name in self.reward_names]
        if "reward_kick_direction" not in reward_names:
            reward_names.append("reward_kick_direction")
        self.csv_headers.extend(reward_names)  # Add all individual reward terms
        
        # Initialize CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_headers)
        
        print(f"CSV logging initialized: {self.csv_filename}")

    def _log_rewards_to_csv(self):
        """Log current step rewards to CSV file"""
        # Check if CSV logging is enabled
        if not getattr(self, 'csv_logging_enabled', False):
            return
            
        # Only log for the specified environment (0)
        if hasattr(self, 'episode_length_buf'):
            episode_step = self.episode_length_buf[self.log_env_id].item()
            total_reward = self.rew_buf[self.log_env_id].item()
            
            # Get ball and robot state information
            ball_pos = self.ball_pos[self.log_env_id].cpu().numpy()
            ball_vel_world = self.root_states[self.log_env_id, 1, 7:10].cpu().numpy()
            robot_pos = self.base_pos[self.log_env_id].cpu().numpy()
            robot_lin_vel = self.base_lin_vel[self.log_env_id].cpu().numpy()
            feet_pos = self.feet_pos[self.log_env_id].cpu().numpy()
            feet_contact = self.feet_contact[self.log_env_id].cpu().numpy()
            base_height = (
                self.base_pos[self.log_env_id, 2]
                - self._terrain_heights(self.base_pos[self.log_env_id].unsqueeze(0))[0]
            ).item()
            base_pitch, base_roll, _ = get_euler_xyz(
                self.base_quat[self.log_env_id].unsqueeze(0)
            )
            foot_ball_distances = torch.norm(
                self.feet_pos[self.log_env_id]
                - self.ball_pos[self.log_env_id].unsqueeze(0),
                dim=-1,
            ).cpu().numpy()
            
            # Calculate derived metrics
            ball_speed = torch.norm(self.root_states[self.log_env_id, 1, 7:10]).item()
            ball_distance_to_robot = torch.norm(self.ball_pos[self.log_env_id] - self.base_pos[self.log_env_id]).item()
            action_norm = torch.norm(self.actions[self.log_env_id]).item()
            
            # Prepare row data
            row_data = [
                self.csv_episode_id, episode_step, total_reward,
                ball_pos[0], ball_pos[1], ball_pos[2],
                ball_vel_world[0], ball_vel_world[1], ball_vel_world[2],
                robot_pos[0], robot_pos[1], robot_pos[2],
                robot_lin_vel[0], robot_lin_vel[1], robot_lin_vel[2],
                base_height, base_pitch[0].item(), base_roll[0].item(),
                feet_pos[0, 0], feet_pos[0, 1], feet_pos[0, 2],
                feet_pos[1, 0], feet_pos[1, 1], feet_pos[1, 2],
                int(feet_contact[0]), int(feet_contact[1]),
                foot_ball_distances[0], foot_ball_distances[1],
                action_norm, ball_speed, ball_distance_to_robot,
                int(self.valid_kick_buf[self.log_env_id].item()),
                int(self.fall_buf[self.log_env_id].item()),
            ]
            
            # Add individual reward terms
            for reward_name in self.reward_names:
                if reward_name in self.extras["rew_terms"]:
                    reward_value = self.extras["rew_terms"][reward_name][self.log_env_id].item()
                    row_data.append(reward_value)
                else:
                    row_data.append(0.0)  # Default if reward term not found

            kick_direction = self.extras["rew_terms"].get("kick_direction")
            row_data.append(
                kick_direction[self.log_env_id].item()
                if kick_direction is not None
                else 0.0
            )
            
            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        self.env_resets += env_ids.shape[0]
        if (
            getattr(self, "csv_logging_enabled", False)
            and (env_ids == self.log_env_id).any()
        ):
            self.csv_episode_id += 1

        velocity_before_reset = torch.norm(self.root_states[env_ids, 1, 7:10], dim=1)
        
        # only get velocities that are greater than 0.1
        velocity_before_reset = velocity_before_reset[velocity_before_reset > 0.1]
        if len(velocity_before_reset) > 0:
            # append velocities sepeate
            for velocity in velocity_before_reset:
                self.ball_velocities.append(velocity.item())

        # Reset robot
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Continue with existing reset code...
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.min_ball_vel_buf[env_ids] = 0.0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.time_since_ball_is_still_buf[env_ids] = 0.0
        self.time_since_ball_is_moving_buf[env_ids] = 0.0
        self.valid_kick_buf[env_ids] = False
        self.cmd_resample_time[env_ids] = 0

        self.delay_steps[env_ids] = torch.randint(0, self.task_cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf
        self.last_ball_lin_vel_world[env_ids] = 0.0 # Reset for selected envs
        self._resample_commands(env_ids)

    def _reset_root_states(self, env_ids):
        # Initialize robot states (index 0)
        self.root_states[env_ids, 0, :] = self.base_init_state
        self.root_states[env_ids, 0, :2] += self.env_origins[env_ids, :2]
        #self.root_states[env_ids, 0, :2] = apply_randomization(self.root_states[env_ids, 0, :2], self.task_cfg["randomization"].get("init_base_pos_xy"))
        base_roll_pitch = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.task_cfg["randomization"].get("init_base_roll_pitch"),
        )
        self.root_states[env_ids, 0, 3:7] = quat_from_euler_xyz(
            base_roll_pitch[:, 0],
            base_roll_pitch[:, 1],
            apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.task_cfg["randomization"].get("init_base_ang")
                ),
        )
        self.root_states[env_ids, 0, 2] = self._initial_root_height(env_ids)
        self.root_states[env_ids, 0, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.task_cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.root_states[env_ids, 0, 10:13] = apply_randomization(
            torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device),
            self.task_cfg["randomization"].get("init_base_ang_vel"),
        )

        # Reset ball in front of the (newly reset) robot
        self._reset_ball_at_robot_front(env_ids)

        self._write_root_states(env_ids, robot=True, ball=True)

    def _initial_root_height(self, env_ids):
        terrain_height = self._terrain_heights(
            self.root_states[env_ids, 0, :2]
        )
        return self.base_init_state[2] + terrain_height

    def _kick_target_distance_range(self, command_cfg=None):
        if command_cfg is None:
            command_cfg = self._sampling_stage().get("commands", self.task_cfg["commands"])
        return self._sampling_stage().get(
            "target_distance",
            command_cfg.get("target_distance", self.task_cfg["commands"].get("target_distance", [3.0, 5.0])),
        )

    def _update_kick_target_positions(self, env_ids):
        self.kick_start_ball_pos[env_ids] = self.ball_pos[env_ids]
        self.kick_target_pos_world[env_ids, 0:2] = (
            self.kick_start_ball_pos[env_ids, 0:2]
            + self.kick_target_dir_world[env_ids] * self.kick_target_distance[env_ids].unsqueeze(-1)
        )
        self.kick_target_pos_world[env_ids, 2] = self.kick_start_ball_pos[env_ids, 2]

    def _resample_commands(self, env_ids=None):
        if env_ids is None:
            env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        sampling_stage = self._sampling_stage()
        command_cfg = sampling_stage.get("commands", self.task_cfg["commands"])
        if command_cfg.get("curriculum", self.task_cfg["commands"].get("curriculum", False)):
            self._resample_curriculum_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                command_cfg["lin_vel_x"][0], command_cfg["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                command_cfg["lin_vel_y"][0], command_cfg["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                command_cfg["ang_vel_yaw"][0], command_cfg["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            command_cfg["gait_frequency"][0], command_cfg["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        target_yaw_range = sampling_stage.get("target_yaw", self.task_cfg["commands"].get("target_yaw", [0.0, 0.0]))
        self.kick_target_yaw[env_ids] = torch_rand_float(
            target_yaw_range[0], target_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.kick_target_dir_world[env_ids, 0] = torch.cos(self.kick_target_yaw[env_ids])
        self.kick_target_dir_world[env_ids, 1] = torch.sin(self.kick_target_yaw[env_ids])
        target_distance_range = self._kick_target_distance_range(command_cfg)
        self.kick_target_distance[env_ids] = torch_rand_float(
            target_distance_range[0], target_distance_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self._update_kick_target_positions(env_ids)

        still_proportion = command_cfg.get("still_proportion", self.task_cfg["commands"].get("still_proportion", 0.0))
        still_envs = env_ids[torch.randperm(len(env_ids), device=self.device)[: int(still_proportion * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        resampling_time_s = command_cfg.get("resampling_time_s", self.task_cfg["commands"].get("resampling_time_s", [2.0, 4.0]))
        min_resample_steps = int(resampling_time_s[0] / self.dt)
        max_resample_steps = int(resampling_time_s[1] / self.dt)
        if min_resample_steps >= max_resample_steps:
            self.cmd_resample_time[env_ids] += min_resample_steps
        else:
            self.cmd_resample_time[env_ids] += torch.randint(
                min_resample_steps,
                max_resample_steps,
                (len(env_ids),),
                device=self.device,
            )

    def _update_curriculum(self, env_ids):
        if not self.task_cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self._reward_value("episode_length_s") / self.dt) * (
            1 - self.task_cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.task_cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.task_cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.task_cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.task_cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.task_cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.task_cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.task_cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.task_cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.task_cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.task_cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.task_cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.task_cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.task_cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.task_cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.task_cfg["commands"]["ang_vel_resolution"]

    def _update_delayed_dof_targets(self, dof_targets, substep_index):
        """Apply the task's command delay before the next physics substep."""
        delayed_envs = self.delay_steps == substep_index
        self.last_dof_targets[delayed_envs] = dof_targets[delayed_envs]

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.task_cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self._terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _check_termination(self):
        """Check if environments need to be reset"""
        terminate_vel = self._reward_value("terminate_vel")
        terminate_height = self._reward_value("terminate_height")
        episode_length_s = self._reward_value("episode_length_s")
        termination_contact = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        excessive_root_vel = self.root_states[:, 0, 7:13].square().sum(dim=-1) > terminate_vel
        low_base_height = self.base_pos[:, 2] - self._terrain_heights(self.base_pos) < terminate_height
        self.fall_buf = termination_contact | excessive_root_vel | low_base_height
        self.reset_buf = self.fall_buf.clone()
        self.time_out_buf = self.episode_length_buf > np.ceil(episode_length_s / self.dt)
        self.reset_buf |= self.time_out_buf
        walking_stage = self.active_stage.get("mode", "").startswith("walk")
        if not walking_stage:
            self.reset_buf |= self.min_ball_vel_buf > np.ceil(self._reward_value("min_ball_vel_s") / self.dt)
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

        if not walking_stage:
            # Add termination if ball is still for too long
            max_ball_still_time = self._reward_value("max_ball_still_time_s", 4.0)
            self.reset_buf |= self.time_since_ball_is_still_buf > max_ball_still_time

            # Add termination if ball is moving for too long
            max_ball_moving_time = self._reward_value("max_ball_moving_time_s", 4.0)
            self.reset_buf |= self.time_since_ball_is_moving_buf > max_ball_moving_time

            # count a success if ball is moving for too long
            self.env_successes += torch.sum(self.min_ball_vel_buf > np.ceil(self._reward_value("min_ball_vel_s") / self.dt))
        self.env_falling += int(torch.sum(self.fall_buf).item())

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """

        ball_is_moving = torch.norm(self.ball_lin_vel, dim=-1) >= 0.1  # True if ball is "still"

        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            normal_scale = self.reward_scales.get(name, 0.0) # Scalar, default scale for this reward
            ball_moving_specific_scale = self.reward_scales_ball_rolling.get(name) # Scalar or None

            if normal_scale == 0.0 and (ball_moving_specific_scale is None or ball_moving_specific_scale == 0.0):
                self.extras["rew_terms"][name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                continue

            raw_reward_values = self.reward_functions[i]() # Shape: (num_envs)

            # Initialize effective_scales with normal_scale. This applies if:
            # 1. No specific ball_rolling_scale is defined for this reward.
            # 2. A specific ball_rolling_scale is defined, but the ball is NOT moving.
            effective_scales_for_envs = torch.full_like(raw_reward_values, normal_scale)

            if ball_moving_specific_scale is not None:
                # A specific scale for a moving ball exists for this reward.
                # We apply this scale IF the ball is moving.
                # If the ball is not moving, effective_scales_for_envs (which is normal_scale) is used.
                effective_scales_for_envs = torch.where(ball_is_moving,
                                                        torch.full_like(raw_reward_values, ball_moving_specific_scale),
                                                        effective_scales_for_envs)

            rew = raw_reward_values * effective_scales_for_envs
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew # Store the final scaled reward
        if self.task_cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _ball_to_kick_target_dir_xy(self):
        ball_to_target_xy = self.kick_target_pos_world[:, 0:2] - self.kick_start_ball_pos[:, 0:2]
        distance = torch.norm(ball_to_target_xy, dim=-1, keepdim=True)
        target_dir = ball_to_target_xy / (distance + 1.0e-6)
        return torch.where(distance > 1.0e-6, target_dir, self.kick_target_dir_world)

    def _reward_ball_travel_distance_target(self):
        travel_distance = torch.norm(self.ball_pos[:, 0:2] - self.kick_start_ball_pos[:, 0:2], dim=-1)
        distance_error = travel_distance - self.kick_target_distance
        sigma = self._reward_value("ball_travel_distance_sigma", 0.5)
        max_reward = self._reward_value("max_ball_travel_distance_reward", 1.0)
        return torch.exp(-torch.square(distance_error) / sigma) * max_reward

    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_fall(self):
        return self.fall_buf.float()

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / self._reward_value("tracking_sigma"))

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / self._reward_value("tracking_sigma"))

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / self._reward_value("tracking_sigma"))

    def _reward_commanded_forward_vel(self):
        # Gives no credit for standing still when a forward command is active.
        command = torch.clamp(self.commands[:, 0], min=0.0)
        active = (command > 1.0e-3).float()
        progress = torch.clamp(self.filtered_lin_vel[:, 0] / (command + 1.0e-6), min=0.0, max=1.0)
        return progress * active

    def _reward_stand_still(self):
        # Penalize near-zero planar speed only when the policy is commanded to move.
        command_active = (torch.abs(self.commands[:, 0]) + torch.abs(self.commands[:, 1]) + torch.abs(self.commands[:, 2]) > 1.0e-3).float()
        planar_speed_sq = torch.sum(torch.square(self.filtered_lin_vel[:, :2]), dim=-1)
        stand_still_sigma = self._reward_value("stand_still_sigma", 0.03)
        return torch.exp(-planar_speed_sq / stand_still_sigma) * command_active

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self._terrain_heights(self.base_pos)
        return torch.square(base_height - self._reward_value("base_height_target"))

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 0, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        soft_dof_pos_limit = self._reward_value("soft_dof_pos_limit")
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - soft_dof_pos_limit) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - soft_dof_pos_limit) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self._reward_value("soft_dof_vel_limit")).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self._reward_value("soft_torque_limit")).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_kicking_foot_height(self):
        """Penalize lifting the configured kicking foot excessively before a valid kick."""
        kicking_foot_pos = self.feet_pos[:, self._kicking_foot_index(), :]
        foot_height = kicking_foot_pos[:, 2] - self._terrain_heights(kicking_foot_pos)
        height_limit = float(self._reward_value("kicking_foot_height_limit", 0.18))
        excess_scale = max(float(self._reward_value("kicking_foot_height_excess_scale", 0.10)), 1.0e-6)
        height_excess = torch.clamp(foot_height - height_limit, min=0.0) / excess_scale

        penalty_is_active = ~self.valid_kick_buf
        return torch.square(height_excess) * penalty_is_active.float()

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        return torch.clip(self._reward_value("feet_distance_ref", 0.192) - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        swing_period = self._reward_value("swing_period")
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * swing_period) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * swing_period) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()

    def _reward_left_feet_x(self):
        # Calculate the distance between feet on the x-axis
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_x_distance =  torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
        )
        
        # Get the reference distance from config
        target_distance = self.task_cfg["rewards"]["feet_distance_ref_x"]
        
        # Normalize the difference between actual and target distance
        normalized_diff = -torch.abs(feet_x_distance - target_distance)
        
        reward = torch.exp(2.0 * (normalized_diff - 1.0) + 2)  # Exponential increase

        #print(f"X-axis: feet_x_distance={feet_x_distance.mean().item():.4f}, target={target_distance:.4f}, normalized_diff={normalized_diff.mean().item():.4f}, reward={reward.mean().item():.4f}")
        
        return reward

    def _reward_left_feet_y(self):
        # Calculate the distance between feet on the y-axis
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_y_distance = torch.abs(
            torch.sin(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            + torch.cos(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        
        # Get the reference distance from config
        target_distance = self.task_cfg["rewards"]["feet_distance_ref_y"]
        
        # Normalize the difference between actual and target distance
        normalized_diff = -torch.abs(feet_y_distance - target_distance)
        
        reward = torch.exp(2.0 * (normalized_diff - 1.0) + 2)  # Exponential increase
        
        #print(f"Y-axis: feet_y_distance={feet_y_distance.mean().item():.4f}, target={target_distance:.4f}, normalized_diff={normalized_diff.mean().item():.4f}, reward={reward.mean().item():.4f}")
        
        return reward

    def _reward_ball_velocity_target_direction(self):
        """Rewards kicking the ball towards a target position in the world frame."""
        # cfg["rewards"]["ball_target_position"] - e.g. [5.0, 0.0, 0.0] (target position in world space)
        # cfg["rewards"]["max_ball_vel_target_reward"] - max reward for this component
        # cfg["rewards"]["ball_vel_target_direction_sigma"] - for scaling the reward
        # cfg["rewards"]["ball_velocity_decay_time"] - time constant for exponential decay when ball is moving
        
        ball_vel_world = self.body_states[:, -1, 7:10]
        target_dir = torch.cat(
            (self._ball_to_kick_target_dir_xy(), torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)),
            dim=-1,
        )
        
        # Project ball velocity onto the target direction
        velocity_towards_target = torch.sum(ball_vel_world * target_dir, dim=-1)
        
        # Reward only positive velocity towards the target
        # Using an exponential function for a smoother reward landscape
        sigma = self.task_cfg["rewards"].get("ball_vel_target_direction_sigma", 1.0)
        base_reward = velocity_towards_target
        
        # Add decay factor based on how long the ball has been moving
        decay_time_constant = self.task_cfg["rewards"].get("ball_velocity_decay_time", 2.0)  # Time constant in seconds
        decay_factor = torch.exp(-self.time_since_ball_is_moving_buf / decay_time_constant)
        
        # Apply decay to the reward
        reward = base_reward * decay_factor
        
        # Clamp the reward to avoid excessively large values
        max_reward = self.task_cfg["rewards"].get("max_ball_vel_target_reward", 5.0)
        
        return torch.clamp(reward, min=0.0, max=max_reward)

    def _reward_kicking_foot_approach_ball_stationary(self):
        """Rewards moving the kicking foot towards the ball, only if the ball is stationary."""
        # cfg["rewards"]["ball_stationary_speed_threshold"] - Max speed for ball to be "stationary"
        # cfg["rewards"]["approach_proximity_sigma"] - For exp decay of distance reward
        # cfg["rewards"]["max_approach_reward"]
        # cfg["rewards"]["foot_velocity_towards_ball_scale"] - Scale for velocity reward
        # cfg["rewards"]["foot_velocity_weight"] - Weight for velocity vs proximity reward (0-1)

        current_ball_pos_world = self.body_states[:, -1, 0:3]
        approach_mode = self._reward_value("kicking_foot_approach_mode", "proximity")

        if approach_mode == "progress":
            distance_scale = max(
                float(self._reward_value("kicking_foot_approach_distance_scale", 0.05)),
                1.0e-6,
            )
            kicking_foot_index = self._kicking_foot_index()
            current_distance = torch.norm(
                self.feet_pos[:, kicking_foot_index, :] - current_ball_pos_world,
                dim=-1,
            )
            previous_distance = torch.norm(
                self.last_feet_pos[:, kicking_foot_index, :] - current_ball_pos_world,
                dim=-1,
            )
            distance_progress = (previous_distance - current_distance) / distance_scale
            valid_history = self.episode_length_buf > 1
            return (
                distance_progress
                * valid_history.float()
                * (~self.valid_kick_buf).float()
            )

        foot_ball_dist_left = torch.norm(self.feet_pos[:, 0, :] - current_ball_pos_world, dim=-1)
        foot_ball_dist_right = torch.norm(self.feet_pos[:, 1, :] - current_ball_pos_world, dim=-1)

        foot_ball_dist = torch.min(foot_ball_dist_left, foot_ball_dist_right)

        # Proximity reward (existing)
        proximity_sigma = self.task_cfg["rewards"].get("approach_proximity_sigma", 0.1)
        proximity_value = torch.exp(-foot_ball_dist / proximity_sigma) 
        
        # Only give reward if the ball is stationary
        reward = proximity_value

        max_reward = self.task_cfg["rewards"].get("max_approach_reward", 2.0)
        return torch.clamp(reward, min=0.0, max=max_reward)

    def _reward_body_alignment_for_kick(self):
        """Rewards aligning the robot's body towards the ball and a target."""
        # cfg["rewards"]["kick_target_pos_world"] - e.g., [5.0, 0.0, 0.0] (a point in world space)
        # cfg["rewards"]["alignment_to_ball_sigma"]
        # cfg["rewards"]["alignment_to_target_sigma"]
        # cfg["rewards"]["max_alignment_reward"]

        robot_pos_world = self.base_pos
        robot_quat_world = self.base_quat
        ball_pos_world = self.root_states[:, 1, 0:3]
        
        # Robot's forward vector in world frame
        # Assuming robot's local forward is X-axis: [1,0,0]
        robot_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        robot_forward_world = quat_rotate(robot_quat_world, robot_forward_local)
        
        # Vector from robot to ball
        robot_to_ball_world = ball_pos_world - robot_pos_world
        robot_to_ball_world_normalized = robot_to_ball_world / (torch.norm(robot_to_ball_world, dim=-1, keepdim=True) + 1e-6)

        robot_to_target_world = self.kick_target_pos_world - robot_pos_world
        robot_to_target_world_normalized = robot_to_target_world / (torch.norm(robot_to_target_world, dim=-1, keepdim=True) + 1e-6)
        
        alignment_to_target = torch.sum(robot_forward_world * robot_to_target_world_normalized, dim=-1)
        sigma_target = self.task_cfg["rewards"].get("alignment_to_target_sigma", 0.5)
        reward_align_target = torch.exp((alignment_to_target - 1.0) / sigma_target)


        max_reward = self.task_cfg["rewards"].get("max_alignment_reward", 1.0)
        return torch.clamp(reward_align_target, min=0.0, max=max_reward)

    def _reward_body_angle(self):
        base_pitch, base_roll, base_yaw = get_euler_xyz(self.base_quat)
        pitch_normalized = (base_pitch + torch.pi) % (2 * torch.pi) - torch.pi
        roll_normalized = (base_roll + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Calculate absolute distance from 0
        pitch_distance = torch.abs(pitch_normalized)
        roll_distance = torch.abs(roll_normalized)
        
        # Calculate penalty (current implementation)
        penalty = torch.square(pitch_distance) + torch.square(roll_distance)
        
        # Convert to positive reward (decreasing from 1 to 0)
        reward = 1.0 / (0.1 + penalty**2) - 1.0
        
        return reward

    def _reward_ball_acceleration(self):
        """Rewards ball acceleration towards the target direction, encouraging effective kicks."""
        # Get current and previous ball velocities in world frame
        current_ball_vel_world = self.body_states[:, -1, 7:9]
        prev_ball_vel_world = self.last_ball_lin_vel_world[:,:2]
        
        # Calculate ball acceleration (change in velocity / time)
        ball_acceleration = (current_ball_vel_world - prev_ball_vel_world) / self.dt

        ball_effective_acceleration = torch.sum(ball_acceleration * self._ball_to_kick_target_dir_xy(), dim=-1)
        
        # Get parameters from config with defaults
        acceleration_scale = self.task_cfg["rewards"].get("ball_acceleration_scale", 10.0)
        max_acceleration_reward = self.task_cfg["rewards"].get("max_ball_acceleration_reward", 1.0)
        
        # Only reward positive acceleration towards target
        # Using tanh for smooth, bounded rewardball_effective_acceleration
        reward = torch.tanh(torch.clamp(ball_effective_acceleration, min=0.0) / acceleration_scale) * max_acceleration_reward
        
        return reward

    def _reward_waiting(self):
        """Penalty basis that grows quadratically until a valid kick is detected."""
        if not self._reward_value("use_valid_kick_gating", False):
            max_wait_steps = np.ceil(self.task_cfg["rewards"]["max_ball_still_time_s"] / self.dt)
            progress = self.episode_length_buf / max_wait_steps
            return torch.square(progress)

        max_wait_time = max(float(self._reward_value("max_ball_still_time_s", 2.0)), 1.0e-6)
        elapsed_time = self.episode_length_buf.float() * self.dt
        progress = torch.clamp(elapsed_time / max_wait_time, min=0.0, max=1.0)
        return torch.square(progress) * (~self.valid_kick_buf).float()
