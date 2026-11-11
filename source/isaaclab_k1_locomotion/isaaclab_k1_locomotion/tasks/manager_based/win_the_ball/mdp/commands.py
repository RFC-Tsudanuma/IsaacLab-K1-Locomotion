# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""win_the_ball 固有のキック方向コマンド。"""

import math
import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ...walk_kick.mdp.commands import BallFollowVelocityCommand, KickDirectionCommand


class WinTheBallKickDirectionCommand(KickDirectionCommand):
    """側方・後方を分類し、後方指令だけ目標球速を低くする。

    前方は相対角±45度以内、側方は45度超135度以内、後方は135度超とする。
    """

    rear_target_speed_range = (0.25, 0.8)
    forward_half_angle = math.pi / 4
    rear_start_angle = 3 * math.pi / 4
    straight_rear_half_angle = math.radians(20.0)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reference_yaw = self._robot_yaw(env_ids)
        relative_angle = self._relative_command_angle(env_ids)
        self._is_side_command = self._side_mask(relative_angle)
        self._is_rear_command = self._rear_mask(relative_angle)
        self._is_straight_rear_command = self._straight_rear_mask(relative_angle)
        self.metrics["rear_command_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["side_command_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["straight_rear_command_frac"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_rate_side"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_rate_rear"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_rate_straight_rear"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_dir_error_straight_rear_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_to_kick"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kick_vel_ratio_rear"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["side_contact_quality"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heel_contact_quality_rear"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["facing_error_rear"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["direction_matched_foot_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["right_target_kick_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["right_foot_on_right_target"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["left_target_kick_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["left_foot_on_left_target"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["side_ball_orthogonality"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rear_foot_crossed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["straight_rear_foot_crossed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rear_crossing_clearance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rear_low_crossing_danger"] = torch.zeros(self.num_envs, device=self.device)

    def _robot_yaw(self, env_ids: torch.Tensor) -> torch.Tensor:
        robot_quat = self.robot.data.root_quat_w[env_ids]
        qw, qx, qy, qz = (
            robot_quat[:, 0],
            robot_quat[:, 1],
            robot_quat[:, 2],
            robot_quat[:, 3],
        )
        return torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

    def _relative_command_angle(self, env_ids: torch.Tensor) -> torch.Tensor:
        kick_yaw = torch.atan2(self.command[env_ids, 0], self.command[env_ids, 1])
        reference_yaw = self._reference_yaw[env_ids]
        return torch.atan2(
            torch.sin(kick_yaw - reference_yaw),
            torch.cos(kick_yaw - reference_yaw),
        )

    def _side_mask(self, relative_angle: torch.Tensor) -> torch.Tensor:
        angle = relative_angle.abs()
        return (angle > self.forward_half_angle) & (angle <= self.rear_start_angle)

    def _rear_mask(self, relative_angle: torch.Tensor) -> torch.Tensor:
        return relative_angle.abs() > self.rear_start_angle

    def _straight_rear_mask(self, relative_angle: torch.Tensor) -> torch.Tensor:
        return (math.pi - relative_angle.abs()) <= self.straight_rear_half_angle

    def _update_metrics(self):
        super()._update_metrics()
        state = getattr(self._env, "_kick_latch_state", None)
        if state is None:
            return

        is_rear = self._is_rear_command.float()
        is_side = self._is_side_command.float()
        is_straight_rear = self._is_straight_rear_command.float()
        kick_done = state["kick_done"].float()
        ratio = state["v_ball_frozen"] / torch.clamp(state["v_target"], min=1e-6)
        side_quality = torch.clamp(
            (1.0 - state["foot_kick_dot_frozen"].abs()) / (1.0 - 0.34), min=0.0, max=1.0
        )
        heel_quality = torch.clamp(-state["foot_kick_dot_frozen"], min=0.0, max=1.0)
        env_ids = torch.arange(self.num_envs, device=self.device)
        relative_angle = self._relative_command_angle(env_ids)
        used_right = state["kick_foot_frozen"] == 1.0
        preferred_right = relative_angle < 0.0
        center = relative_angle.abs() <= math.radians(5.0)
        foot_match = (center | (used_right == preferred_right)).float()
        right_target = (relative_angle < -math.radians(5.0)).float()
        left_target = (relative_angle > math.radians(5.0)).float()
        current_yaw = self._robot_yaw(torch.arange(self.num_envs, device=self.device))
        yaw_error = torch.atan2(
            torch.sin(current_yaw - self._reference_yaw),
            torch.cos(current_yaw - self._reference_yaw),
        ).abs()
        self.metrics["rear_command_frac"] = is_rear
        self.metrics["side_command_frac"] = is_side
        self.metrics["straight_rear_command_frac"] = is_straight_rear
        self.metrics["kick_rate_side"] = kick_done * is_side
        self.metrics["kick_rate_rear"] = kick_done * is_rear
        self.metrics["kick_rate_straight_rear"] = kick_done * is_straight_rear
        self.metrics["kick_dir_error_straight_rear_deg"] = (
            torch.rad2deg(state["tau_direction_frozen"]) * kick_done * is_straight_rear
        )
        kick_time = getattr(self._env, "_win_ball_kick_time", torch.zeros_like(kick_done))
        self.metrics["time_to_kick"] = kick_time * kick_done
        self.metrics["kick_vel_ratio_rear"] = ratio * kick_done * is_rear
        self.metrics["side_contact_quality"] = side_quality * kick_done * is_side
        self.metrics["heel_contact_quality_rear"] = heel_quality * kick_done * is_rear
        self.metrics["facing_error_rear"] = torch.rad2deg(yaw_error) * is_rear
        self.metrics["direction_matched_foot_rate"] = foot_match * kick_done
        self.metrics["right_target_kick_rate"] = right_target * kick_done
        self.metrics["right_foot_on_right_target"] = right_target * used_right.float() * kick_done
        self.metrics["left_target_kick_rate"] = left_target * kick_done
        self.metrics["left_foot_on_left_target"] = left_target * (~used_right).float() * kick_done
        side_orthogonality = getattr(
            self._env, "_side_ball_orthogonality_quality", torch.zeros_like(kick_done)
        )
        rear_crossed = getattr(self._env, "_rear_foot_crossed", torch.zeros_like(kick_done, dtype=torch.bool))
        rear_clearance = getattr(self._env, "_rear_crossing_clearance", torch.zeros_like(kick_done))
        rear_danger = getattr(self._env, "_rear_low_crossing_danger", torch.zeros_like(kick_done))
        self.metrics["side_ball_orthogonality"] = side_orthogonality * kick_done * is_side
        self.metrics["rear_foot_crossed"] = rear_crossed.float() * is_rear
        self.metrics["straight_rear_foot_crossed"] = rear_crossed.float() * is_straight_rear
        self.metrics["rear_crossing_clearance"] = rear_clearance * rear_crossed.float() * is_rear
        self.metrics["rear_low_crossing_danger"] = rear_danger

    def _resample_command(self, env_ids: torch.Tensor):
        super()._resample_command(env_ids)

        reference_yaw = self._robot_yaw(env_ids)
        kick_yaw = torch.atan2(self.command[env_ids, 0], self.command[env_ids, 1])
        relative_angle = torch.atan2(
            torch.sin(kick_yaw - reference_yaw),
            torch.cos(kick_yaw - reference_yaw),
        )
        is_side = self._side_mask(relative_angle)
        is_rear = self._rear_mask(relative_angle)
        is_straight_rear = self._straight_rear_mask(relative_angle)
        if hasattr(self, "_is_rear_command"):
            self._reference_yaw[env_ids] = reference_yaw
            self._is_side_command[env_ids] = is_side
            self._is_rear_command[env_ids] = is_rear
            self._is_straight_rear_command[env_ids] = is_straight_rear

        if is_rear.any():
            rear_low, rear_high = self.rear_target_speed_range
            rear_count = int(is_rear.sum())
            rear_speed = torch.empty(rear_count, device=self.device).uniform_(rear_low, rear_high)
            self.command[env_ids[is_rear], 2] = rear_speed


class WinTheBallBallFollowVelocityCommand(BallFollowVelocityCommand):
    """側方・後方指令では回り込まず、正面姿勢を保ってボールへ接近する。"""

    rear_body_offset = 0.2

    def _update_command(self):
        super()._update_command()

        kick_term = self._env.command_manager.get_term(
            self.cfg.kick_direction_command_name or "kick_direction"
        )
        is_side = getattr(kick_term, "_is_side_command", None)
        is_rear = getattr(kick_term, "_is_rear_command", None)
        if is_side is None or is_rear is None:
            return
        keep_facing = is_side | is_rear
        if not keep_facing.any():
            return

        robot = self._env.scene["robot"]
        ball = self._env.scene["soccer_ball"]
        reference_yaw = kick_term._reference_yaw
        reference_forward = torch.stack(
            [torch.cos(reference_yaw), torch.sin(reference_yaw)], dim=-1
        )

        # ボールを跨いで反対側へ行かず、初期正面から見たボール手前で構える。
        rear_target_w = ball.data.root_pos_w[:, :2] - self.rear_body_offset * reference_forward
        to_target_w = torch.zeros(self.num_envs, 3, device=self.device)
        to_target_w[:, :2] = rear_target_w - robot.data.root_pos_w[:, :2]
        to_target_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), to_target_w)
        self.vel_command_b[keep_facing, :2] = torch.clamp(
            to_target_b[keep_facing, :2], -self.cfg.max_vel, self.cfg.max_vel
        )

        # キック方向へ180度回らず、エピソード開始時の正面へ復元する角速度指令。
        quat = robot.data.root_quat_w
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        yaw_error = torch.atan2(
            torch.sin(reference_yaw - current_yaw),
            torch.cos(reference_yaw - current_yaw),
        )
        self.vel_command_b[keep_facing, 2] = torch.clamp(
            yaw_error[keep_facing], -self.cfg.max_ang_vel, self.cfg.max_ang_vel
        )