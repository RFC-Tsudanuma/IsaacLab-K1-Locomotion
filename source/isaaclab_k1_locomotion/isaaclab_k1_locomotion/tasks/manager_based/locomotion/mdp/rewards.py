# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, euler_xyz_from_quat, wrap_to_pi
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .data_logger import send_data_stream

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def minimum_height(env: ManagerBasedRLEnv, min_height: float = 0.47, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """
    minimum heightよりもロボットの高さが低い場合にペナルティを与える報酬関数
    """
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:  # これはRaycasterである必要あり.roughの場合に使う
        sensor = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_min_height = min_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_min_height = min_height
    # Compute the L2 squared penalty
    return torch.where(asset.data.root_pos_w[:, 2] < adjusted_min_height, torch.square(asset.data.root_pos_w[:, 2] - adjusted_min_height), torch.zeros_like(asset.data.root_pos_w[:, 2]))

def feet_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.15,
) -> torch.Tensor:
    """Penalize when the two feet are closer than ``min_distance`` in the XY plane.

    This prevents the robot from crossing its legs or dragging one foot past the other.

    Args:
        env: The learning environment.
        asset_cfg: Robot asset config. The body_names must contain exactly two foot bodies
            (left first, right second).
        min_distance: Minimum desired lateral separation between feet [m]. Default 0.15 m.

    Returns:
        Per-environment penalty tensor of shape (N,).  Values are >= 0.
    """
    asset = env.scene[asset_cfg.name]
    # foot positions in world frame  [N, 2, 3]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # XY distance between the two feet
    dist = torch.norm(foot_pos[:, 0, :2] - foot_pos[:, 1, :2], dim=-1)  # [N]
    # penalise only when closer than min_distance
    return torch.clamp(min_distance - dist, min=0.0)


def feet_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    phase_freq: float = 1.5,
    stance_ratio: float = 0.55,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward based on matching foot contact pattern to a periodic phase oscillator.

    This function encourages a natural alternating bipedal gait by comparing actual
    foot contacts against a desired contact schedule derived from a phase oscillator.

    The two feet are driven in anti-phase (left: phi, right: phi + pi).  A foot is
    expected to be in *stance* (on the ground) when its phase falls in the first
    ``stance_ratio`` of the cycle, and in *swing* (in the air) for the rest.

    The reward is +1 only when *both* feet match their desired contact state simultaneously,
    and 0 otherwise (partial matches give no reward).  When the velocity command is below
    ``cmd_threshold`` (standing still), the desired pattern switches to *both feet in contact*
    so that the agent is actively rewarded for keeping both feet planted instead of lifting
    one to follow the oscillator.

    Args:
        env: The learning environment.
        sensor_cfg: Contact sensor configuration (must cover both foot bodies, left first).
        command_name: Name of the velocity command in the command manager.
        phase_freq: Frequency of the gait cycle in Hz.  Default 1.5 Hz (~normal walking).
        stance_ratio: Fraction of the cycle that each foot spends in stance (0–1).
            Default 0.55 (slightly more stance than swing, typical for walking).
        cmd_threshold: Speed below which the robot is considered "stopped" and both feet
            are expected to be in contact.

    Returns:
        Per-environment reward tensor of shape (N,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Current time within the episode for each environment  [N]
    t = env.episode_length_buf * env.step_dt

    # Phase angle in [0, 2*pi) for the LEFT foot
    phase_left = (2.0 * math.pi * phase_freq * t) % (2.0 * math.pi)   # [N]
    # RIGHT foot is half-cycle offset (anti-phase alternating gait)
    phase_right = (phase_left + math.pi) % (2.0 * math.pi)             # [N]

    # Desired contact: True when phase < stance_ratio * 2*pi  (stance phase)
    stance_threshold = 2.0 * math.pi * stance_ratio
    desired_stance_left = phase_left < stance_threshold    # [N]
    desired_stance_right = phase_right < stance_threshold  # [N]
    # Stack: shape [N, 2] — column 0 = left, column 1 = right
    desired_stance = torch.stack([desired_stance_left, desired_stance_right], dim=1)

    # When command speed is small, override desired pattern to "both feet in contact"
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_stopped = (cmd_speed <= cmd_threshold).unsqueeze(1)  # [N, 1]
    desired_stance = torch.where(is_stopped, torch.ones_like(desired_stance), desired_stance)

    # Actual contact: True when net contact force exceeds 1 N
    # net_forces_w_history: [N, history, num_bodies, 3]
    actual_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )  # [N, 2]

    # +1 only when both feet match their desired contact state simultaneously
    reward = torch.all(actual_contact == desired_stance, dim=1).float()  # [N]

    return reward

def track_lin_vel_xy_discrete_exp(
    env,
    std: float,
    command_name: str,
    stop_std: float = 0.1,
    stop_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    
    cmd = env.command_manager.get_command(command_name)[:, :2]
    actual = vel_yaw[:, :2]
    
    error = torch.sum(torch.square(cmd - actual), dim=1)
    is_zero = cmd.norm(dim=-1) < stop_threshold

    reward_moving = torch.exp(-error / std ** 2)
    reward_stop = torch.exp(-actual.norm(dim=-1) ** 2 / stop_std ** 2)

    return torch.where(is_zero, reward_stop, reward_moving)


def track_ang_vel_z_discrete_exp(
    env,
    command_name: str,
    std: float,
    stop_std: float = 0.1,
    stop_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    cmd = env.command_manager.get_command(command_name)[:, 2]
    actual = asset.data.root_ang_vel_w[:, 2]
    
    error = torch.square(cmd - actual)
    is_zero = cmd.abs() < stop_threshold

    reward_moving = torch.exp(-error / std ** 2)
    reward_stop = torch.exp(-actual ** 2 / stop_std ** 2)

    return torch.where(is_zero, reward_stop, reward_moving)

def joint_mirror_symmetry(
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """左右の股関節・膝角度が鏡像対称になるほど高報酬。
        
        対称ペア:
        Left_Hip_Pitch  ↔  Right_Hip_Pitch  (同符号)
        Left_Hip_Roll   ↔  Right_Hip_Roll   (逆符号)
        Left_Hip_Yaw    ↔  Right_Hip_Yaw    (逆符号)
        Left_Knee_Pitch ↔  Right_Knee_Pitch (同符号)
        """
        asset = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos

        def get_joint(name):
            idx = asset.find_joints(name)[0][0]
            return joint_pos[:, idx]

        l_hip_pitch  = get_joint("Left_Hip_Pitch")
        r_hip_pitch  = get_joint("Right_Hip_Pitch")
        l_hip_roll   = get_joint("Left_Hip_Roll")
        r_hip_roll   = get_joint("Right_Hip_Roll")
        l_hip_yaw    = get_joint("Left_Hip_Yaw")
        r_hip_yaw    = get_joint("Right_Hip_Yaw")
        l_knee       = get_joint("Left_Knee_Pitch")
        r_knee       = get_joint("Right_Knee_Pitch")

        # 同符号ペア: 差が0に近いほど対称
        # 逆符号ペア: 和が0に近いほど対称
        error = (
            torch.square(l_hip_pitch - r_hip_pitch) +
            torch.square(l_hip_roll  + r_hip_roll)  +
            torch.square(l_hip_yaw   + r_hip_yaw)   +
            torch.square(l_knee      - r_knee)
        )

        return torch.exp(-error / 0.1)

def get_feet_offset(env: ManagerBasedRLEnv, feet_distance_ref = 0.3) -> torch.Tensor:
    """Get the offset between left and right foot in the robot frame.

    This function computes the offset between the left and right foot positions in the robot frame.
    """
    asset = env.scene["robot"]
    _,_,base_yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    feet_x_offset = (
        torch.cos(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 0] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 0])
         - torch.sin(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 1] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 1])
    )
    feet_y_offset = (
        -torch.sin(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 0] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 0])
         + torch.cos(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 1] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 1])
    )

    feet_y_offset = feet_y_offset - feet_distance_ref
    return feet_x_offset, feet_y_offset

def feet_close_penalty(env: ManagerBasedRLEnv, feet_distance_threshold = 0.15) -> torch.Tensor:
    """Penalize feet being too close.

    This function penalizes the agent for having its feet too close together. The reward is computed as the
    distance between the feet positions.
    """
    _, feet_y_offset = get_feet_offset(env, 0.0) # そのままの値が欲しいのでrefは0にする

    return (feet_y_offset < feet_distance_threshold).float()


def feet_parallel_to_ground(env: ManagerBasedRLEnv, 
                            sigma: float = 0.3,
                            enable_potential: bool = True, 
                            discount_factor: float = 0.99) -> torch.Tensor:
    """Reward feet being parallel to the ground.

    This function rewards the agent for keeping its feet parallel to the ground.
    The reward is computed based on the pitch and roll angles of the feet.
    When the feet are perfectly parallel to the ground, pitch and roll should be close to zero.

    Args:
        env: Environment instance
        sigma: Exponential kernel width parameter (default: 0.3)

    Returns:
        torch.Tensor: Reward value for each environment
    """
    asset = env.scene["robot"]

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    # Get foot orientations (quaternions) in world frame
    left_foot_quat = asset.data.body_quat_w[:, left_foot_idx, :]
    right_foot_quat = asset.data.body_quat_w[:, right_foot_idx, :]

    # Convert quaternions to euler angles (roll, pitch, yaw)
    left_roll, left_pitch, _ = euler_xyz_from_quat(left_foot_quat)
    right_roll, right_pitch, _ = euler_xyz_from_quat(right_foot_quat)
    left_roll = wrap_to_pi(left_roll)
    right_roll = wrap_to_pi(right_roll)
    left_pitch = wrap_to_pi(left_pitch)
    right_pitch = wrap_to_pi(right_pitch)

    # Compute squared errors for pitch and roll
    # When feet are parallel to ground, both pitch and roll should be ~0
    left_foot_error = torch.square(left_pitch) + torch.square(left_roll)
    right_foot_error = torch.square(right_pitch) + torch.square(right_roll)

    # Total error for both feet
    total_error = left_foot_error + right_foot_error

    current_potential = torch.exp(-total_error / sigma)

    if enable_potential:
        buffer_key = "feet_parallel_to_ground_potential_prev"
        if not hasattr(env, "_custom_buffers"):
            env._custom_buffers = {}
        if buffer_key not in env._custom_buffers:
            env._custom_buffers[buffer_key] = current_potential.clone()
        prev_potential = env._custom_buffers[buffer_key]
        shaped_reward = discount_factor * current_potential - prev_potential
        reset_mask = env.reset_buf > 0
        shaped_reward = torch.where(reset_mask, torch.zeros_like(shaped_reward), shaped_reward)
        env._custom_buffers[buffer_key] = current_potential.clone()
    else:
        shaped_reward = current_potential

    return shaped_reward

def both_feet_not_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize when both feet are not in contact with the ground.

    This function penalizes the agent when both feet are not in contact with the ground.
    The reward is computed based on the contact forces of the feet. If both feet have low contact force, a penalty is applied.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w_history: [N, history, num_bodies, 3]
    foot_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)  # [N, history, 2]
    foot_in_contact = foot_contact_forces > 1.0  # Consider in contact if force > 1 N
    both_feet_not_in_contact = ~(foot_in_contact[:, -1, 0] | foot_in_contact[:, -1, 1])  # Check last timestep for both feet

    return both_feet_not_in_contact.float() * -1.0  # Penalty of -1 when both feet are not in contact

def foot_clearance_ji(env: ManagerBasedRLEnv, target_clearance: float = 0.06) -> torch.Tensor:
    asset = env.scene["robot"]

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    right_foot_xy_vel_sq = torch.sqrt(torch.norm(asset.data.body_lin_vel_w[:, right_foot_idx, :2], dim=1))
    left_foot_xy_vel_sq = torch.sqrt(torch.norm(asset.data.body_lin_vel_w[:, left_foot_idx, :2], dim=1))

    right_foot_height_err = torch.square(target_clearance - asset.data.body_pos_w[:, right_foot_idx, 2])
    left_foot_height_err = torch.square(target_clearance - asset.data.body_pos_w[:, left_foot_idx, 2])

    right_reward = right_foot_height_err * right_foot_xy_vel_sq
    left_reward = left_foot_height_err * left_foot_xy_vel_sq
    return right_reward + left_reward


def _expected_foot_height_bezier(phi: torch.Tensor, swing_height: float, stance_ratio: float = 0.5) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile.

    Args:
        phi: Gait phase in [-pi, pi].
        swing_height: Peak foot height during swing [m].
        stance_ratio: Fraction of the cycle spent in stance (foot on ground).
                      e.g. 0.6 means 60% stance, 40% swing.
    """

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)  # x ∈ [0, 1]

    # Normalize to swing phase: t ∈ [0, 1] only during swing; clamped to 0 during stance
    t = torch.clamp((x - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0)

    up   = cubic_bezier_interpolation(torch.zeros_like(t), torch.full_like(t, swing_height), 2 * t)
    down = cubic_bezier_interpolation(torch.full_like(t, swing_height), torch.zeros_like(t), 2 * t - 1)
    profile = torch.where(t <= 0.5, up, down)

    return torch.where(x <= stance_ratio, torch.zeros_like(x), profile)

def feet_height_bezier(env: ManagerBasedRLEnv, 
                        swing_height: float = 0.09, 
                        sigma: float = 0.008, 
                        phase_freq: float = 1.5,
                        stance_ratio: float = 0.55,
                        ground_height: float = 0.02) -> torch.Tensor:
    asset = env.scene["robot"]

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    # Compute foot heights
    right_foot_height = asset.data.body_pos_w[:, right_foot_idx, 2]
    left_foot_height = asset.data.body_pos_w[:, left_foot_idx, 2]

    # Current time within the episode for each environment  [N]
    t = env.episode_length_buf * env.step_dt

    # Phase angle in [0, 2*pi) for the LEFT foot
    phase_left = (2.0 * math.pi * phase_freq * t) % (2.0 * math.pi)   # [N]
    # RIGHT foot is half-cycle offset (anti-phase alternating gait)
    phase_right = (phase_left + math.pi) % (2.0 * math.pi)             # [N]

    rz_left = _expected_foot_height_bezier(phase_left, swing_height, stance_ratio)
    rz_right = _expected_foot_height_bezier(phase_right, swing_height, stance_ratio)

    # 歩行コマンドが非常に小さい時は目標高さは0にする（足を上げない歩行も許容する）
    command_lin_vel = env.command_manager.get_command("base_velocity")[:, :2]
    command_speed = torch.norm(command_lin_vel, dim=1)
    rz_left = torch.where(command_speed > 0.1, rz_left, torch.zeros_like(rz_left))
    rz_right = torch.where(command_speed > 0.1, rz_right, torch.zeros_like(rz_right))
    rz_left = torch.clamp(rz_left, min=ground_height)
    rz_right = torch.clamp(rz_right, min=ground_height)

    # Calculate height tracking errors
    error_left = torch.square(left_foot_height - rz_left)
    error_right = torch.square(right_foot_height - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right
    return torch.exp(-total_error / sigma)

def feet_swing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    phase_freq: float = 1.3,
    stance_ratio: float = 0.55,
    swing_period: float = 0.30,
    cmd_threshold: float = 0.1,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # gait_process: 0.0〜1.0
    t = env.episode_length_buf * env.step_dt
    gait_process = torch.remainder(phase_freq * t, 1.0)

    # feet_phase と同じ考え方:
    # stance_ratio までが接地期、その後がスイング期
    swing_center_left = stance_ratio + 0.5 * (1.0 - stance_ratio)
    swing_center_right = (swing_center_left + 0.5) % 1.0

    def phase_dist(a, b):
        d = torch.abs(a - b)
        return torch.minimum(d, 1.0 - d)

    left_swing = phase_dist(gait_process, swing_center_left) < 0.5 * swing_period
    right_swing = phase_dist(gait_process, swing_center_right) < 0.5 * swing_period

    actual_contact = (
        contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        > 1.0
    )  # [N, 2]

    left_not_contact = ~actual_contact[:, 0]
    right_not_contact = ~actual_contact[:, 1]

    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving = cmd_speed > cmd_threshold

    # 移動時：スイング位相で足が非接地なら報酬
    swing_reward = (
        (left_swing & left_not_contact).float()
        + (right_swing & right_not_contact).float()
    )

    # 停止時：両足が接地していたら報酬
    both_feet_contact = actual_contact[:, 0] & actual_contact[:, 1]
    stop_reward = both_feet_contact.float()

    # 移動時は swing_reward、停止時は stop_reward
    reward = torch.where(is_moving, swing_reward, stop_reward)

    return reward

#追加
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default pose only when command is near zero."""
    asset = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_stopped = cmd_speed < cmd_threshold

    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    penalty = torch.sum(joint_error, dim=1)

    return penalty * is_stopped.float()

__all__ = [
    "minimum_height",
    "track_lin_vel_xy_discrete_exp",
    "track_ang_vel_z_discrete_exp",
    "feet_distance",
    "feet_phase",
    "feet_close_penalty",
    "joint_mirror_symmetry",
    "feet_parallel_to_ground",
    "both_feet_not_in_contact",
    "foot_clearance_ji",
    "feet_height_bezier",
    "feet_swing",
    "stand_still_joint_deviation_l1",
]