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
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import  yaw_quat, euler_xyz_from_quat, wrap_to_pi
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from .data_logger import send_data_stream
from .observations import ball_vel as get_ball_vel

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
    # send_data_stream({"current_height": asset.data.root_pos_w[:, 2][0], "rewards": torch.square(asset.data.root_pos_w[:, 2][0] - adjusted_min_height)})
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
    one to follow the oscillator.  Additionally, while stopped, every lifted foot incurs a
    -1 penalty (片足浮き → -1, 両足浮き → -2) to actively suppress in-place stepping.

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
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_stopped = (cmd_speed < cmd_threshold).unsqueeze(1)  # [N, 1]
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

    # 停止時に足が浮いていたら、浮いている足の数に比例した負報酬を加える
    # (両足接地: 0, 片足浮き: -1, 両足浮き: -2 が reward に加算される)
    is_stopped_flat = is_stopped.squeeze(1)
    num_lifted = (~actual_contact).sum(dim=1).float()  # [N], in {0, 1, 2}
    stopped_penalty = torch.where(
        is_stopped_flat, -num_lifted, torch.zeros_like(num_lifted)
    )
    reward = reward + stopped_penalty

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

def foot_clearance_ji(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_clearance: float = 0.10,
    phase_freq: float = 1.5,
    stance_ratio: float = 0.55,
    cmd_threshold: float = 0.1,
    sigma: float = 0.03,
) -> torch.Tensor:
    """遊脚にのみ高さ追従報酬を与える関数

    遊脚判定は ``feet_phase`` と同じ規約（位相オシレータの desired stance、
    コマンド速度が ``cmd_threshold`` 未満の時は両足 stance 扱い）。
    """
    asset = env.scene["robot"]

    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    right_foot_height_err = torch.exp(-torch.square(target_clearance - asset.data.body_pos_w[:, right_foot_idx, 2]) / (sigma **2))
    left_foot_height_err = torch.exp(-torch.square(target_clearance - asset.data.body_pos_w[:, left_foot_idx, 2]) / (sigma **2))

    # feet_phase と同一の desired-stance 判定
    t = env.episode_length_buf * env.step_dt
    phase_left = (2.0 * math.pi * phase_freq * t) % (2.0 * math.pi)
    phase_right = (phase_left + math.pi) % (2.0 * math.pi)
    stance_threshold = 2.0 * math.pi * stance_ratio
    desired_stance_left = phase_left < stance_threshold
    desired_stance_right = phase_right < stance_threshold

    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_stopped = cmd_speed < cmd_threshold
    desired_stance_left = desired_stance_left | is_stopped
    desired_stance_right = desired_stance_right | is_stopped

    swing_left = (~desired_stance_left).float()
    swing_right = (~desired_stance_right).float()

    return right_foot_height_err * swing_right + left_foot_height_err * swing_left

def foot_clearance_ji_pen(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_clearance: float = 0.10,
    phase_freq: float = 1.5,
    stance_ratio: float = 0.55,
    cmd_threshold: float = 0.1,
    sigma: float = 0.03,
) -> torch.Tensor:
    """遊脚にのみ高さ追従報酬を与える関数

    遊脚判定は ``feet_phase`` と同じ規約（位相オシレータの desired stance、
    コマンド速度が ``cmd_threshold`` 未満の時は両足 stance 扱い）。
    """
    asset = env.scene["robot"]

    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    right_foot_vel = torch.norm(asset.data.body_lin_vel_w[:, right_foot_idx, :2], dim=1)    # xy速度が速いほどペナルティを大きくする
    left_foot_vel =  torch.norm(asset.data.body_lin_vel_w[:, left_foot_idx, :2], dim=1)
    right_foot_height_err = right_foot_vel * torch.square(target_clearance - asset.data.body_pos_w[:, right_foot_idx, 2]) 
    left_foot_height_err = left_foot_vel * torch.square(target_clearance - asset.data.body_pos_w[:, left_foot_idx, 2])

    # feet_phase と同一の desired-stance 判定
    t = env.episode_length_buf * env.step_dt
    phase_left = (2.0 * math.pi * phase_freq * t) % (2.0 * math.pi)
    phase_right = (phase_left + math.pi) % (2.0 * math.pi)
    stance_threshold = 2.0 * math.pi * stance_ratio
    desired_stance_left = phase_left < stance_threshold
    desired_stance_right = phase_right < stance_threshold

    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_stopped = cmd_speed < cmd_threshold
    desired_stance_left = desired_stance_left | is_stopped
    desired_stance_right = desired_stance_right | is_stopped

    swing_left = (~desired_stance_left).float()
    swing_right = (~desired_stance_right).float()

    return -(right_foot_height_err * swing_right + left_foot_height_err * swing_left)

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
    command_vel = env.command_manager.get_command("base_velocity")[:, :3]
    command_speed = torch.norm(command_vel, dim=1)
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

def feet_stride_length(
    env: ManagerBasedRLEnv,
    command_name: str,
    phase_freq: float = 1.5,
    stance_ratio: float = 0.55,
    sigma: float = 0.04,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """コマンド速度に応じた目標歩幅に追従するほど高い報酬。

    目標歩幅
        L = v_cmd_x * stance_ratio / phase_freq
    （1サイクル中に遊脚が前進する距離 = 接地中に後退する距離）

    左右足の前後方向距離 (x_L - x_R, base yaw frame) を位相に応じた目標値
        target_gap(φ_L) = L * cos(φ_L)
    と比較し、 exp(-error² / σ²) で報酬化する。
    （各足が ±L/2 の振幅で逆位相に振動する単純な正弦近似モデル。
      接地/遊脚比は左右ともに対称で、x方向のオフセットは0近傍と仮定。）

    Args:
        env: 学習環境
        command_name: 速度コマンド名
        phase_freq: 歩行周期の周波数 [Hz] (feet_phase と揃えること)
        stance_ratio: 接地時間の割合 (feet_phase と揃えること)
        sigma: 指数報酬のスケール [m]
        cmd_threshold: コマンドがこれ未満の時は目標歩幅を0にする
    """
    asset = env.scene["robot"]

    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    base_pos_w = asset.data.root_pos_w[:, :3]
    base_quat_yaw = yaw_quat(asset.data.root_quat_w)

    foot_rel_left = quat_apply_inverse(
        base_quat_yaw, asset.data.body_pos_w[:, left_foot_idx, :3] - base_pos_w
    )
    foot_rel_right = quat_apply_inverse(
        base_quat_yaw, asset.data.body_pos_w[:, right_foot_idx, :3] - base_pos_w
    )
    fwd_gap = foot_rel_left[:, 0] - foot_rel_right[:, 0]

    t = env.episode_length_buf * env.step_dt
    phase_left = (2.0 * math.pi * phase_freq * t) % (2.0 * math.pi)

    cmd = env.command_manager.get_command(command_name)
    L = cmd[:, 0] * stance_ratio / phase_freq

    cmd_speed = torch.norm(cmd[:, :3], dim=1)
    L = torch.where(cmd_speed < cmd_threshold, torch.zeros_like(L), L)

    target_gap = L * torch.cos(phase_left)
    error = torch.square(fwd_gap - target_gap)
    return torch.exp(-error / (sigma ** 2))

# ボールの速度方向がコマンド(目標位置)へ向く方向とどの程度一致するかを [0,1] で返す。
# ボールが (ほぼ) 停止している間は 0 になるよう速度でゲートする。
def ball_command_tracking(env: ManagerBasedRLEnv, command_name: str = "target_pos",
                          speed_gate: float = 0.3,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    # Why world-frame: get_command() の戻り値はロボット base frame の目標位置オフセットなので、
    # world frame の ball_vel と直接コサイン類似度を取ると frame が合わない。
    # ball の現在位置から目標位置への方向 (world) と ball_vel (world) を比較する。
    ball = env.scene[asset_cfg.name]
    ball_pos_w = ball.data.root_pos_w[:, :2]
    ball_vel_w = ball.data.root_com_vel_w[:, :2]

    target_pos_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
    desired_dir_w = target_pos_w - ball_pos_w

    cos_sim = torch.nn.functional.cosine_similarity(ball_vel_w, desired_dir_w, dim=1, eps=1e-6)
    alignment = (cos_sim + 1.0) / 2.0  # [-1,1] -> [0,1]

    # 速度ゲート: speed_gate [m/s] 未満では reward を線形に減衰させ、停止時は 0 にする
    ball_speed = torch.norm(ball_vel_w, dim=1)
    speed_factor = torch.clamp(ball_speed / speed_gate, 0.0, 1.0)
    return alignment * speed_factor

def ball_speed(env: ManagerBasedRLEnv, max_speed: float = 6.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    ball_vel = get_ball_vel(env)
    speed = torch.norm(ball_vel, dim=1)
    return torch.clamp(speed / max_speed, 0.0, 1.0)

# ボール速度の目標方向成分 (内積) を [0, max_speed] -> [0,1] に正規化して返す。
# Why: ball_speed と ball_command_tracking を別々に与えると「速いが方向無視」 or
# 「方向は正しいが遅い」の局所解に陥るため、両者を一本化する。
def ball_velocity_toward_target(
    env: ManagerBasedRLEnv,
    command_name: str = "target_pos",
    max_speed: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    ball = env.scene[asset_cfg.name]
    ball_pos_w = ball.data.root_pos_w[:, :2]
    ball_vel_w = ball.data.root_com_vel_w[:, :2]

    target_pos_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
    desired_dir_w = target_pos_w - ball_pos_w
    desired_unit = desired_dir_w / (torch.norm(desired_dir_w, dim=1, keepdim=True) + 1e-6)

    v_along = (ball_vel_w * desired_unit).sum(dim=1)
    v_along = torch.clamp(v_along, min=0.0)
    return torch.clamp(v_along / max_speed, 0.0, 1.0)

# ボールの飛距離に応じて報酬を与える。14mで最大1.0
def ball_distance(env: ManagerBasedRLEnv, max_distance: float = 14.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    ball_vel = get_ball_vel(env) # 積分してる事になったのでこちらの方が正しく距離である。
    ball_distance = torch.norm(ball_vel,dim=1,p=2) #l2ノルム計算してボールの距離を出す
    return torch.clamp(ball_distance / max_distance, 0.0, 1.0)

# ボールに触っていると報酬を得る。これを最後までオンにしていると蹴るよりも触ってしまうため、
# カリキュラムで途中で重みを途中で下げる等した方がよさそう。
def touch_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    contact_threshould = 0.1
    sensor_right = env.scene.sensors["contact_balls_right"]
    sensor_left = env.scene.sensors["contact_balls_left"]
    force_right = torch.norm(sensor_right.data.force_matrix_w[:,0,0],p=2,dim=1)
    force_left = torch.norm(sensor_left.data.force_matrix_w[:,0,0],p=2,dim=1)
    has_contact = torch.where(force_right > contact_threshould,1.0,0.0) + torch.where(force_left > contact_threshould,1.0,0.0)
    return torch.clip(has_contact,min=0.0,max=1.0)

def base_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ロボットの xy 平面の線速度の L2 二乗和をペナルティとして返す。"""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def base_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ロボットの z 軸 (yaw) 回転速度の L2 二乗をペナルティとして返す。"""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def force_touch_ball_downhalf(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_upper: float = 0.23,
    z_lower: float = 0.06,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """足が望ましい高さ範囲 [z_lower, z_upper] の外でボールに触れたらペナルティ。

    左右それぞれの足について「その足の高さが範囲外」かつ「その足がボールに接触」
    の場合にペナルティを与える。両足同時の場合も最大 1.0 にクリップする。
    """
    robot = env.scene[asset_cfg.name]
    left_foot_idx = robot.find_bodies("left_foot_link")[0][0]
    right_foot_idx = robot.find_bodies("right_foot_link")[0][0]
    left_z = robot.data.body_link_pos_w[:, left_foot_idx, 2]
    right_z = robot.data.body_link_pos_w[:, right_foot_idx, 2]

    out_range_left = (left_z > z_upper) | (left_z < z_lower)
    out_range_right = (right_z > z_upper) | (right_z < z_lower)

    sensor_right = env.scene.sensors["contact_balls_right"]
    sensor_left = env.scene.sensors["contact_balls_left"]
    force_right = torch.norm(sensor_right.data.force_matrix_w[:, 0, 0], p=2, dim=1)
    force_left = torch.norm(sensor_left.data.force_matrix_w[:, 0, 0], p=2, dim=1)
    touch_left = force_left > contact_threshold
    touch_right = force_right > contact_threshold

    penalty_left = (touch_left & out_range_left).float()
    penalty_right = (touch_right & out_range_right).float()
    return torch.clamp(penalty_left + penalty_right, max=1.0)

def action_smoothness_l2(env):
    # これはcassieのcausal transformer論文から取ってきた
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    if not hasattr(env, "_prev_prev_action"):
        env._prev_prev_action = torch.zeros_like(a)
    diff1 = torch.square(a - a_prev)
    diff2 = torch.square(a - 2.0 * a_prev + env._prev_prev_action)
    env._prev_prev_action = a_prev.clone()
    return torch.sum((diff1 + diff2), dim=1)

def fix_stance_foot_pos(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg ) -> torch.tensor:
    robot = env.scene[asset_cfg.name]
    stance_foot_vel = robot.data.body_com_vel_w[:,asset_cfg.body_ids,:2].squeeze(1) # x,y
    stance_foot_vel_sum = torch.linalg.norm(stance_foot_vel,dim=1)
    return stance_foot_vel_sum.squeeze()


# 蹴り足の z 高さが threshold を超えた量の合計を返す (penalty 用 / weight を負にする)。
# Why threshold: キック動作中の一時的な持ち上げは許容したいので、地面付近はペナルティ 0 にする。
# 蹴った後に足を地面まで戻させるための報酬。
def foot_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.08,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    excess = (foot_z - threshold).clamp(min=0.0)
    return excess.sum(dim=1)

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
    "feet_stride_length",
]