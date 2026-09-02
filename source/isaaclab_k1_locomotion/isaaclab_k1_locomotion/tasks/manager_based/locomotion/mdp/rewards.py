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
from .events import get_gait_phase, get_phase_freq

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
    adaptive: bool = False,
    l_fwd: float = 0.31,
    l_lat: float = 0.11,
    l_back: float = 0.0,
    f_min: float = 1.2,
    f_max: float = 5.0,
    dr_base: float = 1.6,
    use_actual_speed: bool = False,
    cmd_gain: float = 1.0,
    vel_lag_s: float = 0.0,
    vel_noise_std: float = 0.0,
    lateral_phase_flip: bool = False,
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

    # Phase angle in [0, 2*pi) for the LEFT foot
    # adaptive=True なら速度指令から周波数を決める (:func:`~.events.adaptive_phase_freq`)
    phase_left = get_gait_phase(
        env, phase_freq, adaptive=adaptive, command_name=command_name,
        l_fwd=l_fwd, l_lat=l_lat, l_back=l_back, f_min=f_min, f_max=f_max, dr_base=dr_base,
        use_actual_speed=use_actual_speed, cmd_gain=cmd_gain,
        vel_lag_s=vel_lag_s, vel_noise_std=vel_noise_std,
        lateral_phase_flip=lateral_phase_flip,
    ) % (2.0 * math.pi)   # [N]
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
    adaptive: bool = False,
    l_fwd: float = 0.31,
    l_lat: float = 0.11,
    l_back: float = 0.0,
    f_min: float = 1.2,
    f_max: float = 5.0,
    dr_base: float = 1.6,
    use_actual_speed: bool = False,
    cmd_gain: float = 1.0,
    vel_lag_s: float = 0.0,
    vel_noise_std: float = 0.0,
    lateral_phase_flip: bool = False,
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
    # ☠ 2026-08-21 まで、本関数だけ `get_phase_freq` を経由せず引数の phase_freq を
    #   **直接**使っていた。位相 DR (randomize_phase_freq、±0.05Hz) が掛かった env では
    #   feet_phase / phase_obs と位相がズレ、エピソード後半では逆位相になりうる。
    #   その状態で「遊脚の高さ」を要求すると **接地している足に高さを要求する**ことに
    #   なり、それを満たす唯一の手段が跳躍。「foot_clearance の weight を上げるたび
    #   跳躍に退行した」履歴の一因と見て、他項と同じ get_gait_phase に統一した。
    phase_left = get_gait_phase(
        env, phase_freq, adaptive=adaptive, command_name=command_name,
        l_fwd=l_fwd, l_lat=l_lat, l_back=l_back, f_min=f_min, f_max=f_max, dr_base=dr_base,
        use_actual_speed=use_actual_speed, cmd_gain=cmd_gain,
        vel_lag_s=vel_lag_s, vel_noise_std=vel_noise_std,
        lateral_phase_flip=lateral_phase_flip,
    ) % (2.0 * math.pi)
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
    adaptive: bool = False,
    l_fwd: float = 0.31,
    l_lat: float = 0.11,
    l_back: float = 0.0,
    f_min: float = 1.2,
    f_max: float = 5.0,
    dr_base: float = 1.6,
    use_actual_speed: bool = False,
    cmd_gain: float = 1.0,
    vel_lag_s: float = 0.0,
    vel_noise_std: float = 0.0,
    lateral_phase_flip: bool = False,
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
    phase_left = get_gait_phase(
        env, phase_freq, adaptive=adaptive, command_name=command_name,
        l_fwd=l_fwd, l_lat=l_lat, l_back=l_back, f_min=f_min, f_max=f_max, dr_base=dr_base,
        use_actual_speed=use_actual_speed, cmd_gain=cmd_gain,
        vel_lag_s=vel_lag_s, vel_noise_std=vel_noise_std,
        lateral_phase_flip=lateral_phase_flip,
    ) % (2.0 * math.pi)
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


def ball_velocity_along_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    speed_gate: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールのワールド座標 xy 速度の「方向」がコマンドのキック方向 (ワールド) と
    どれだけ揃っているかだけを評価する。速度の大きさは見ない。

    ``kick_direction`` コマンドは既に単位ベクトルを返す。ボール速度方向との
    コサイン類似度 [-1, 1] を **そのまま** 返す (対称評価)。速度の大きさ自体は
    別途 ``ball_speed`` 報酬で評価する。

    Why 対称: [0, 1] にマップすると逆方向 (cos=-1) が「報酬ゼロ」になり罰せられず、
    無方向に報酬を出す ``ball_speed`` と組み合わさると「方向を無視してとにかく蹴る」
    局所解に落ちる。生の cos を使えば逆方向は負の報酬となり、目標と反対に蹴る挙動を
    積極的にペナルティ化できる。

    ただし停止時 (速度 ~0) に報酬が残ると悪影響なので、`speed_gate` [m/s] 未満では
    線形に減衰させて停止時は 0 にする。`speed_gate` を小さく取ることで、動き出せば
    すぐにゲートが 1 に飽和し「方向のみ」の評価という性質は保たれる。
    """
    ball = env.scene[asset_cfg.name]
    ball_vel_w = ball.data.root_com_vel_w[:, :2]
    kick_dir_w = env.command_manager.get_term(command_name).command  # (N, 2)
    cos_sim = torch.nn.functional.cosine_similarity(ball_vel_w, kick_dir_w, dim=1, eps=1e-6)

    ball_speed = torch.norm(ball_vel_w, dim=1)
    speed_factor = torch.clamp(ball_speed / speed_gate, 0.0, 1.0)
    return cos_sim * speed_factor


def ball_speed(
    env: ManagerBasedRLEnv,
    max_speed: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールのワールド座標 xy 速度のノルムを `max_speed` で割って [0, 1] にクランプ。"""
    ball = env.scene[asset_cfg.name]
    ball_vel_w = ball.data.root_com_vel_w[:, :2]
    speed = torch.norm(ball_vel_w, dim=1)
    return torch.clamp(speed / max_speed, 0.0, 1.0)


def ball_speed_along_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    max_speed: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボール速度の「キック方向 (ワールド単位ベクトル) への射影成分」を [0, 1] に正規化して返す。

    ``v_along = dot(ball_vel_w, kick_dir_w)`` を ``max_speed`` で割り、負 (逆方向) は 0 に
    クランプする。方向に依らず大きさだけを見る :func:`ball_speed` と違い、「速く **かつ
    正しい方向に**」蹴ったときだけ報酬が出るので、方向を無視してとにかく蹴る局所解を
    資金提供しない。:func:`ball_velocity_along_kick` (cos のみ・大きさ無視) と組み合わせると
    「方向精度」と「速度の大きさ」を両取りできる。
    """
    ball = env.scene[asset_cfg.name]
    ball_vel_w = ball.data.root_com_vel_w[:, :2]
    kick_dir_w = env.command_manager.get_term(command_name).command  # (N, 2) 単位ベクトル
    v_along = (ball_vel_w * kick_dir_w).sum(dim=1)
    v_along = torch.clamp(v_along, min=0.0)
    return torch.clamp(v_along / max_speed, 0.0, 1.0)


def _approach_target_w(
    env: ManagerBasedRLEnv,
    ball_pos_w: torch.Tensor,
    command_name: str | None,
    behind_offset: float,
) -> torch.Tensor:
    """接近報酬のターゲット点 (ワールド xy) を返す。

    ``command_name`` が None または ``behind_offset`` が 0 ならボール中心をそのまま返す。
    指定された場合は「キック方向から見てボールの後ろ側」に ``behind_offset`` [m] ずらした
    staging 地点 ``p = ball_pos - kick_dir * behind_offset`` を返す。ここを目標に接近させると
    「ボールに寄る」と「正しい側 (裏) に回り込む」が一本化され、最短直線接近と裏取りの
    対立が消える。
    """
    if command_name is None or behind_offset == 0.0:
        return ball_pos_w
    kick_dir_w = env.command_manager.get_term(command_name).command  # (N, 2) 単位ベクトル
    return ball_pos_w - kick_dir_w * behind_offset


def com_jerk_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ロボットの root body COM の線加速度の jerk (時間微分) の L2 二乗をペナルティとして返す。

    jerk_t ≈ (acc_t - acc_{t-1}) / dt
    前ステップの加速度は `env._prev_com_lin_acc_w` に保持する
    (`action_smoothness_l2` と同じパターン)。エピソードリセット直後は
    勾配を 0 にしたいので、`episode_length_buf` が小さいときはペナルティを 0 にする。
    """
    robot = env.scene[asset_cfg.name]
    # body_com_lin_acc_w: (N, num_bodies, 3) ; 0番目が root body
    acc = robot.data.body_com_lin_acc_w[:, 0, :]  # (N, 3)
    if not hasattr(env, "_prev_com_lin_acc_w") or env._prev_com_lin_acc_w.shape != acc.shape:
        env._prev_com_lin_acc_w = acc.clone()
    prev_acc = env._prev_com_lin_acc_w
    dt = env.step_dt
    jerk = (acc - prev_acc) / max(dt, 1e-6)
    env._prev_com_lin_acc_w = acc.clone()
    penalty = torch.sum(torch.square(jerk), dim=1)
    # リセット直後 (前回 acc が前エピソードのもの) は無効化
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(penalty), penalty)


def robot_velocity_toward_ball(
    env: ManagerBasedRLEnv,
    max_speed: float = 1.0,
    min_distance: float = 0.05,
    command_name: str | None = None,
    behind_offset: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットが接近ターゲットに向かって進んでいる成分を [0, 1] で返す shaping 報酬。

    ワールド xy で、ロボット→ターゲット方向の単位ベクトルにロボットの線速度を射影する。
    ターゲットに十分近づいたとき (`min_distance` 未満) は方向が定義しにくいので 0 を返す。

    ``command_name`` / ``behind_offset`` を指定すると、ターゲットがボール中心ではなく
    「キック方向から見てボールの後ろ側」に ``behind_offset`` [m] ずらした staging 地点になる。
    こうすると「ボールに寄る」と「正しい側へ回り込む」が同じ勾配になり、最短直線接近で
    間違った方向に蹴ってしまう挙動を抑えられる。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    target = _approach_target_w(env, ball.data.root_pos_w[:, :2], command_name, behind_offset)
    to_target = target - robot.data.root_pos_w[:, :2]
    dist = torch.norm(to_target, dim=1, keepdim=True)
    direction = to_target / (dist + 1e-6)
    robot_vel_w = robot.data.root_lin_vel_w[:, :2]
    v_along = (robot_vel_w * direction).sum(dim=1)
    v_along = torch.clamp(v_along, min=0.0)
    reward = torch.clamp(v_along / max_speed, 0.0, 1.0)
    # ターゲットに十分近づいたら shaping 不要にする。
    near = (dist.squeeze(-1) < min_distance)
    reward = torch.where(near, torch.zeros_like(reward), reward)
    return reward


def approach_ball_progress(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    behind_offset: float = 0.0,
    max_progress_per_step: float | None = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボット→接近ターゲット距離の 1 ステップ減少量 (m) を返すポテンシャル形式の接近報酬。

    ``progress = dist_{t-1} - dist_t``。近づくと正、離れると負、止まると 0。
    速度を直接見る :func:`robot_velocity_toward_ball` (上限クランプ・近接で 0) と違い、
    上限がなく常時密な勾配がかかるため「動かない」局所解に陥りにくい。

    ``command_name`` / ``behind_offset`` を指定すると、ターゲットがボール中心ではなく
    「キック方向の後ろ側」に ``behind_offset`` [m] ずらした staging 地点になる。
    ポテンシャル差なのでエピソード総和は ``dist_0 - dist_final`` に telescope する。

    リセット直後 (``episode_length_buf < 2``) は距離が不連続に飛ぶので 0 を返す。
    staging 地点はキック方向に依存して動くので、キック方向の再サンプリング時に
    ターゲットが最大 ``2*behind_offset`` 飛び、その 1 ステップだけ偽の巨大 progress が
    乗る。``max_progress_per_step`` (既定 0.1m) で |progress| をクランプしてこのスパイクを
    潰す。1 ステップの物理的な距離変化は高々 ``robot_speed * dt`` (~0.04m) なので、
    0.1m クランプは通常移動には影響しない。``None`` でクランプ無効。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    target = _approach_target_w(env, ball.data.root_pos_w[:, :2], command_name, behind_offset)
    to_target = target - robot.data.root_pos_w[:, :2]
    dist = torch.norm(to_target, dim=1)
    prev = getattr(env, "_prev_ball_distance", None)
    if prev is None or prev.shape != dist.shape:
        env._prev_ball_distance = dist.clone()
        return torch.zeros_like(dist)
    progress = prev - dist
    env._prev_ball_distance = dist.clone()
    if max_progress_per_step is not None:
        progress = torch.clamp(progress, -max_progress_per_step, max_progress_per_step)
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(progress), progress)


def robot_facing_ball(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットの Trunk 正面 (base +x) がボール方向を向いているほど大きい [0, 1] の報酬。

    ボール位置をロボットの base yaw frame に変換し、xy 単位ベクトルの x 成分
    (= cos(θ), θ は Trunk 正面とボール方向の角度) を返す。
    cos が負 (ボールが背後) のときは 0 にクランプ。
    `min_distance` 未満では方向が不安定になるので 0 を返す。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    offset_w = ball.data.root_pos_w - robot.data.root_pos_w
    offset_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), offset_w)
    dist_xy = torch.norm(offset_b[:, :2], dim=1)
    cos_theta = offset_b[:, 0] / (dist_xy + 1e-6)
    reward = torch.clamp(cos_theta, min=0.0, max=1.0)
    near = dist_xy < min_distance
    return torch.where(near, torch.zeros_like(reward), reward)


def robot_behind_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    min_distance: float = 0.15,
    engage_distance: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットが「キック方向から見てボールの後ろ側」かつ「ボールに近い」ほど大きい
    [0, 1] の報酬。

    ロボット→ボール方向の単位ベクトルと ``kick_direction`` (ワールド単位ベクトル) の
    内積を取る。+1 ならロボットはボールの真後ろ (前進すればボールをキック方向へ押せる
    理想配置)、-1 なら逆側 (前進するとボールを目標と反対に蹴ってしまう配置)。

    Why 密な非ポテンシャル形式: 良い配置に早く着くほど報酬を受け取れるステップ数が
    増えるので、累積報酬として「なるべく早く位置合わせする」挙動が自然に誘導される。

    Why 近接ゲート (``engage_distance``): 距離に依存しない素の密報酬だと、ボールから
    遠い後ろ側で「居座る」だけで永久に報酬を稼げてしまい (farmable)、接近してキック
    する危険・労力を避ける局所最適に落ちる。``proximity = clamp(1 - dist/engage)`` を
    掛けて、ボールに近いほど報酬が大きくなるようにすることで「後ろ側 かつ 接近」を
    要求し、回り込んだ後そのまま接近 → 接触 → キックへ繋がるようにする。

    逆側 (内積 < 0) は 0 にクランプして「ゼロ報酬」とする (積極的な罰はしない)。
    ``min_distance`` 未満では方向が不安定になるので 0 を返す。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    to_ball = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.norm(to_ball, dim=1)
    to_ball_unit = to_ball / (dist.unsqueeze(-1) + 1e-6)
    kick_dir = env.command_manager.get_term(command_name).command  # (N, 2) 単位ベクトル
    align = (to_ball_unit * kick_dir).sum(dim=1)  # [-1, 1]
    # ボールに近いほど 1、engage_distance 以遠で 0 になる近接係数。
    proximity = torch.clamp(1.0 - dist / engage_distance, 0.0, 1.0)
    reward = torch.clamp(align, min=0.0, max=1.0) * proximity
    near = dist < min_distance
    return torch.where(near, torch.zeros_like(reward), reward)


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

def feet_landing_impact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """着地時の衝撃力に応じてペナルティを与える報酬関数 (weight < 0 で使う想定)。

    前ステップで airborne (|F| < threshold) だった足が、現ステップで接地状態 (|F| >= threshold)
    になった瞬間を「着地イベント」とみなし、その時の接地力ノルムをそのまま値として返す。
    着地以外 (定常接地中 / 空中) は 0。

    Args:
        env: 学習環境。
        sensor_cfg: 足の Contact sensor (両足の foot body を含める)。
        contact_threshold: 接地判定の力ノルム閾値 [N]。

    Returns:
        環境ごとのペナルティ値 [N], 両足の衝撃力ノルム [N] の合計。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w_history: [N, history, num_bodies, 3]; index 0 = 最新, 1 = 前ステップ
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)  # [N, history, num_bodies]

    current = force_mag[:, 0]
    previous = force_mag[:, 1]

    landing = (current >= contact_threshold) & (previous < contact_threshold)
    impact_force = torch.where(landing, current, torch.zeros_like(current))

    return impact_force.sum(dim=1)


def feet_landing_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    contact_threshold: float = 1.0,
    vertical_only: bool = True,
) -> torch.Tensor:
    """着地瞬間の足の速度に応じてペナルティを与える報酬関数 (weight < 0 で使う想定)。

    `feet_landing_impact` が「着地時の接地力」を罰するのに対し、本関数は「着地時の足の速度」
    を罰する。前ステップで airborne (|F| < threshold) だった足が、現ステップで接地状態
    (|F| >= threshold) になった瞬間を「着地イベント」とみなし、その時の足の速度を値として返す。
    速度が大きいほど地面を強く踏みつけている (=硬い着地) ことを意味するため、これを抑制すると
    柔らかく接地する歩容を促せる。着地以外 (定常接地中 / 空中) は 0。

    Args:
        env: 学習環境。
        sensor_cfg: 足の Contact sensor (両足の foot body を含める)。着地イベント判定に使う。
        asset_cfg: 足の速度を取る Articulation。sensor_cfg と同じ足 body を同じ順序で指す必要がある。
        contact_threshold: 接地判定の力ノルム閾値 [N]。
        vertical_only: True のとき鉛直方向の下向き速度 |v_z| のみを対象にする (踏みつけの直接原因)。
            False のときは足速度の 3D ノルムを使う (水平方向の擦り・滑りも抑制)。

    Returns:
        環境ごとのペナルティ値 [m/s], 着地した足の速度の合計。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # net_forces_w_history: [N, history, num_bodies, 3]; index 0 = 最新, 1 = 前ステップ
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)  # [N, history, num_feet]
    current = force_mag[:, 0]
    previous = force_mag[:, 1]
    landing = (current >= contact_threshold) & (previous < contact_threshold)  # [N, num_feet]

    # 足の速度 (world frame)。body_lin_vel_w は現ステップ値のみ取得可能なため着地直前を最新値で近似。
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # [N, num_feet, 3]
    if vertical_only:
        vel_mag = foot_vel[:, :, 2].abs()  # |v_z|
    else:
        vel_mag = foot_vel.norm(dim=-1)

    impact_vel = torch.where(landing, vel_mag, torch.zeros_like(vel_mag))
    return impact_vel.sum(dim=1)


def feet_heel_strike(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
    contact_threshold: float = 1.0,
    target_pitch: float = 0.2,
    std: float = 0.15,
    pitch_sign: float = 1.0,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """着地の瞬間に「かかとから接地する」姿勢 (つま先上げ=背屈) を促す報酬関数 (weight > 0 で使う想定)。

    前ステップで airborne (|F| < threshold) だった足が、現ステップで接地状態 (|F| >= threshold) に
    なった瞬間を「着地イベント」とみなす。その瞬間の足のピッチ角が、かかと側が下がった姿勢
    (``target_pitch``) に近いほど大きい報酬を与える。これにより、足裏ベタ着き/つま先着地ではなく、
    かかとから接地する歩容を促す。着地以外 (定常接地中 / 空中) は 0。

    立ち止まり時 (コマンド速度 < ``cmd_threshold``) は、接地閾値付近のノイズで擬似的な着地イベントが
    頻発し、足を ``target_pitch`` 方向に傾ける勾配が効いてしまう (足の傾き/微振動の原因)。これを防ぐため、
    停止時は報酬を 0 にゲートする (``feet_phase`` と同じ停止判定規約)。

    Note:
        足リンクのローカル座標系の取り方によってピッチの符号は変わりうる。学習しても逆 (つま先着地)
        が促進されてしまう場合は ``pitch_sign`` を ``-1.0`` に反転すること。デバッグ時は
        ``send_data_stream`` で着地時の ``pitch`` 値を確認すると符号を特定しやすい。

    Args:
        env: 学習環境。
        sensor_cfg: 足の Contact sensor (両足の foot body を含める)。着地イベント判定に使う。
        asset_cfg: 足の姿勢を取る Articulation。sensor_cfg と同じ足 body を同じ順序で指すこと。
        contact_threshold: 接地判定の力ノルム閾値 [N]。
        target_pitch: 着地時に狙うかかと下がりピッチ角 [rad] (toe-up / 背屈方向を正とする)。
        std: 報酬カーネル幅。``target_pitch`` からの許容ズレの広さ。小さいほどシビアになる。
        pitch_sign: 計測ピッチの符号補正 (+1.0 / -1.0)。toe-up が正になるよう合わせる。
        command_name: 速度コマンド名。停止判定に使う。
        cmd_threshold: 停止判定の速度ノルム閾値 [m/s]。これ未満は立ち止まりとみなし報酬を 0 にする。

    Returns:
        環境ごとの報酬値 (着地イベント時のみ非0、両足合計)。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # net_forces_w_history: [N, history, num_bodies, 3]; index 0 = 最新, 1 = 前ステップ
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)  # [N, history, num_feet]
    current = force_mag[:, 0]
    previous = force_mag[:, 1]
    landing = (current >= contact_threshold) & (previous < contact_threshold)  # [N, num_feet]

    # 足のピッチ角 (world frame)。euler_xyz_from_quat は [*, 4] を想定するため平坦化して処理。
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N, num_feet, 4]
    num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]
    _, pitch, _ = euler_xyz_from_quat(foot_quat.reshape(-1, 4))
    pitch = wrap_to_pi(pitch).reshape(num_envs, num_feet)  # [N, num_feet]
    pitch = pitch_sign * pitch

    # かかと下がり (target_pitch) に近い着地ほど高報酬
    heel_reward = torch.exp(-torch.square(pitch - target_pitch) / (std ** 2))
    heel_reward = torch.where(landing, heel_reward, torch.zeros_like(heel_reward))
    reward = heel_reward.sum(dim=1)  # [N]

    # 立ち止まり時 (cmd 速度 < cmd_threshold) は接地ノイズによる擬似着地で足が傾くのを防ぐため 0 にする
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)  # [N]
    is_moving = (cmd_speed >= cmd_threshold).float()
    return reward * is_moving


def _stand_still_boost(
    env: ManagerBasedRLEnv,
    command_name: str,
    cmd_threshold: float,
    lin_vel_threshold: float,
    ang_vel_threshold: float,
    scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """立ち止まり「かつ実際に静止している」ときだけ penalty 倍率を ``scale`` にするゲート係数を返す。

    cmd 速度が小さい (停止指令) ことに加えて base の実速度も小さい (= 外乱で動かされていない)
    ことを条件にする。これにより、push を受けて base 速度が跳ねた瞬間は倍率が 1.0 に戻り、
    立ち直り (push recovery) のための素早い動作を過度に罰さない。

    Args:
        env: 学習環境。
        command_name: 速度コマンド名。停止指令の判定に使う。
        cmd_threshold: 停止指令とみなす cmd 速度ノルム閾値 [m/s]。
        lin_vel_threshold: 静止とみなす base xy 線速度ノルム閾値 [m/s]。これを超えたら倍率 1.0。
        ang_vel_threshold: 静止とみなす base yaw 角速度の絶対値閾値 [rad/s]。
        scale: 立ち止まり静止時にかける倍率 (例: 2.0)。
        asset_cfg: base 速度を取る Articulation。

    Returns:
        各環境の倍率 [N], 立ち止まり静止時は ``scale``、それ以外は 1.0。
    """
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_stopped = cmd_speed < cmd_threshold

    asset = env.scene[asset_cfg.name]
    lin_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    ang_speed = asset.data.root_ang_vel_b[:, 2].abs()
    is_still = (lin_speed < lin_vel_threshold) & (ang_speed < ang_vel_threshold)

    boost = is_stopped & is_still
    return torch.where(boost, torch.full_like(cmd_speed, scale), torch.ones_like(cmd_speed))


def action_smoothness_l2(
    env,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    stand_still_scale: float = 1.0,
    lin_vel_threshold: float = 0.2,
    ang_vel_threshold: float = 0.2,
):
    # これはcassieのcausal transformer論文から取ってきた
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    if not hasattr(env, "_prev_prev_action"):
        env._prev_prev_action = torch.zeros_like(a)
    diff1 = torch.square(a - a_prev)
    diff2 = torch.square(a - 2.0 * a_prev + env._prev_prev_action)
    env._prev_prev_action = a_prev.clone()
    penalty = torch.sum((diff1 + diff2), dim=1)
    # 立ち止まり静止時のみ倍率を上げて、recurrent ポリシー特有の振動を抑える。
    if stand_still_scale != 1.0:
        penalty = penalty * _stand_still_boost(
            env, command_name, cmd_threshold, lin_vel_threshold, ang_vel_threshold, stand_still_scale
        )
    return penalty


def action_rate_l2(
    env,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    stand_still_scale: float = 1.0,
    lin_vel_threshold: float = 0.2,
    ang_vel_threshold: float = 0.2,
):
    """標準 action_rate_l2 (``||a_t - a_{t-1}||²``) に立ち止まり静止時の倍率ゲートを加えたもの。

    停止指令かつ base が実際に静止しているときだけ penalty を ``stand_still_scale`` 倍する。
    ゲートの詳細は :func:`_stand_still_boost` 参照。
    """
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    penalty = torch.sum(torch.square(a - a_prev), dim=1)
    if stand_still_scale != 1.0:
        penalty = penalty * _stand_still_boost(
            env, command_name, cmd_threshold, lin_vel_threshold, ang_vel_threshold, stand_still_scale
        )
    return penalty


def high_action_smoothness_l2(env):
    """上位ポリシーの高レベル action (歩行コマンド 3D) の平滑性ペナルティ。

    既存の :func:`action_smoothness_l2` と同じ式
    ``||a - a_prev||² + ||a - 2*a_prev + a_prev_prev||²`` を上位 action に対して計算する。
    対象は ``env.action_manager.action`` (= frozen の 22D 関節指令) ではなく、
    ``HierarchicalVecEnvWrapper`` が毎ステップ ``env._prev_high_action`` (clipped 3D) に
    書き込む値。

    リセット直後はペナルティを 0 にする (``episode_length_buf < 2``)。
    """
    a = getattr(env, "_prev_high_action", None)
    if a is None:
        return torch.zeros(env.num_envs, device=env.device)
    # 履歴バッファ。初回呼び出し or 形状不整合なら 0 で再初期化。
    if (not hasattr(env, "_high_action_prev")) or env._high_action_prev.shape != a.shape:
        env._high_action_prev = torch.zeros_like(a)
        env._high_action_prev_prev = torch.zeros_like(a)
    a_prev = env._high_action_prev
    a_prev_prev = env._high_action_prev_prev
    diff1 = torch.square(a - a_prev)
    diff2 = torch.square(a - 2.0 * a_prev + a_prev_prev)
    penalty = torch.sum(diff1 + diff2, dim=1)
    # 履歴更新 (次ステップで使う)
    env._high_action_prev_prev = a_prev.clone()
    env._high_action_prev = a.clone()
    # リセット直後は履歴が前エピソードのものなのでペナルティを無効化
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(penalty), penalty)


def high_action_rate_l2(env):
    """上位ポリシーの高レベル action (歩行コマンド 3D) の action rate ペナルティ。

    1 ステップ差分の L2 二乗: ``||a_t - a_{t-1}||²``。
    対象は ``env.action_manager.action`` (= frozen の 22D 関節指令) ではなく、
    ``HierarchicalVecEnvWrapper`` が ``env._prev_high_action`` に書き込む 3D 歩行コマンド。

    バッファは ``high_action_smoothness_l2`` とは独立 (``_high_action_prev_for_rate``)。
    呼び出し順序に依存しないようにするための分離。
    リセット直後 (``episode_length_buf < 2``) は 0 を返す。
    """
    a = getattr(env, "_prev_high_action", None)
    if a is None:
        return torch.zeros(env.num_envs, device=env.device)
    if (not hasattr(env, "_high_action_prev_for_rate")) or env._high_action_prev_for_rate.shape != a.shape:
        env._high_action_prev_for_rate = torch.zeros_like(a)
    a_prev = env._high_action_prev_for_rate
    penalty = torch.sum(torch.square(a - a_prev), dim=1)
    env._high_action_prev_for_rate = a.clone()
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(penalty), penalty)


def high_action_xy_coactivation(env):
    """上位 action の並進成分 vx と vy が同時に大きいほど大きいペナルティ。

    値は ``|vx| * |vy|``。片軸のみ (x+theta / y+theta) なら 0、小さい混合は
    積なので二次的に小さく実質ノーペナルティ。vx・vy が両方大きい「全部盛り」
    だけを狙い撃ちで抑えることで、x+theta か y+theta のどちらかのモードに寄せる。
    theta(vyaw) は対象外で常に自由。

    対象は ``HierarchicalVecEnvWrapper`` が ``env._prev_high_action`` に書き込む
    clipped な 3D 歩行コマンド (vx, vy, vyaw)。
    """
    a = getattr(env, "_prev_high_action", None)
    if a is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.abs(a[:, 0]) * torch.abs(a[:, 1])


def fix_stance_foot_pos(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg ) -> torch.tensor:
    robot = env.scene[asset_cfg.name]
    stance_foot_vel = robot.data.body_com_vel_w[:,asset_cfg.body_ids,:2].squeeze(1) # x,y
    stance_foot_vel_sum = torch.linalg.norm(stance_foot_vel,dim=1)
    return stance_foot_vel_sum.squeeze()


def com_jerk_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """全身重心 (CoM) 位置の jerk の二乗ノルムを罰する報酬関数 (weight < 0 で使う想定)。

    jerk (躍度) は加速度の時間微分。CoM 速度 v の二階差分で近似する::

        jerk_t ≈ (v_t - 2 v_{t-1} + v_{t-2}) / dt²        [m/s³]

    重心の動きの急変 (カクつき) を罰することで、滑らかな体重移動を促す。
    全身 CoM 速度は各 body の CoM 速度を質量で加重平均して求める::

        v_com = Σ_i m_i v_i / Σ_i m_i        (world frame)

    履歴 (v_{t-1}, v_{t-2}) は env 上のバッファに保持する。リセット直後は前エピソードの
    履歴が混ざるため、エピソード開始 2 ステップ (``episode_length_buf < 2``) は 0 を返す
    (``action_smoothness_l2`` / ``high_action_smoothness_l2`` と同じ規約)。

    Args:
        env: 学習環境。
        asset_cfg: CoM を計算する Articulation。全身を対象にするため body は絞らない。

    Returns:
        環境ごとのペナルティ値 [(m/s³)²], shape (N,)。
    """
    asset = env.scene[asset_cfg.name]

    # 全身 CoM 速度 (world frame) = Σ m_i v_i / Σ m_i
    masses = asset.data.default_mass.to(env.device)  # [N, num_bodies]
    body_com_vel = asset.data.body_com_vel_w[:, :, :3]  # [N, num_bodies, 3]
    total_mass = masses.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [N, 1]
    com_vel = (masses.unsqueeze(-1) * body_com_vel).sum(dim=1) / total_mass  # [N, 3]

    # 履歴バッファ (初回 or 形状不整合なら現在値で初期化)
    if (not hasattr(env, "_com_vel_prev")) or env._com_vel_prev.shape != com_vel.shape:
        env._com_vel_prev = com_vel.clone()
        env._com_vel_prev_prev = com_vel.clone()

    dt = env.step_dt
    jerk = (com_vel - 2.0 * env._com_vel_prev + env._com_vel_prev_prev) / (dt ** 2)  # [N, 3]
    penalty = torch.sum(torch.square(jerk), dim=1)  # [N]

    # 履歴更新 (次ステップで使う)
    env._com_vel_prev_prev = env._com_vel_prev.clone()
    env._com_vel_prev = com_vel.clone()

    # リセット直後は前エピソードの速度が二階差分に混ざるのでペナルティを無効化
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(penalty), penalty)


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


def compute_zmp_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """力学ベース ZMP の XY を world frame で返す (E, 2)。"""
    robot = env.scene[asset_cfg.name]
    g = 9.81
    m  = robot.data.default_mass.to(env.device)      # (E, B)
    pc = robot.data.body_com_pos_w                   # (E, B, 3)  無ければ body_pos_w
    ac = robot.data.body_lin_acc_w                   # (E, B, 3)  無ければ body_acc_w の線形成分

    z0 = pc[..., 2].min(dim=1, keepdim=True).values  # 接地高さ近似(最下リンク)
    wz    = m * (ac[..., 2] + g)                     # (E, B)
    denom = wz.sum(1).clamp(min=1e-6)                # (E,)
    px = (wz * pc[..., 0] - m * ac[..., 0] * (pc[..., 2] - z0)).sum(1) / denom
    py = (wz * pc[..., 1] - m * ac[..., 1] * (pc[..., 2] - z0)).sum(1) / denom
    return torch.stack([px, py], dim=-1)             # (E, 2)


def zmp_support_center(
    env: ManagerBasedRLEnv,
    sigma: float = 0.08,            # [m] 距離スケール。足裏半長より少し小さめが目安
    force_threshold: float = 20.0, # [N] 接地判定の Fz しきい値
    ema_alpha: float = 0.2,        # ZMP の平滑化係数。1.0 で無効
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # ZMP 力学は全身質量で計算するため全リンク
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),  # 支持基準点用の足リンク
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
) -> torch.Tensor:
    """ZMP と支持基準点(片脚=立脚足中心 / 両足=両足中点)の距離を exp カーネルで報酬化。"""
    robot   = env.scene[asset_cfg.name]
    contact = env.scene.sensors[sensor_cfg.name]

    # foot_asset_cfg は足リンクのみ、sensor_cfg と本数が一致している必要がある。
    # asset_cfg(全身)を foot_ids に使うと body 数が contact sensor とずれてブロードキャストに失敗する。
    foot_ids = foot_asset_cfg.body_ids                              # 足リンクの body id
    foot_xy  = robot.data.body_pos_w[:, foot_ids, :2]               # (E, F, 2)
    fz       = contact.data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp(min=0.0)  # (E, F)

    # --- 支持基準点:接地している足の XY 平均 ---
    in_contact = fz > force_threshold                               # (E, F) bool
    w     = in_contact.float()                                      # (E, F)
    denom = w.sum(1, keepdim=True).clamp(min=1.0)                   # (E, 1)
    ref_xy = (foot_xy * w.unsqueeze(-1)).sum(1) / denom            # (E, 2)

    # --- ZMP(EMA で平滑化)---
    zmp_xy = compute_zmp_xy(env, asset_cfg)                         # (E, 2)
    if ema_alpha < 1.0:
        prev = getattr(env, "_zmp_ema", None)
        if prev is None or prev.shape[0] != zmp_xy.shape[0]:
            prev = zmp_xy.detach()
        zmp_xy = ema_alpha * zmp_xy + (1.0 - ema_alpha) * prev
        env._zmp_ema = zmp_xy.detach()

    # --- exp カーネル ---
    d = torch.norm(zmp_xy - ref_xy, dim=-1)                         # (E,)
    reward = torch.exp(-(d ** 2) / (sigma ** 2))

    # 遊脚相(無接地)は基準点が定義できないので 0 にゲート
    any_contact = in_contact.any(1)
    return torch.where(any_contact, reward, torch.zeros_like(reward))

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
    "feet_landing_impact",
    "feet_landing_vel",
    "compute_zmp_xy",
    "zmp_support_center",
]


def base_ang_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base (Trunk) の角加速度の L2 二乗をペナルティとして返す (weight < 0 で使う)。

    ★ 2026-08-26: feat/inoue_walk_double_encoder から移植。
      ☠ 我々の ``goalkeeper/mdp/rewards.py::body_jitter`` は角速度の2階差分を
        **有界化** (d/(d+w_ref)) しているので、大きい領域で飽和して勾配が消える。
        本項は生の二乗なので飽和せず、**歩行→停止の減速中のような大きなジッタ**にも
        勾配が残る。補完関係にあるので併用する。

    頭部は Trunk に剛結合されているため、頭の振動 = Trunk の回転ジッタ × レバーアーム。
    角速度ペナルティ (ang_vel_xy_l2) はゆっくりした揺れを抑えるが、高周波の
    カタカタしたジッタは「速度は小さいが加速度が大きい」ため取りこぼす。
    角加速度を直接罰することで頭部の振動を抑える。

    物理エンジンの生の加速度はスパイクを含むため、角速度の有限差分で計算し、
    リセット直後は無効化する (com_jerk_l2 と同じパターン)。
    """
    robot = env.scene[asset_cfg.name]
    ang_vel = robot.data.root_ang_vel_b  # (N, 3)
    if not hasattr(env, "_prev_root_ang_vel_b") or env._prev_root_ang_vel_b.shape != ang_vel.shape:
        env._prev_root_ang_vel_b = ang_vel.clone()
    prev = env._prev_root_ang_vel_b
    dt = env.step_dt
    ang_acc = (ang_vel - prev) / max(dt, 1e-6)
    env._prev_root_ang_vel_b = ang_vel.clone()
    penalty = torch.sum(torch.square(ang_acc), dim=1)
    fresh = env.episode_length_buf < 2
    return torch.where(fresh, torch.zeros_like(penalty), penalty)
