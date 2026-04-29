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
) -> torch.Tensor:
    """Reward based on matching foot contact pattern to a periodic phase oscillator.

    This function encourages a natural alternating bipedal gait by comparing actual
    foot contacts against a desired contact schedule derived from a phase oscillator.

    The two feet are driven in anti-phase (left: phi, right: phi + pi).  A foot is
    expected to be in *stance* (on the ground) when its phase falls in the first
    ``stance_ratio`` of the cycle, and in *swing* (in the air) for the rest.

    The reward is +1 per foot that matches its desired contact state, giving a maximum
    of +2 per step.  The reward is gated to zero when the velocity command is small so
    that standing still is not penalised.

    Args:
        env: The learning environment.
        sensor_cfg: Contact sensor configuration (must cover both foot bodies, left first).
        command_name: Name of the velocity command in the command manager.
        phase_freq: Frequency of the gait cycle in Hz.  Default 1.5 Hz (~normal walking).
        stance_ratio: Fraction of the cycle that each foot spends in stance (0–1).
            Default 0.55 (slightly more stance than swing, typical for walking).

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

    # Actual contact: True when net contact force exceeds 1 N
    # net_forces_w_history: [N, history, num_bodies, 3]
    actual_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )  # [N, 2]

    # +1 for each foot that matches its desired contact state
    reward = torch.sum((actual_contact == desired_stance).float(), dim=1)  # [N]

    # Gate: no reward when the command speed is near zero (standing still)
    cmd_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= cmd_speed > 0.1

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

__all__ = [
    "minimum_height",
    "track_lin_vel_xy_discrete_exp",
    "track_ang_vel_z_discrete_exp",
    "feet_distance",
    "feet_phase",
    "joint_mirror_symmetry",
]