# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for the K1 stand-up task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_height_above_limit(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg,
    maximum_height: float = 0.15,
    grace_s: float = 0.25,
) -> torch.Tensor:
    """Terminate when either foot rises above the maximum ground-relative height."""
    asset = env.scene[foot_cfg.name]
    highest_foot = asset.data.body_pos_w[:, foot_cfg.body_ids, 2].max(dim=1).values
    grace_elapsed = env.episode_length_buf.float() * env.step_dt >= grace_s
    return grace_elapsed & (highest_foot >= maximum_height)


def head_high_while_hip_low(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    minimum_head_height: float = 0.25,
    minimum_hip_height: float = 0.2,
    grace_s: float = 0.25,
) -> torch.Tensor:
    """Terminate when the head rises before the lowest hip reaches bridge height."""
    asset = env.scene[head_cfg.name]
    highest_head = asset.data.body_pos_w[:, head_cfg.body_ids, 2].max(dim=1).values
    lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
    grace_elapsed = env.episode_length_buf.float() * env.step_dt >= grace_s
    return grace_elapsed & (highest_head >= minimum_head_height) & (lowest_hip <= minimum_hip_height)


def fall_above_head_height(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg,
    head_sensor_names: list[str],
    hip_sensor_names: list[str],
    foot_sensor_names: list[str],
    trunk_sensor_names: list[str] | None = None,
    hip_cfg: SceneEntityCfg | None = None,
    shoulder_cfg: SceneEntityCfg | None = None,
    min_head_height: float = 0.35,
    trunk_contact_head_height: float = 0.35,
    bridge_hip_height: float = 0.45,
    min_bridge_upper_body_height: float = 0.2,
    head_support_grace_s: float = 1.0,
    min_head_contact_hip_height: float = 0.3,
    max_trunk_height: float = 0.3,
    contact_threshold: float = 1.0,
    head_contact_threshold: float | None = None,
) -> torch.Tensor:
    """Terminate invalid raised or bridged poses, including head-supported bridges."""
    asset = env.scene[head_cfg.name]
    head_heights = asset.data.body_pos_w[:, head_cfg.body_ids, 2]
    head_height = head_heights.max(dim=1).values
    lowest_head = head_heights.min(dim=1).values
    trunk_height = asset.data.root_pos_w[:, 2]
    if head_contact_threshold is None:
        head_contact_threshold = contact_threshold
    head_on_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in head_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        head_on_ground |= (ground_force > head_contact_threshold).any(dim=(1, 2))

    hip_on_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in hip_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        hip_on_ground |= (ground_force > contact_threshold).any(dim=(1, 2))

    foot_contacts = []
    for sensor_name in foot_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        foot_contacts.append((ground_force > contact_threshold).any(dim=(1, 2)))
    both_feet_airborne = ~torch.stack(foot_contacts, dim=1).any(dim=1)

    trunk_on_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in trunk_sensor_names or []:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is None:
            raise RuntimeError(f"{sensor_name} requires a ground-filtered contact sensor")
        ground_force = sensor.data.force_matrix_w.norm(dim=-1)
        trunk_on_ground |= (ground_force > contact_threshold).any(dim=(1, 2))

    raised_head = head_height >= min_head_height
    late_trunk_contact = (head_height >= trunk_contact_head_height) & trunk_on_ground
    bridge_upper_body_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    head_supported_bridge = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if hip_cfg is not None and shoulder_cfg is not None:
        lowest_hip = asset.data.body_pos_w[:, hip_cfg.body_ids, 2].min(dim=1).values
        lowest_shoulder = asset.data.body_pos_w[:, shoulder_cfg.body_ids, 2].min(dim=1).values
        upper_body_too_low = (lowest_head <= min_bridge_upper_body_height) | (
            lowest_shoulder <= min_bridge_upper_body_height
        )
        bridge_upper_body_low = (lowest_hip >= bridge_hip_height) & upper_body_too_low
        head_supported_bridge = head_on_ground & (lowest_hip >= min_head_contact_hip_height)
    head_support_grace_elapsed = env.episode_length_buf.float() * env.step_dt >= head_support_grace_s
    head_support_after_trunk_lift = head_support_grace_elapsed & head_on_ground & ~trunk_on_ground
    return (
        raised_head & ((trunk_height <= max_trunk_height) | hip_on_ground | both_feet_airborne)
    ) | late_trunk_contact | bridge_upper_body_low | head_supported_bridge | head_support_after_trunk_lift


def standing_success_mask(
    env: ManagerBasedRLEnv,
    foot_sensor_cfg: SceneEntityCfg,
    hand_sensor_cfg: SceneEntityCfg,
    min_height: float = 0.6,
    max_tilt_deg: float = 20.0,
    max_lin_speed: float = 0.2,
    max_ang_speed: float = 0.4,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return environments that currently satisfy the complete standing condition."""
    asset = env.scene[asset_cfg.name]
    foot_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
    hand_sensor: ContactSensor = env.scene.sensors[hand_sensor_cfg.name]

    foot_force = foot_sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :].norm(dim=-1)
    hand_force = hand_sensor.data.net_forces_w[:, hand_sensor_cfg.body_ids, :].norm(dim=-1)
    both_feet = (foot_force > contact_threshold).all(dim=1)
    hands_free = ~(hand_force > contact_threshold).any(dim=1)

    max_tilt = math.radians(max_tilt_deg)
    gravity_xy = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1)
    upright = gravity_xy < math.sin(max_tilt)
    high_enough = asset.data.root_pos_w[:, 2] >= min_height
    slow_linear = torch.linalg.vector_norm(asset.data.root_lin_vel_b, dim=1) <= max_lin_speed
    slow_angular = torch.linalg.vector_norm(asset.data.root_ang_vel_b, dim=1) <= max_ang_speed
    return high_enough & upright & slow_linear & slow_angular & both_feet & hands_free


def stable_standing(
    env: ManagerBasedRLEnv,
    foot_sensor_cfg: SceneEntityCfg,
    hand_sensor_cfg: SceneEntityCfg,
    stable_time_s: float = 0.5,
    min_height: float = 0.6,
    max_tilt_deg: float = 20.0,
    max_lin_speed: float = 0.2,
    max_ang_speed: float = 0.4,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully after the standing condition is held continuously."""
    standing = standing_success_mask(
        env,
        foot_sensor_cfg=foot_sensor_cfg,
        hand_sensor_cfg=hand_sensor_cfg,
        min_height=min_height,
        max_tilt_deg=max_tilt_deg,
        max_lin_speed=max_lin_speed,
        max_ang_speed=max_ang_speed,
        contact_threshold=contact_threshold,
        asset_cfg=asset_cfg,
    )

    counter = getattr(env, "_getup_stable_standing_steps", None)
    if counter is None or counter.shape != standing.shape:
        counter = torch.zeros_like(env.episode_length_buf)
        env._getup_stable_standing_steps = counter

    fresh = env.episode_length_buf < 2
    counter[fresh] = 0
    counter[:] = torch.where(standing, counter + 1, torch.zeros_like(counter))
    required_steps = max(1, math.ceil(stable_time_s / env.step_dt))
    return counter >= required_steps
