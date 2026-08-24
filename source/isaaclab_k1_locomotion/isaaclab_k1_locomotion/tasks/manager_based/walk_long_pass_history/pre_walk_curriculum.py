# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standing inside-kick curriculum used before the walking stage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .inside_rewards import _FORM_STATE_ATTR, _STAGE_STATE_ATTR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_STATE_ATTR = "_inside_pre_walk_curriculum_state"
_INITIAL_YAW_ATTR = "_inside_pre_walk_initial_yaw"
_EPS = 1.0e-6


def reset_ball_for_pre_walk_kick(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "kick_direction",
    r_stance: float = 0.25,
    side_offset: float = 0.096,
    longitudinal_jitter: float = 0.0,
    lateral_jitter: float = 0.0,
    aligned_to_kick_direction: bool = True,
    walk_dist_range: tuple[float, float] = (0.5, 1.5),
    walk_half_angle: float = math.pi / 2,
    ball_radius: float = 0.11,
    spawn_clearance: float = 0.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> None:
    """Reset the ball at a standing kick pose, then at normal walking distances.

    In the pre-walk stages the ball follows the sampled kick direction.  A
    symmetric ``side_offset`` chooses the left or right foot side with equal
    probability.  The longitudinal and lateral jitters are widened together
    with the direction curriculum.  Once ``aligned_to_kick_direction`` becomes
    false, the reset switches to the front-half walking distribution.
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    count = len(env_ids)

    robot_pos = robot.data.root_pos_w[env_ids, :2]
    quat = robot.data.root_quat_w[env_ids]
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3]),
    )
    initial_yaw = getattr(env, _INITIAL_YAW_ATTR, None)
    if initial_yaw is None:
        initial_yaw = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _INITIAL_YAW_ATTR, initial_yaw)
    initial_yaw[env_ids] = yaw

    command = env.command_manager.get_command(command_name)[env_ids]
    kick_dir = torch.stack([command[:, 1], command[:, 0]], dim=-1)
    right_vec = torch.stack([kick_dir[:, 1], -kick_dir[:, 0]], dim=-1)

    if aligned_to_kick_direction:
        side = torch.where(
            torch.rand(count, device=env.device) < 0.5,
            -torch.ones(count, device=env.device),
            torch.ones(count, device=env.device),
        )
        longitudinal = torch.empty(count, device=env.device).uniform_(
            -longitudinal_jitter, longitudinal_jitter
        )
        lateral = torch.empty(count, device=env.device).uniform_(
            -lateral_jitter, lateral_jitter
        )
        ball_xy = (
            robot_pos
            + (r_stance + longitudinal).unsqueeze(-1) * kick_dir
            + (side * side_offset + lateral).unsqueeze(-1) * right_vec
        )
    else:
        distance = torch.empty(count, device=env.device).uniform_(*walk_dist_range)
        angle = yaw + torch.empty(count, device=env.device).uniform_(
            -walk_half_angle, walk_half_angle
        )
        ball_xy = robot_pos + distance.unsqueeze(-1) * torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=-1
        )

    state = ball.data.default_root_state[env_ids].clone()
    state[:, :2] = ball_xy
    state[:, 2] = env.scene.env_origins[env_ids, 2] + ball_radius + spawn_clearance
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0
    ball.write_root_state_to_sim(state, env_ids=env_ids)


def pre_walk_initial_yaw_deviation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize yaw drift from reset orientation during the standing stages."""
    initial_yaw = getattr(env, _INITIAL_YAW_ATTR, None)
    curriculum = getattr(env, _STATE_ATTR, None)
    if initial_yaw is None or curriculum is None:
        return torch.zeros(env.num_envs, device=env.device)

    quat = env.scene["robot"].data.root_quat_w
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3]),
    )
    max_pre_walk_level = len(curriculum["direction_half_angles_deg"]) - 1
    if curriculum["level"] > max_pre_walk_level:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.square(wrap_to_pi(yaw - initial_yaw))


def pre_walk_inside_kick_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    reset_event_name: str,
    direction_half_angles_deg: tuple[float, ...],
    position_jitter_m: tuple[float, ...],
    promote_kick_rate: float,
    promote_inside_contact_rate: float,
    promote_direction_error_deg: float,
    inside_contact_angle_deg: float,
    ema_alpha: float = 0.01,
    standing_max_vel: float = 0.0,
    standing_max_ang_vel: float = 0.0,
    walking_max_vel: float = 1.0,
    walking_max_ang_vel: float = 1.0,
) -> dict[str, float]:
    """Widen direction/ball tolerance, then release the walking command.

    Promotion uses the same kick-rate and inside-contact criteria as the
    existing inside-kick stages plus their existing direction-error threshold.
    Every promotion resets the EMA evidence so one successful distribution
    cannot skip multiple difficulty levels.
    """
    if len(direction_half_angles_deg) == 0:
        raise ValueError("direction_half_angles_deg must not be empty")
    if len(direction_half_angles_deg) != len(position_jitter_m):
        raise ValueError("direction angles and position jitters must have equal length")

    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = {
            "level": 0,
            "direction_half_angles_deg": direction_half_angles_deg,
            "kick_rate_ema": 0.0,
            "inside_contact_rate_ema": 0.0,
            "direction_error_ema_deg": 180.0,
        }
        setattr(env, _STATE_ATTR, state)

    command_term = env.command_manager.get_term(command_name)
    reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
    form = getattr(env, _FORM_STATE_ATTR, None)
    inside_stage = getattr(env, _STAGE_STATE_ATTR, None)

    kick_metric = command_term.metrics.get("kick_rate")
    direction_metric = command_term.metrics.get("kick_dir_error_deg")
    if (
        kick_metric is not None
        and direction_metric is not None
        and form is not None
        and env_ids is not None
        and len(env_ids) > 0
    ):
        kick_done = kick_metric[env_ids]
        kick_rate = float(kick_done.mean())
        state["kick_rate_ema"] += ema_alpha * (kick_rate - state["kick_rate_ema"])

        successful = kick_done > 0.5
        successful_count = float(successful.float().sum())
        if successful_count > 0.0:
            threshold_cos = math.cos(math.radians(inside_contact_angle_deg))
            valid_inside = (
                form["form_valid_frozen"][env_ids]
                & successful
                & (form["inside_contact_cos_frozen"][env_ids] >= threshold_cos)
            )
            inside_rate = float(valid_inside.float().sum()) / successful_count
            direction_error = float(direction_metric[env_ids].sum()) / max(
                float(kick_done.sum()), _EPS
            )
            state["inside_contact_rate_ema"] += ema_alpha * (
                inside_rate - state["inside_contact_rate_ema"]
            )
            state["direction_error_ema_deg"] += ema_alpha * (
                direction_error - state["direction_error_ema_deg"]
            )

    direction_learning_enabled = inside_stage is not None and inside_stage["stage"] >= 2
    qualified = (
        direction_learning_enabled
        and state["kick_rate_ema"] >= promote_kick_rate
        and state["inside_contact_rate_ema"] >= promote_inside_contact_rate
        and state["direction_error_ema_deg"] <= promote_direction_error_deg
    )
    max_pre_walk_level = len(direction_half_angles_deg) - 1
    if qualified and state["level"] <= max_pre_walk_level:
        state["level"] += 1
        state["kick_rate_ema"] = 0.0
        state["inside_contact_rate_ema"] = 0.0
        state["direction_error_ema_deg"] = 180.0

    walking = state["level"] > max_pre_walk_level
    active_level = min(state["level"], max_pre_walk_level)
    half_angle = math.radians(direction_half_angles_deg[active_level])
    jitter = position_jitter_m[active_level]

    command_term.cfg.ranges.heading = (-half_angle, half_angle)
    reset_cfg.params["aligned_to_kick_direction"] = not walking
    reset_cfg.params["longitudinal_jitter"] = jitter
    reset_cfg.params["lateral_jitter"] = jitter

    base_command = env.command_manager.get_term("base_velocity")
    base_command.cfg.max_vel = walking_max_vel if walking else standing_max_vel
    base_command.cfg.max_ang_vel = walking_max_ang_vel if walking else standing_max_ang_vel

    return {
        "level": float(state["level"]),
        "walking": float(walking),
        "direction_half_angle_deg": direction_half_angles_deg[active_level],
        "position_jitter_m": jitter,
        "kick_rate_ema": state["kick_rate_ema"],
        "inside_contact_rate_ema": state["inside_contact_rate_ema"],
        "direction_error_ema_deg": state["direction_error_ema_deg"],
    }


__all__ = [
    "pre_walk_initial_yaw_deviation",
    "pre_walk_inside_kick_curriculum",
    "reset_ball_for_pre_walk_kick",
]
