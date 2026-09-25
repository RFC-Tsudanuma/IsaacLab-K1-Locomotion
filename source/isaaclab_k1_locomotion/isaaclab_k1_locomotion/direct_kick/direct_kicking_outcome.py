# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/direct_kicking_outcome.py
"""Outcome helpers for the DirectKicking task."""

from typing import Tuple

import torch


def normalize_xy_direction(
    vector_xy: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize XY vectors while mapping zero-length vectors to zero."""
    if vector_xy.ndim < 1 or vector_xy.shape[-1] != 2:
        raise ValueError("vector_xy must end with two XY components")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    norm = torch.norm(vector_xy, dim=-1, keepdim=True)
    normalized = vector_xy / torch.clamp(norm, min=epsilon)
    return torch.where(norm > epsilon, normalized, torch.zeros_like(normalized))


def target_direction_alignment(
    trajectory_xy: torch.Tensor,
    target_direction_xy: torch.Tensor,
    sharpness: float = 4.0,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return a distance-independent alignment score in ``[0, 1]``.

    Only the angle between the observed trajectory and target direction affects
    the score. Zero-length trajectories have no defined direction and receive
    zero reward.
    """
    if trajectory_xy.shape != target_direction_xy.shape:
        raise ValueError("trajectory_xy and target_direction_xy must match")
    if trajectory_xy.ndim < 1 or trajectory_xy.shape[-1] != 2:
        raise ValueError("trajectory tensors must end with two XY components")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")

    trajectory_norm = torch.norm(trajectory_xy, dim=-1)
    target_norm = torch.norm(target_direction_xy, dim=-1)
    safe_trajectory_norm = torch.clamp(trajectory_norm, min=epsilon)
    safe_target_norm = torch.clamp(target_norm, min=epsilon)
    cosine = torch.sum(trajectory_xy * target_direction_xy, dim=-1) / (
        safe_trajectory_norm * safe_target_norm
    )
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)
    alignment = torch.exp(float(sharpness) * (cosine - 1.0))
    direction_is_valid = (trajectory_norm > epsilon) & (target_norm > epsilon)
    return torch.where(direction_is_valid, alignment, torch.zeros_like(alignment))


def sigmoid_velocity_change_scale(
    velocity_change_xy: torch.Tensor,
    center_mps: float = 0.5,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """Return a smooth kick-strength scale centered on a velocity change.

    The scale is 0.5 when the XY velocity-change magnitude equals
    ``center_mps`` and approaches 1.0 for larger changes.
    """
    if velocity_change_xy.ndim < 1 or velocity_change_xy.shape[-1] != 2:
        raise ValueError("velocity_change_xy must end with two XY components")
    if center_mps < 0.0:
        raise ValueError("center_mps must be non-negative")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")

    speed_change = torch.norm(velocity_change_xy, dim=-1)
    return torch.sigmoid(float(sharpness) * (speed_change - float(center_mps)))


def one_shot_direction_reward(
    resulting_velocity_xy: torch.Tensor,
    target_direction_xy: torch.Tensor,
    new_kick: torch.Tensor,
    max_reward: float,
    sharpness: float = 4.0,
) -> torch.Tensor:
    """Score resulting ball direction only on the first physical-kick step."""
    if new_kick.shape != resulting_velocity_xy.shape[:-1]:
        raise ValueError("new_kick must match the velocity batch shape")
    if max_reward < 0.0:
        raise ValueError("max_reward must be non-negative")
    alignment = target_direction_alignment(
        resulting_velocity_xy,
        target_direction_xy,
        sharpness=sharpness,
    )
    return alignment * float(max_reward) * new_kick.float()


def physical_kick_event(
    foot_candidate: torch.Tensor,
    current_ball_velocity_xy: torch.Tensor,
    previous_ball_velocity_xy: torch.Tensor,
    min_velocity_change: float,
) -> torch.Tensor:
    """Detect a physical kick without consulting the desired target direction."""
    if current_ball_velocity_xy.shape != previous_ball_velocity_xy.shape:
        raise ValueError("current and previous ball velocities must match")
    if (
        current_ball_velocity_xy.ndim < 1
        or current_ball_velocity_xy.shape[-1] != 2
    ):
        raise ValueError("ball velocity tensors must end with two XY components")
    if foot_candidate.shape != current_ball_velocity_xy.shape[:-1]:
        raise ValueError("foot_candidate must match the velocity batch shape")
    if min_velocity_change <= 0.0:
        raise ValueError("min_velocity_change must be positive")
    velocity_change = torch.norm(
        current_ball_velocity_xy - previous_ball_velocity_xy,
        dim=-1,
    )
    return foot_candidate & (velocity_change >= min_velocity_change)


def latch_first_kick_step(
    first_kick_step: torch.Tensor,
    current_step: torch.Tensor,
    new_kick: torch.Tensor,
) -> torch.Tensor:
    """Latch a kick step once without allowing later detections to replace it."""
    if (
        first_kick_step.shape != current_step.shape
        or new_kick.shape != current_step.shape
    ):
        raise ValueError("kick-step tensors must have matching shapes")
    should_latch = (first_kick_step < 0) & new_kick
    return torch.where(should_latch, current_step, first_kick_step)


def update_post_kick_phase_target(
    kicking_foot_mask: torch.Tensor,
    previous_feet_contact: torch.Tensor,
    feet_contact: torch.Tensor,
    kicking_foot_airborne: torch.Tensor,
    post_kick_phase_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Latch phase one when a kicking foot lands after being airborne."""
    if kicking_foot_mask.shape != feet_contact.shape:
        raise ValueError("kicking_foot_mask and feet_contact must match")
    if previous_feet_contact.shape != kicking_foot_mask.shape:
        raise ValueError("previous_feet_contact must match kicking_foot_mask")
    if kicking_foot_airborne.shape != kicking_foot_mask.shape:
        raise ValueError("kicking_foot_airborne must match kicking_foot_mask")
    if post_kick_phase_target.shape != kicking_foot_mask.shape[:-1]:
        raise ValueError(
            "post_kick_phase_target must match the environment batch shape"
        )
    if kicking_foot_mask.ndim < 1 or kicking_foot_mask.shape[-1] != 2:
        raise ValueError("kick phase tracking requires exactly two feet")

    was_airborne = kicking_foot_airborne | (
        kicking_foot_mask & ~previous_feet_contact
    )
    landed = torch.any(
        kicking_foot_mask & was_airborne & feet_contact,
        dim=-1,
    )
    next_airborne = was_airborne | (
        kicking_foot_mask & ~feet_contact
    )
    next_target = post_kick_phase_target | landed
    return next_airborne, next_target


def post_kick_termination_mask(
    valid_kick: torch.Tensor,
    first_kick_step: torch.Tensor,
    current_step: torch.Tensor,
    duration_steps: int,
) -> torch.Tensor:
    """Return environments whose first kick is at least ``duration_steps`` old."""
    if (
        valid_kick.shape != first_kick_step.shape
        or current_step.shape != first_kick_step.shape
    ):
        raise ValueError("post-kick timer tensors must have matching shapes")
    if duration_steps <= 0:
        raise ValueError("duration_steps must be positive")
    return (
        valid_kick
        & (first_kick_step >= 0)
        & ((current_step - first_kick_step) >= duration_steps)
    )


def post_kick_walking_pose_reward(
    current_dof_pos: torch.Tensor,
    walking_dof_pos: torch.Tensor,
    valid_kick: torch.Tensor,
    error_scale_rad: float,
) -> torch.Tensor:
    """Reward recovery toward the deployed walking policy's nominal pose."""
    if current_dof_pos.shape != walking_dof_pos.shape:
        raise ValueError("current_dof_pos and walking_dof_pos must match")
    if current_dof_pos.ndim < 1:
        raise ValueError("dof position tensors must have at least one dimension")
    if valid_kick.shape != current_dof_pos.shape[:-1]:
        raise ValueError("valid_kick must match the DOF position batch shape")
    if error_scale_rad <= 0.0:
        raise ValueError("error_scale_rad must be positive")

    mean_squared_error = torch.mean(
        torch.square(current_dof_pos - walking_dof_pos),
        dim=-1,
    )
    reward = torch.exp(
        -mean_squared_error / float(error_scale_rad) ** 2
    )
    return reward * valid_kick.float()
