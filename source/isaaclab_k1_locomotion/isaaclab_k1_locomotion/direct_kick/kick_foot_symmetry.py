# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/kick_foot_symmetry.py
"""Foot-symmetric tensor helpers for direct kicking."""

import torch


def per_foot_kick_candidates(
    current_feet_pos,
    previous_feet_pos,
    current_ball_pos,
    previous_ball_pos,
    dt,
    max_foot_ball_distance,
    min_foot_speed_towards_ball,
):
    """Return per-foot contact-like motion toward the ball."""
    current_ball_pos = current_ball_pos.unsqueeze(1)
    previous_ball_pos = previous_ball_pos.unsqueeze(1)

    current_distance = torch.norm(current_feet_pos - current_ball_pos, dim=-1)
    previous_distance = torch.norm(previous_feet_pos - previous_ball_pos, dim=-1)
    foot_ball_distance = torch.minimum(current_distance, previous_distance)

    previous_foot_to_ball = previous_ball_pos - previous_feet_pos
    previous_foot_to_ball_direction = previous_foot_to_ball / (
        torch.norm(previous_foot_to_ball, dim=-1, keepdim=True) + 1.0e-6
    )
    foot_velocity = (current_feet_pos - previous_feet_pos) / dt
    foot_speed_towards_ball = torch.sum(
        foot_velocity * previous_foot_to_ball_direction,
        dim=-1,
    )

    return (
        (foot_ball_distance <= max_foot_ball_distance)
        & (foot_speed_towards_ball >= min_foot_speed_towards_ball)
    )


def either_foot_kick_candidate(
    current_feet_pos,
    previous_feet_pos,
    current_ball_pos,
    previous_ball_pos,
    dt,
    max_foot_ball_distance,
    min_foot_speed_towards_ball,
):
    """Return whether either foot produced contact-like motion toward the ball."""
    return torch.any(
        per_foot_kick_candidates(
            current_feet_pos,
            previous_feet_pos,
            current_ball_pos,
            previous_ball_pos,
            dt,
            max_foot_ball_distance,
            min_foot_speed_towards_ball,
        ),
        dim=-1,
    )


def select_kicking_foot_mask(
    per_foot_candidate,
    current_feet_pos,
    previous_feet_pos,
    current_ball_pos,
    previous_ball_pos,
):
    """Select the closest candidate; exact distance ties choose foot index zero."""
    if per_foot_candidate.ndim < 1 or per_foot_candidate.shape[-1] != 2:
        raise ValueError("kicking-foot selection requires exactly two feet")
    if (
        current_feet_pos.shape != previous_feet_pos.shape
        or current_feet_pos.shape[:-1] != per_foot_candidate.shape
    ):
        raise ValueError("foot positions must match per-foot candidates")
    if (
        current_ball_pos.shape != previous_ball_pos.shape
        or current_ball_pos.shape != current_feet_pos.shape[:-2] + (3,)
    ):
        raise ValueError("ball positions must match the environment batch")

    current_distance = torch.norm(
        current_feet_pos - current_ball_pos.unsqueeze(-2),
        dim=-1,
    )
    previous_distance = torch.norm(
        previous_feet_pos - previous_ball_pos.unsqueeze(-2),
        dim=-1,
    )
    distance = torch.minimum(current_distance, previous_distance)
    masked_distance = torch.where(
        per_foot_candidate,
        distance,
        torch.full_like(distance, torch.inf),
    )
    selected_index = torch.argmin(masked_distance, dim=-1)
    selected = torch.nn.functional.one_hot(
        selected_index,
        num_classes=2,
    ).to(dtype=torch.bool)
    return selected & torch.any(
        per_foot_candidate,
        dim=-1,
        keepdim=True,
    )


def nearest_foot_distance_progress(
    current_feet_pos,
    previous_feet_pos,
    ball_pos,
    distance_scale,
):
    """Return progress of the nearest-foot distance as a symmetric potential."""
    ball_pos = ball_pos.unsqueeze(1)
    current_distance = torch.norm(current_feet_pos - ball_pos, dim=-1)
    previous_distance = torch.norm(previous_feet_pos - ball_pos, dim=-1)
    current_nearest = torch.min(current_distance, dim=-1).values
    previous_nearest = torch.min(previous_distance, dim=-1).values
    return (previous_nearest - current_nearest) / distance_scale


def mirrored_strike_position_progress(
    current_ball_pos_local,
    previous_ball_pos_local,
    nominal_strike_point,
    distance_scale,
):
    """Return body-pose progress toward either mirrored strike position."""
    if current_ball_pos_local.shape != previous_ball_pos_local.shape:
        raise ValueError("current and previous local ball positions must match")
    if (
        current_ball_pos_local.ndim < 1
        or current_ball_pos_local.shape[-1] != 2
    ):
        raise ValueError("local ball positions must end with two XY components")
    if len(nominal_strike_point) != 2:
        raise ValueError("nominal_strike_point must contain [x, abs(y)]")
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive")

    strike_x = float(nominal_strike_point[0])
    strike_y = float(nominal_strike_point[1])
    if strike_x <= 0.0 or strike_y <= 0.0:
        raise ValueError("nominal strike coordinates must be positive")

    def nearest_strike_cost(ball_pos_local):
        forward_error = ball_pos_local[..., 0] - strike_x
        lateral_error = torch.minimum(
            torch.abs(ball_pos_local[..., 1] - strike_y),
            torch.abs(ball_pos_local[..., 1] + strike_y),
        )
        return torch.sqrt(
            torch.square(forward_error) + torch.square(lateral_error)
        )

    current_cost = nearest_strike_cost(current_ball_pos_local)
    previous_cost = nearest_strike_cost(previous_ball_pos_local)
    return (previous_cost - current_cost) / distance_scale


def max_foot_height_excess_penalty(foot_heights, height_limit, excess_scale):
    """Penalize excessive lift equally for either foot without double counting."""
    height_excess = torch.clamp(foot_heights - height_limit, min=0.0) / excess_scale
    per_foot_penalty = torch.square(height_excess)
    return torch.max(per_foot_penalty, dim=-1).values


def update_edge_only_support_penalty(
    toe_contact,
    heel_contact,
    loaded,
    active,
    previous_steps,
    required_steps,
):
    """Track persistent loaded support on only the toe or heel edge."""
    if (
        toe_contact.shape != heel_contact.shape
        or toe_contact.shape != loaded.shape
        or toe_contact.shape != previous_steps.shape
    ):
        raise ValueError("toe, heel, load, and support-step tensors must match")
    if active.shape != toe_contact.shape[:-1]:
        raise ValueError("active must match the environment batch dimensions")
    if required_steps <= 0:
        raise ValueError("required_steps must be positive")

    edge_only = torch.logical_xor(toe_contact, heel_contact)
    qualifying = edge_only & loaded & active.unsqueeze(-1)
    next_steps = torch.where(
        qualifying,
        previous_steps + 1,
        torch.zeros_like(previous_steps),
    )
    persistent = next_steps >= int(required_steps)
    penalty = torch.any(persistent, dim=-1).to(dtype=torch.float)
    return next_steps, penalty


def nonparallel_step_penalty(
    foot_velocity_local,
    feet_contact,
    lateral_velocity_tolerance,
    lateral_velocity_scale,
):
    """Penalize lateral motion when a swing foot is stepping forward."""
    if foot_velocity_local.ndim < 2 or foot_velocity_local.shape[-2:] != (2, 2):
        raise ValueError("foot_velocity_local must end with [two feet, XY]")
    if feet_contact.shape != foot_velocity_local.shape[:-1]:
        raise ValueError("feet_contact must match the foot dimensions")
    if lateral_velocity_tolerance < 0.0:
        raise ValueError("lateral_velocity_tolerance must be non-negative")
    if lateral_velocity_scale <= 0.0:
        raise ValueError("lateral_velocity_scale must be positive")

    single_support_swing = (~feet_contact) & feet_contact.flip(-1)
    stepping_forward = foot_velocity_local[..., 0] > 0.0
    lateral_excess = torch.clamp(
        torch.abs(foot_velocity_local[..., 1]) - lateral_velocity_tolerance,
        min=0.0,
    )
    per_foot_penalty = torch.square(lateral_excess / lateral_velocity_scale)
    per_foot_penalty *= (single_support_swing & stepping_forward).float()
    return torch.max(per_foot_penalty, dim=-1).values


def large_stationary_kick_attempt_candidate(
    foot_position_forward,
    foot_velocity_forward_relative_to_base,
    feet_contact,
    base_planar_speed,
    minimum_forward_position,
    minimum_forward_speed,
    maximum_base_speed,
):
    """Detect a large, fast single-support swing while the base stays slow."""
    if foot_position_forward.shape[-1] != 2:
        raise ValueError("foot position must end with two feet")
    if (
        foot_velocity_forward_relative_to_base.shape
        != foot_position_forward.shape
    ):
        raise ValueError("foot velocity must match foot position")
    if feet_contact.shape != foot_position_forward.shape:
        raise ValueError("feet_contact must match foot position")
    if base_planar_speed.shape != foot_position_forward.shape[:-1]:
        raise ValueError("base planar speed must match the batch dimensions")
    if minimum_forward_position < 0.0:
        raise ValueError("minimum forward position must be non-negative")
    if minimum_forward_speed < 0.0 or maximum_base_speed < 0.0:
        raise ValueError("speed thresholds must be non-negative")

    single_support_swing = (~feet_contact) & feet_contact.flip(-1)
    per_foot_candidate = (
        (foot_position_forward >= minimum_forward_position)
        & (
            foot_velocity_forward_relative_to_base
            >= minimum_forward_speed
        )
        & single_support_swing
        & (base_planar_speed <= maximum_base_speed).unsqueeze(-1)
    )
    return torch.any(per_foot_candidate, dim=-1)


def update_kick_attempt_outcome(
    candidate,
    new_valid_kick,
    pending,
    previous_candidate,
    deadline_step,
    current_step,
    outcome_window_steps,
):
    """Advance the per-environment kick-attempt outcome state by one step."""
    expected_shape = candidate.shape
    for value, name in (
        (new_valid_kick, "new_valid_kick"),
        (pending, "pending"),
        (previous_candidate, "previous_candidate"),
        (deadline_step, "deadline_step"),
        (current_step, "current_step"),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must match candidate")
    if outcome_window_steps <= 0:
        raise ValueError("outcome_window_steps must be positive")

    next_pending = pending & ~new_valid_kick
    failed = next_pending & (current_step >= deadline_step)
    next_pending = next_pending & ~failed

    rising_edge = candidate & ~previous_candidate
    new_attempt = rising_edge & ~next_pending & ~failed & ~new_valid_kick
    next_pending = next_pending | new_attempt
    next_deadline = deadline_step.clone()
    next_deadline[new_attempt] = (
        current_step[new_attempt] + int(outcome_window_steps)
    )
    return next_pending, candidate.clone(), next_deadline, failed
