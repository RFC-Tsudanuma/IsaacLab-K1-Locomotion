# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/direct_kicking_symmetry.py
"""Sagittal mirror contract and ambiguity weighting for DirectKicking."""

import torch

from .direct_kicking_observation import (
    BELIEF_STATUS_SIZE,
    COVARIANCE_START,
    COVARIANCE_END,
    RELATIVE_VY_INDEX,
    HORIZON_TOKEN_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    RELATIVE_Y_INDEX,
    expected_direct_kicking_observation_size,
)


DIRECT_KICKING_ACTION_SIZE = 12
LEG_ACTION_SIZE = 6
LEG_MIRROR_SIGNS = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)

PROJECTED_GRAVITY_SLICE = slice(0, 3)
BASE_ANGULAR_VELOCITY_SLICE = slice(3, 6)
COMMAND_SLICE = slice(6, 9)
GAIT_PHASE_SLICE = slice(9, 11)
JOINT_POSITION_SLICE = slice(11, 23)
JOINT_VELOCITY_SLICE = slice(23, 35)
PREVIOUS_ACTION_SLICE = slice(35, 47)


def mirror_leg_actions(actions: torch.Tensor) -> torch.Tensor:
    """Swap K1's legs and mirror roll/yaw action coordinates."""
    if actions.ndim == 0 or actions.shape[-1] != DIRECT_KICKING_ACTION_SIZE:
        raise ValueError("DirectKicking actions must end with 12 values")

    signs = actions.new_tensor(LEG_MIRROR_SIGNS)
    left = actions[..., :LEG_ACTION_SIZE]
    right = actions[..., LEG_ACTION_SIZE:]
    return torch.cat((right * signs, left * signs), dim=-1)


def mirror_direct_kicking_observation(
    observation: torch.Tensor,
    horizon_count: int,
) -> torch.Tensor:
    """Reflect a DirectKicking observation across the robot's sagittal plane."""
    expected_size = expected_direct_kicking_observation_size(horizon_count)
    if observation.ndim == 0 or observation.shape[-1] != expected_size:
        raise ValueError(
            "DirectKicking observation must end with {} values".format(
                expected_size
            )
        )

    mirrored = observation.clone()
    mirrored[..., PROJECTED_GRAVITY_SLICE] = (
        observation[..., PROJECTED_GRAVITY_SLICE]
        * observation.new_tensor((1.0, -1.0, 1.0))
    )
    # Angular velocity is an axial vector: reflection uses det(S) * S.
    mirrored[..., BASE_ANGULAR_VELOCITY_SLICE] = (
        observation[..., BASE_ANGULAR_VELOCITY_SLICE]
        * observation.new_tensor((-1.0, 1.0, -1.0))
    )
    mirrored[..., COMMAND_SLICE] = (
        observation[..., COMMAND_SLICE]
        * observation.new_tensor((1.0, -1.0, -1.0))
    )
    # A half-cycle shift swaps the left/right swing phases.
    mirrored[..., GAIT_PHASE_SLICE] = -observation[..., GAIT_PHASE_SLICE]
    for leg_slice in (
        JOINT_POSITION_SLICE,
        JOINT_VELOCITY_SLICE,
        PREVIOUS_ACTION_SLICE,
    ):
        mirrored[..., leg_slice] = mirror_leg_actions(observation[..., leg_slice])

    forecast_start = LOCOMOTION_OBSERVATION_SIZE
    forecast_end = forecast_start + horizon_count * HORIZON_TOKEN_SIZE
    forecast_shape = observation.shape[:-1] + (horizon_count, HORIZON_TOKEN_SIZE)
    forecast = observation[..., forecast_start:forecast_end].reshape(forecast_shape)
    mirrored_forecast = forecast.clone()
    mirrored_forecast[..., RELATIVE_Y_INDEX] = -forecast[..., RELATIVE_Y_INDEX]
    mirrored_forecast[..., RELATIVE_VY_INDEX] = -forecast[..., RELATIVE_VY_INDEX]
    # Reflection acts on the complete state covariance as M P M^T.
    state_signs = observation.new_tensor((1., -1., 1., -1.))
    covariance_signs = (state_signs[:, None] * state_signs[None, :]).flatten()
    mirrored_forecast[..., COVARIANCE_START:COVARIANCE_END] = (
        forecast[..., COVARIANCE_START:COVARIANCE_END] * covariance_signs
    )
    mirrored[..., forecast_start:forecast_end] = mirrored_forecast.reshape(
        observation.shape[:-1] + (horizon_count * HORIZON_TOKEN_SIZE,)
    )

    target_start = forecast_end + BELIEF_STATUS_SIZE
    mirrored[..., target_start : target_start + 2] = (
        observation[..., target_start : target_start + 2]
        * observation.new_tensor((1.0, -1.0))
    )
    return mirrored


def kick_feasibility_ambiguity_weight(
    observation: torch.Tensor,
    horizon_count: int,
    nominal_strike_point_m,
    cost_gap_m,
    ball_position_observation_scale: float,
) -> torch.Tensor:
    """Return a smooth confidence weight derived from left/right kick costs.

    Each foot's cost is the minimum forecast distance to its mirrored nominal
    strike point. Equal costs are intentionally left unconstrained.
    """
    expected_size = expected_direct_kicking_observation_size(horizon_count)
    if observation.ndim == 0 or observation.shape[-1] != expected_size:
        raise ValueError(
            "DirectKicking observation must end with {} values".format(
                expected_size
            )
        )
    if len(nominal_strike_point_m) != 2:
        raise ValueError("nominal_strike_point_m must contain [x, abs(y)]")
    if len(cost_gap_m) != 2:
        raise ValueError("cost_gap_m must contain [zero_weight, full_weight]")

    strike_x = float(nominal_strike_point_m[0])
    strike_y = float(nominal_strike_point_m[1])
    gap_low = float(cost_gap_m[0])
    gap_high = float(cost_gap_m[1])
    observation_scale = float(ball_position_observation_scale)
    if strike_x <= 0.0 or strike_y <= 0.0:
        raise ValueError("nominal strike coordinates must be positive")
    if gap_low < 0.0 or gap_high <= gap_low:
        raise ValueError("cost gap bounds must satisfy 0 <= low < high")
    if observation_scale <= 0.0:
        raise ValueError("ball position observation scale must be positive")

    forecast_start = LOCOMOTION_OBSERVATION_SIZE
    forecast_end = forecast_start + horizon_count * HORIZON_TOKEN_SIZE
    forecast = observation[..., forecast_start:forecast_end].reshape(
        observation.shape[:-1] + (horizon_count, HORIZON_TOKEN_SIZE)
    )
    forecast_position = forecast[..., :2]
    strike_point = observation.new_tensor((strike_x, strike_y)) * observation_scale
    left_point = strike_point
    right_point = strike_point * observation.new_tensor((1.0, -1.0))
    left_cost = torch.linalg.vector_norm(
        forecast_position - left_point,
        dim=-1,
    ).amin(dim=-1)
    right_cost = torch.linalg.vector_norm(
        forecast_position - right_point,
        dim=-1,
    ).amin(dim=-1)
    cost_gap = torch.abs(left_cost - right_cost)

    gap_low *= observation_scale
    gap_high *= observation_scale
    transition = torch.clamp(
        (cost_gap - gap_low) / (gap_high - gap_low),
        min=0.0,
        max=1.0,
    )
    smooth_weight = transition.square() * (3.0 - 2.0 * transition)

    valid_index = (
        forecast_end + BELIEF_STATUS_SIZE - 1
    )
    belief_valid = torch.clamp(observation[..., valid_index], min=0.0, max=1.0)
    return smooth_weight * belief_valid


def weighted_mirror_consistency_loss(
    action_mean: torch.Tensor,
    mirrored_observation_action_mean: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute E[w(o) * ||mu(M(o)) - M(mu(o))||^2]."""
    if action_mean.shape != mirrored_observation_action_mean.shape:
        raise ValueError("original and mirrored action means must have equal shape")
    if action_mean.ndim == 0 or action_mean.shape[-1] != DIRECT_KICKING_ACTION_SIZE:
        raise ValueError("DirectKicking action means must end with 12 values")
    if weight.shape != action_mean.shape[:-1]:
        raise ValueError("weight must match the action mean leading dimensions")

    expected_mirrored_mean = mirror_leg_actions(action_mean)
    per_sample_error = torch.square(
        mirrored_observation_action_mean - expected_mirrored_mean
    ).sum(dim=-1)
    return torch.mean(weight * per_sample_error)
