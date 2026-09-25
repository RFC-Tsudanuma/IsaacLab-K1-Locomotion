# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/ego_motion.py
"""Planar ego-motion forecasting utilities."""

from typing import Tuple

import torch


def forecast_constant_body_twist_variance(
    offsets: torch.Tensor,
    position_noise_std: float,
    position_bias_std: float,
    velocity_noise_std: float,
    velocity_bias_std: float,
    velocity_drift_std: float,
    yaw_noise_std: float,
    yaw_bias_std: float,
    yaw_rate_noise_std: float,
    yaw_rate_bias_std: float,
    yaw_rate_drift_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate base-position and base-yaw variance over forecast offsets.

    The velocity and yaw-rate random walks use continuous-time integration:
    their integrated variance grows with ``t**3 / 3``.  The returned values
    are scalar variances per batch/horizon and assume independent XY axes.
    """
    if offsets.ndim != 2:
        raise ValueError("offsets must have shape (batch, horizons)")
    values = (
        position_noise_std,
        position_bias_std,
        velocity_noise_std,
        velocity_bias_std,
        velocity_drift_std,
        yaw_noise_std,
        yaw_bias_std,
        yaw_rate_noise_std,
        yaw_rate_bias_std,
        yaw_rate_drift_std,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("noise standard deviations must be non-negative")

    time = offsets
    position_variance = (
        position_noise_std**2
        + position_bias_std**2
        + time.square() * (velocity_noise_std**2 + velocity_bias_std**2)
        + time.pow(3) * velocity_drift_std**2 / 3.0
    )
    yaw_variance = (
        yaw_noise_std**2
        + yaw_bias_std**2
        + time.square() * (yaw_rate_noise_std**2 + yaw_rate_bias_std**2)
        + time.pow(3) * yaw_rate_drift_std**2 / 3.0
    )
    return position_variance, yaw_variance


def relative_velocity_from_world(
    ball_velocity_world: torch.Tensor,
    base_velocity_xy: torch.Tensor,
    base_yaw: torch.Tensor,
    yaw_rate: torch.Tensor,
    relative_position_xy: torch.Tensor,
) -> torch.Tensor:
    """Convert world ball velocity to velocity relative to a planar base.

    The rotating-frame term is included so a stationary world object has the
    correct apparent velocity when the base turns.
    """
    if ball_velocity_world.ndim != 2 or ball_velocity_world.shape[-1] != 2:
        raise ValueError("ball_velocity_world must have shape (batch, 2)")
    batch = ball_velocity_world.shape[0]
    if base_velocity_xy.shape != (batch, 2):
        raise ValueError("base_velocity_xy must have shape (batch, 2)")
    if base_yaw.shape != (batch,):
        raise ValueError("base_yaw must have shape (batch,)")
    if yaw_rate.shape != (batch,):
        raise ValueError("yaw_rate must have shape (batch,)")
    if relative_position_xy.shape != (batch, 2):
        raise ValueError("relative_position_xy must have shape (batch, 2)")

    cosine = torch.cos(base_yaw)
    sine = torch.sin(base_yaw)
    ball_velocity_local = torch.stack(
        (
            cosine * ball_velocity_world[:, 0]
            + sine * ball_velocity_world[:, 1],
            -sine * ball_velocity_world[:, 0]
            + cosine * ball_velocity_world[:, 1],
        ),
        dim=-1,
    )
    omega_cross_position = torch.stack(
        (
            -yaw_rate * relative_position_xy[:, 1],
            yaw_rate * relative_position_xy[:, 0],
        ),
        dim=-1,
    )
    return ball_velocity_local - base_velocity_xy - omega_cross_position


def forecast_constant_body_twist(
    base_position: torch.Tensor,
    base_yaw: torch.Tensor,
    body_velocity_xy: torch.Tensor,
    yaw_rate: torch.Tensor,
    offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forecast planar base pose under a constant body-frame twist.

    Args:
        base_position: Current world position with shape ``(batch, 2)``.
        base_yaw: Current world yaw with shape ``(batch,)``.
        body_velocity_xy: Current planar velocity in the base-yaw frame,
            shape ``(batch, 2)``.
        yaw_rate: Constant yaw rate in rad/s, shape ``(batch,)``.
        offsets: Non-negative forecast times, shape ``(batch, horizons)``.

    Returns:
        Forecast world positions ``(batch, horizons, 2)`` and yaws
        ``(batch, horizons)``.

    The closed-form integration avoids a small-yaw-rate division singularity
    by using the first-order limit when the rate is near zero.
    """
    if base_position.ndim != 2 or base_position.shape[-1] != 2:
        raise ValueError("base_position must have shape (batch, 2)")
    batch = base_position.shape[0]
    if base_yaw.shape != (batch,):
        raise ValueError("base_yaw must have shape (batch,)")
    if body_velocity_xy.shape != (batch, 2):
        raise ValueError("body_velocity_xy must have shape (batch, 2)")
    if yaw_rate.shape != (batch,):
        raise ValueError("yaw_rate must have shape (batch,)")
    if offsets.ndim != 2 or offsets.shape[0] != batch:
        raise ValueError("offsets must have shape (batch, horizons)")
    if torch.any(offsets < 0.0):
        raise ValueError("offsets must be non-negative")

    time = offsets
    angle = yaw_rate.unsqueeze(-1) * time
    rate = yaw_rate.unsqueeze(-1)
    small_rate = torch.abs(rate) < 1.0e-6

    sin_over_rate = torch.where(
        small_rate,
        time,
        torch.sin(angle) / rate,
    )
    cos_minus_one_over_rate = torch.where(
        small_rate,
        torch.zeros_like(time),
        (torch.cos(angle) - 1.0) / rate,
    )
    one_minus_cos_over_rate = -cos_minus_one_over_rate

    velocity_x = body_velocity_xy[:, 0].unsqueeze(-1)
    velocity_y = body_velocity_xy[:, 1].unsqueeze(-1)
    displacement_body_x = (
        sin_over_rate * velocity_x
        + cos_minus_one_over_rate * velocity_y
    )
    displacement_body_y = (
        one_minus_cos_over_rate * velocity_x
        + sin_over_rate * velocity_y
    )

    cosine = torch.cos(base_yaw).unsqueeze(-1)
    sine = torch.sin(base_yaw).unsqueeze(-1)
    displacement_world = torch.stack(
        (
            cosine * displacement_body_x - sine * displacement_body_y,
            sine * displacement_body_x + cosine * displacement_body_y,
        ),
        dim=-1,
    )
    position = base_position.unsqueeze(1) + displacement_world
    yaw = base_yaw.unsqueeze(-1) + angle
    return position, yaw


def forecast_ego_state_covariance(
    offsets: torch.Tensor,
    position_noise_std: float,
    position_bias_std: float,
    velocity_noise_std: float,
    velocity_bias_std: float,
    velocity_drift_std: float,
    yaw_noise_std: float,
    yaw_bias_std: float,
    yaw_rate_noise_std: float,
    yaw_rate_bias_std: float,
    yaw_rate_drift_std: float,
) -> torch.Tensor:
    """Covariance of [position_local_x/y, yaw, body_vx/vy, yaw_rate].

    Extend the existing independent-axis integrated random-walk approximation;
    retain its position/yaw marginals and include pose/rate cross-covariances.
    This is not an exact covariance integration of a turning trajectory.
    """
    position_variance, yaw_variance = forecast_constant_body_twist_variance(
        offsets, position_noise_std, position_bias_std,
        velocity_noise_std, velocity_bias_std, velocity_drift_std,
        yaw_noise_std, yaw_bias_std,
        yaw_rate_noise_std, yaw_rate_bias_std, yaw_rate_drift_std,
    )
    velocity_prior = velocity_noise_std**2 + velocity_bias_std**2
    rate_prior = yaw_rate_noise_std**2 + yaw_rate_bias_std**2
    velocity_variance = velocity_prior + offsets * velocity_drift_std**2
    rate_variance = rate_prior + offsets * yaw_rate_drift_std**2
    position_velocity_covariance = (
        offsets * velocity_prior + offsets.square() * velocity_drift_std**2 / 2.0
    )
    yaw_rate_covariance = (
        offsets * rate_prior + offsets.square() * yaw_rate_drift_std**2 / 2.0
    )
    covariance = offsets.new_zeros(offsets.shape + (6, 6))
    for axis in (0, 1):
        covariance[..., axis, axis] = position_variance
        covariance[..., axis + 3, axis + 3] = velocity_variance
        covariance[..., axis, axis + 3] = position_velocity_covariance
        covariance[..., axis + 3, axis] = position_velocity_covariance
    covariance[..., 2, 2] = yaw_variance
    covariance[..., 5, 5] = rate_variance
    covariance[..., 2, 5] = yaw_rate_covariance
    covariance[..., 5, 2] = yaw_rate_covariance
    return covariance


def relative_state_and_covariance(
    ball_state_world: torch.Tensor,
    ball_covariance_world: torch.Tensor,
    base_position_world: torch.Tensor,
    base_yaw: torch.Tensor,
    base_velocity_xy: torch.Tensor,
    yaw_rate: torch.Tensor,
    ego_covariance: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform full [x,y,vx,vy] beliefs into the moving ego frame.

    Inputs have batch/horizon leading dimensions. Ball and ego errors are
    assumed independent, as in the previous position-only uncertainty model.
    Ego yaw/rate uncertainty is propagated by a first-order Jacobian.
    """
    cosine, sine = torch.cos(base_yaw), torch.sin(base_yaw)
    rotation = torch.stack((cosine, sine, -sine, cosine), dim=-1).reshape(
        base_yaw.shape + (2, 2)
    )
    turn = rotation.new_tensor([[0., -1.], [1., 0.]])
    position = (rotation @ (ball_state_world[..., :2] - base_position_world).unsqueeze(-1)).squeeze(-1)
    ball_velocity_local = (rotation @ ball_state_world[..., 2:].unsqueeze(-1)).squeeze(-1)
    turned_position = (turn @ position.unsqueeze(-1)).squeeze(-1)
    velocity = ball_velocity_local - base_velocity_xy - yaw_rate.unsqueeze(-1) * turned_position

    ball_jacobian = ball_covariance_world.new_zeros(ball_covariance_world.shape)
    ball_jacobian[..., :2, :2] = rotation
    ball_jacobian[..., 2:, 2:] = rotation
    ball_jacobian[..., 2:, :2] = -yaw_rate[..., None, None] * (turn @ rotation)
    ego_jacobian = ball_state_world.new_zeros(ball_state_world.shape[:-1] + (4, 6))
    identity = torch.eye(2, device=rotation.device, dtype=rotation.dtype)
    ego_jacobian[..., :2, :2] = -identity
    ego_jacobian[..., :2, 2] = -turned_position
    ego_jacobian[..., 2:, :2] = yaw_rate[..., None, None] * turn
    ego_jacobian[..., 2:, 2] = -(turn @ (velocity + base_velocity_xy).unsqueeze(-1)).squeeze(-1)
    ego_jacobian[..., 2:, 3:5] = -identity
    ego_jacobian[..., 2:, 5] = -turned_position
    covariance = (
        ball_jacobian @ ball_covariance_world @ ball_jacobian.transpose(-1, -2)
        + ego_jacobian @ ego_covariance @ ego_jacobian.transpose(-1, -2)
    )
    return torch.cat((position, velocity), dim=-1), covariance
