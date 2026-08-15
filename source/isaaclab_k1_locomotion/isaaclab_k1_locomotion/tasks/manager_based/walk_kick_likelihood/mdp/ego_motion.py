"""Planar ego-motion forecasting utilities for the DirectKicking contract."""

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
    """Approximate base-position and base-yaw variance over forecast offsets."""
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
    """Convert world ball velocity to velocity relative to a rotating planar base."""
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
            cosine * ball_velocity_world[:, 0] + sine * ball_velocity_world[:, 1],
            -sine * ball_velocity_world[:, 0] + cosine * ball_velocity_world[:, 1],
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
    """Forecast planar base pose under a constant body-frame twist."""
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
    sin_over_rate = torch.where(small_rate, time, torch.sin(angle) / rate)
    cos_minus_one_over_rate = torch.where(
        small_rate,
        torch.zeros_like(time),
        (torch.cos(angle) - 1.0) / rate,
    )
    one_minus_cos_over_rate = -cos_minus_one_over_rate

    velocity_x = body_velocity_xy[:, 0].unsqueeze(-1)
    velocity_y = body_velocity_xy[:, 1].unsqueeze(-1)
    displacement_body_x = sin_over_rate * velocity_x + cos_minus_one_over_rate * velocity_y
    displacement_body_y = one_minus_cos_over_rate * velocity_x + sin_over_rate * velocity_y

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
