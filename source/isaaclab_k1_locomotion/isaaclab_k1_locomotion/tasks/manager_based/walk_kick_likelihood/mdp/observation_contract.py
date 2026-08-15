"""Observation contract shared by the likelihood task and its policy model."""

from typing import Sequence

import torch


LOCOMOTION_OBSERVATION_SIZE = 47
HORIZON_TOKEN_SIZE = 6
BELIEF_VELOCITY_SIZE = 2
BELIEF_STATUS_SIZE = 3
TARGET_OBSERVATION_SIZE = 2
NON_FORECAST_OBSERVATION_SIZE = (
    LOCOMOTION_OBSERVATION_SIZE
    + BELIEF_VELOCITY_SIZE
    + BELIEF_STATUS_SIZE
    + TARGET_OBSERVATION_SIZE
)

# Per-horizon token layout. The first five entries preserve the previous
# DirectKicking belief layout; normalized time is appended explicitly.
RELATIVE_X_INDEX = 0
RELATIVE_Y_INDEX = 1
LOG_STD_X_INDEX = 2
LOG_STD_Y_INDEX = 3
CORRELATION_INDEX = 4
NORMALIZED_HORIZON_INDEX = 5


def expected_direct_kicking_observation_size(horizon_count: int) -> int:
    if horizon_count <= 0:
        raise ValueError("horizon_count must be positive")
    return NON_FORECAST_OBSERVATION_SIZE + HORIZON_TOKEN_SIZE * horizon_count


def build_horizon_tokens(
    relative_position: torch.Tensor,
    log_std: torch.Tensor,
    correlation: torch.Tensor,
    horizons_s: Sequence[float],
    valid: torch.Tensor,
    invalid_log_std: float,
) -> torch.Tensor:
    """Pack fixed future beliefs as ``[x, y, log_std_x, log_std_y, rho, t]``.

    ``t`` is the nominal policy-relative horizon normalized by the largest
    configured horizon. It intentionally excludes perception latency: latency
    is already included when the filter forecast is generated.
    """
    if relative_position.ndim != 3 or relative_position.shape[-1] != 2:
        raise ValueError("relative_position must have shape (batch, horizons, 2)")
    if log_std.shape != relative_position.shape:
        raise ValueError("log_std must match relative_position")
    if correlation.shape != relative_position.shape[:2]:
        raise ValueError("correlation must have shape (batch, horizons)")
    if valid.shape != relative_position.shape[:1]:
        raise ValueError("valid must have shape (batch,)")

    horizon_values = tuple(float(value) for value in horizons_s)
    if not horizon_values or horizon_values[-1] <= 0.0:
        raise ValueError("horizons_s must end at a positive horizon")
    horizons = torch.as_tensor(
        horizon_values,
        device=relative_position.device,
        dtype=relative_position.dtype,
    )
    if horizons.ndim != 1 or horizons.numel() != relative_position.shape[1]:
        raise ValueError("horizons_s must match the horizon dimension")

    normalized_horizon = (horizons / horizons[-1]).view(1, -1, 1)
    normalized_horizon = normalized_horizon.expand(relative_position.shape[0], -1, -1)
    tokens = torch.cat(
        (
            relative_position,
            log_std,
            correlation.unsqueeze(-1),
            normalized_horizon,
        ),
        dim=-1,
    )

    invalid_tokens = torch.zeros_like(tokens)
    invalid_tokens[:, :, LOG_STD_X_INDEX : LOG_STD_Y_INDEX + 1] = float(
        invalid_log_std
    )
    invalid_tokens[:, :, NORMALIZED_HORIZON_INDEX] = normalized_horizon.squeeze(-1)
    return torch.where(valid.view(-1, 1, 1), tokens, invalid_tokens)
