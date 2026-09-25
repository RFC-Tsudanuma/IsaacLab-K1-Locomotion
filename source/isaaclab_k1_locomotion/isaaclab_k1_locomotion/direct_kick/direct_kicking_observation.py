# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/direct_kicking_observation.py
"""Observation contract shared by DirectKicking and its policy model."""

from typing import Sequence

import torch


LOCOMOTION_OBSERVATION_SIZE = 47
HORIZON_TOKEN_SIZE = 21
BELIEF_STATUS_SIZE = 3
TARGET_OBSERVATION_SIZE = 2
NON_FORECAST_OBSERVATION_SIZE = (
    LOCOMOTION_OBSERVATION_SIZE
    + BELIEF_STATUS_SIZE
    + TARGET_OBSERVATION_SIZE
)

# State order defines the rows/columns of the row-major 4x4 covariance.
RELATIVE_X_INDEX = 0
RELATIVE_Y_INDEX = 1
RELATIVE_VX_INDEX = 2
RELATIVE_VY_INDEX = 3
COVARIANCE_START = 4
COVARIANCE_END = 20
NORMALIZED_HORIZON_INDEX = 20


def expected_direct_kicking_observation_size(horizon_count: int) -> int:
    if horizon_count <= 0:
        raise ValueError("horizon_count must be positive")
    return NON_FORECAST_OBSERVATION_SIZE + HORIZON_TOKEN_SIZE * horizon_count


def build_horizon_tokens(
    relative_state: torch.Tensor,
    covariance: torch.Tensor,
    horizons_s: Sequence[float],
    valid: torch.Tensor,
    invalid_covariance: torch.Tensor,
) -> torch.Tensor:
    """Pack fixed future beliefs as ``[x, y, vx, vy, covariance.flatten(row-major), t]``.

    ``t`` is the nominal policy-relative horizon normalized by the largest
    configured horizon.  It intentionally excludes perception latency: latency
    is already included when the filter forecast is generated.
    """
    if relative_state.ndim != 3 or relative_state.shape[-1] != 4:
        raise ValueError("relative_state must have shape (batch, horizons, 4)")
    if covariance.shape != relative_state.shape[:2] + (4, 4):
        raise ValueError("covariance must have shape (batch, horizons, 4, 4)")
    if valid.shape != relative_state.shape[:1]:
        raise ValueError("valid must have shape (batch,)")
    if invalid_covariance.shape != (4, 4):
        raise ValueError("invalid_covariance must have shape (4, 4)")

    horizon_values = tuple(float(value) for value in horizons_s)
    if not horizon_values or horizon_values[-1] <= 0.0:
        raise ValueError("horizons_s must end at a positive horizon")
    horizons = torch.as_tensor(
        horizon_values,
        device=relative_state.device,
        dtype=relative_state.dtype,
    )
    if horizons.ndim != 1 or horizons.numel() != relative_state.shape[1]:
        raise ValueError("horizons_s must match the horizon dimension")

    normalized_horizon = (horizons / horizons[-1]).view(1, -1, 1)
    normalized_horizon = normalized_horizon.expand(relative_state.shape[0], -1, -1)
    tokens = torch.cat(
        (
            relative_state,
            covariance.flatten(start_dim=-2),
            normalized_horizon,
        ),
        dim=-1,
    )

    invalid_tokens = torch.zeros_like(tokens)
    invalid_tokens[:, :, COVARIANCE_START:COVARIANCE_END] = invalid_covariance.flatten()
    invalid_tokens[:, :, NORMALIZED_HORIZON_INDEX] = normalized_horizon.squeeze(-1)
    return torch.where(valid.view(-1, 1, 1), tokens, invalid_tokens)
