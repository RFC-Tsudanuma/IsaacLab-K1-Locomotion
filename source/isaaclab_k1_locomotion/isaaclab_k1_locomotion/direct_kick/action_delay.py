# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/action_delay.py
import math

import torch


def action_delay_step_range(delay_range_s, physics_dt):
    """Convert an exactly representable time range to inclusive physics steps."""
    if len(delay_range_s) != 2:
        raise ValueError("delay_range_s must contain two values")
    lower = float(delay_range_s[0])
    upper = float(delay_range_s[1])
    if lower <= 0.0 or lower > upper:
        raise ValueError("delay_range_s must be positive and ordered")
    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive")

    minimum_steps = int(round(lower / physics_dt))
    maximum_steps = int(round(upper / physics_dt))
    tolerance = max(1.0e-12, physics_dt * 1.0e-6)
    if not math.isclose(minimum_steps * physics_dt, lower, abs_tol=tolerance):
        raise ValueError("delay_range_s lower bound must align with physics_dt")
    if not math.isclose(maximum_steps * physics_dt, upper, abs_tol=tolerance):
        raise ValueError("delay_range_s upper bound must align with physics_dt")
    return minimum_steps, maximum_steps


def sample_action_delay_steps(delay_step_range, count, device):
    """Sample an inclusive integer delay range independently per environment."""
    if len(delay_step_range) != 2:
        raise ValueError("delay_step_range must contain two values")
    minimum_steps = int(delay_step_range[0])
    maximum_steps = int(delay_step_range[1])
    if minimum_steps <= 0 or minimum_steps > maximum_steps:
        raise ValueError("delay_step_range must be positive and ordered")
    if count < 0:
        raise ValueError("count must be non-negative")
    return torch.randint(
        minimum_steps,
        maximum_steps + 1,
        (count,),
        device=device,
        dtype=torch.long,
    )


def delayed_targets_for_substep(
    target_history,
    history_cursor,
    delay_steps,
    substep_index,
    decimation,
):
    """Select the most recent target old enough for each environment."""
    if target_history.ndim != 3:
        raise ValueError("target_history must have shape [env, history, dof]")
    if delay_steps.ndim != 1 or delay_steps.shape[0] != target_history.shape[0]:
        raise ValueError("delay_steps must have one value per environment")
    if decimation <= 0:
        raise ValueError("decimation must be positive")
    if not 0 <= substep_index < decimation:
        raise ValueError("substep_index must be inside the control step")
    if target_history.shape[1] <= 0:
        raise ValueError("target_history must not be empty")

    history_age = torch.div(
        delay_steps - substep_index + decimation - 1,
        decimation,
        rounding_mode="floor",
    ).clamp_min(0)
    history_indices = (history_cursor - history_age) % target_history.shape[1]
    gather_indices = history_indices.view(-1, 1, 1).expand(
        -1,
        1,
        target_history.shape[2],
    )
    return torch.gather(target_history, 1, gather_indices).squeeze(1)
