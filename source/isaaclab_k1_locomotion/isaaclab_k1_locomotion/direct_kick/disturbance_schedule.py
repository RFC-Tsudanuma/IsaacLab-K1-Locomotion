# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/disturbance_schedule.py
import math

import torch


def sample_interval_steps(interval_range_s, control_dt, count, device):
    """Sample positive disturbance intervals and convert them to control steps."""
    if len(interval_range_s) != 2:
        raise ValueError("interval_range_s must contain two values")
    lower = float(interval_range_s[0])
    upper = float(interval_range_s[1])
    if lower <= 0.0 or lower > upper:
        raise ValueError("interval_range_s must be positive and ordered")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if lower == upper:
        return torch.full(
            (count,),
            math.ceil(lower / control_dt),
            device=device,
            dtype=torch.long,
        )

    interval_s = lower + (upper - lower) * torch.rand(count, device=device)
    return torch.ceil(interval_s / control_dt).to(dtype=torch.long)
