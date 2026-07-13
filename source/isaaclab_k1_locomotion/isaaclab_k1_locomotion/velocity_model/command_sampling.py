"""Command-time-series sampling for velocity tracking data collection.

Mixes constant / step / ramp / sinusoidal / piecewise-constant / random-walk patterns
so the policy sees a wide distribution of command shapes.

The ``piecewise`` pattern issues a fresh random command every ~0.25-1.25 s (discrete
jumps), mimicking a path planner that re-issues velocity targets frequently. It is given
the largest share because the planning use case switches commands far more often than the
flat env's native 10 s resampling.
"""

from __future__ import annotations

import numpy as np
import torch

# Pattern order is the contract shared with eval_velocity_predictor.py (label indices).
PATTERN_NAMES = ["constant", "step", "ramp", "sinusoid", "random_walk", "piecewise"]


def pattern_partition(num_envs: int) -> list[tuple[str, int]]:
    """Split ``num_envs`` across patterns. ``piecewise`` (frequent switching) gets ~3/8.

    Returns a list of (pattern_name, count) in PATTERN_NAMES order, summing to num_envs.
    """
    base = num_envs // 8
    counts = {name: base for name in PATTERN_NAMES}
    counts["piecewise"] = num_envs - base * (len(PATTERN_NAMES) - 1)
    return [(name, counts[name]) for name in PATTERN_NAMES]


def sample_commands(
    num_envs: int,
    T: int,
    dt: float,
    device: torch.device | str,
    vx_range: tuple[float, float] = (-1.0, 1.0),
    vy_range: tuple[float, float] = (-0.5, 0.5),
    wz_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Return (num_envs, T, 3) command tensor on device."""
    cmd = torch.zeros(num_envs, T, 3, device=device)
    start = 0
    for name, cnt in pattern_partition(num_envs):
        if cnt <= 0:
            continue
        sl = slice(start, start + cnt)
        if name == "constant":
            cmd[sl] = _constant(cnt, T, vx_range, vy_range, wz_range, device)
        elif name == "step":
            cmd[sl] = _step(cnt, T, vx_range, vy_range, wz_range, device)
        elif name == "ramp":
            cmd[sl] = _ramp(cnt, T, vx_range, vy_range, wz_range, device)
        elif name == "sinusoid":
            cmd[sl] = _sinusoidal(cnt, T, dt, vx_range, vy_range, wz_range, device)
        elif name == "random_walk":
            cmd[sl] = _random_walk(cnt, T, dt, vx_range, vy_range, wz_range, device)
        elif name == "piecewise":
            cmd[sl] = _piecewise_const(cnt, T, dt, vx_range, vy_range, wz_range, device)
        else:
            raise ValueError(f"Unknown pattern: {name}")
        start += cnt
    return cmd


def _sample_uniform(n: int, vx_r, vy_r, wz_r, device) -> torch.Tensor:
    return torch.stack(
        [
            torch.empty(n, device=device).uniform_(*vx_r),
            torch.empty(n, device=device).uniform_(*vy_r),
            torch.empty(n, device=device).uniform_(*wz_r),
        ],
        dim=-1,
    )


def _constant(n, T, vx_r, vy_r, wz_r, device) -> torch.Tensor:
    v = _sample_uniform(n, vx_r, vy_r, wz_r, device)
    return v.unsqueeze(1).expand(-1, T, -1).clone()


def _step(n, T, vx_r, vy_r, wz_r, device) -> torch.Tensor:
    v1 = _sample_uniform(n, vx_r, vy_r, wz_r, device)
    v2 = _sample_uniform(n, vx_r, vy_r, wz_r, device)
    change_t = torch.randint(T // 4, max(T // 4 + 1, 3 * T // 4), (n,), device=device)
    t_idx = torch.arange(T, device=device).unsqueeze(0)
    mask = t_idx < change_t.unsqueeze(1)
    return torch.where(mask.unsqueeze(-1), v1.unsqueeze(1), v2.unsqueeze(1))


def _ramp(n, T, vx_r, vy_r, wz_r, device) -> torch.Tensor:
    v_start = _sample_uniform(n, vx_r, vy_r, wz_r, device)
    v_end = _sample_uniform(n, vx_r, vy_r, wz_r, device)
    alpha = torch.linspace(0.0, 1.0, T, device=device).view(1, T, 1)
    return v_start.unsqueeze(1) * (1.0 - alpha) + v_end.unsqueeze(1) * alpha


def _sinusoidal(n, T, dt, vx_r, vy_r, wz_r, device) -> torch.Tensor:
    t = torch.arange(T, device=device).float() * dt
    freq = torch.empty(n, 3, device=device).uniform_(0.2, 1.5)
    phase = torch.empty(n, 3, device=device).uniform_(0.0, 2.0 * np.pi)
    amp = torch.stack(
        [
            torch.empty(n, device=device).uniform_(0.0, vx_r[1]),
            torch.empty(n, device=device).uniform_(0.0, vy_r[1]),
            torch.empty(n, device=device).uniform_(0.0, wz_r[1]),
        ],
        dim=-1,
    )
    omega = 2.0 * np.pi * freq.unsqueeze(1)
    return amp.unsqueeze(1) * torch.sin(omega * t.view(1, T, 1) + phase.unsqueeze(1))


def _random_walk(n, T, dt, vx_r, vy_r, wz_r, device, sigma: float = 0.5) -> torch.Tensor:
    cmd = torch.zeros(n, T, 3, device=device)
    cmd[:, 0] = _sample_uniform(n, vx_r, vy_r, wz_r, device)
    tau = 1.0
    alpha = float(np.exp(-dt / tau))
    noise_scale = sigma * float(np.sqrt(1.0 - alpha**2))
    bounds = torch.tensor([vx_r[1], vy_r[1], wz_r[1]], device=device)
    for t in range(1, T):
        noise = torch.randn(n, 3, device=device) * noise_scale
        cmd[:, t] = torch.clamp(alpha * cmd[:, t - 1] + noise, -bounds, bounds)
    return cmd


def _piecewise_const(
    n, T, dt, vx_r, vy_r, wz_r, device,
    min_hold_s: float = 0.25, max_hold_s: float = 1.25,
) -> torch.Tensor:
    """Frequent discrete switching: hold a random command for a random duration then jump.

    Each env independently re-samples its target every ``[min_hold_s, max_hold_s]`` seconds,
    matching a path planner that issues new velocity goals frequently.
    """
    min_h = max(1, int(round(min_hold_s / dt)))
    max_h = max(min_h, int(round(max_hold_s / dt)))
    # Enough segments to always cover T even if every hold is the minimum length.
    n_seg = T // min_h + 2
    vals = torch.stack(
        [
            torch.empty(n, n_seg, device=device).uniform_(*vx_r),
            torch.empty(n, n_seg, device=device).uniform_(*vy_r),
            torch.empty(n, n_seg, device=device).uniform_(*wz_r),
        ],
        dim=-1,
    )  # (n, n_seg, 3)
    lengths = torch.randint(min_h, max_h + 1, (n, n_seg), device=device)
    bounds = torch.cumsum(lengths, dim=1)                       # (n, n_seg) segment end (exclusive)
    t_idx = torch.arange(T, device=device).view(1, T, 1)        # (1, T, 1)
    # segment index for each t = number of segment boundaries <= t
    seg_id = (t_idx >= bounds.unsqueeze(1)).sum(dim=-1).clamp(max=n_seg - 1)  # (n, T)
    return torch.gather(vals, 1, seg_id.unsqueeze(-1).expand(n, T, 3))
