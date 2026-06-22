"""Command-time-series sampling for velocity tracking data collection.

Mixes constant / step / ramp / sinusoidal / random-walk patterns so the policy
sees a wide distribution of command shapes.
"""

from __future__ import annotations

import numpy as np
import torch


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
    n = num_envs // 5
    cmd[0:n] = _constant(n, T, vx_range, vy_range, wz_range, device)
    cmd[n : 2 * n] = _step(n, T, vx_range, vy_range, wz_range, device)
    cmd[2 * n : 3 * n] = _ramp(n, T, vx_range, vy_range, wz_range, device)
    cmd[3 * n : 4 * n] = _sinusoidal(n, T, dt, vx_range, vy_range, wz_range, device)
    cmd[4 * n :] = _random_walk(num_envs - 4 * n, T, dt, vx_range, vy_range, wz_range, device)
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
