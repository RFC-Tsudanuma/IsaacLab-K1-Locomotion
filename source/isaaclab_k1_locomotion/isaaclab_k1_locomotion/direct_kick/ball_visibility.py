# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/ball_visibility.py
"""Visibility helpers for robot-relative ball observations."""

import torch


def horizontal_fov_mask(local_xy: torch.Tensor, fov_yaw: float) -> torch.Tensor:
    """Return whether each local XY point lies inside the horizontal FOV."""
    if local_xy.ndim != 2 or local_xy.shape[-1] != 2:
        raise ValueError("local_xy must have shape (batch, 2)")
    yaw = torch.atan2(local_xy[:, 1], local_xy[:, 0])
    return torch.abs(yaw) < 0.5 * float(fov_yaw)
