"""Trajectory construction for moving-ball resets."""

from __future__ import annotations

import torch


def build_ball_trajectory(
    spawn_distance: torch.Tensor,
    spawn_bearing: torch.Tensor,
    closest_approach_offset: torch.Tensor,
    base_speed: torch.Tensor,
    incoming: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build robot-local spawn positions and velocities.

    This is the trajectory contract from ``DirectKicking`` commit ``3af2acc``.
    The signed closest-approach offset keeps incoming trajectories from always
    passing through the robot origin.  ``incoming`` changes only the radial
    direction; the supplied distance and bearing determine the spawn point.
    """
    radial = torch.stack(
        (torch.cos(spawn_bearing), torch.sin(spawn_bearing)),
        dim=-1,
    )
    tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=-1)

    offset_ratio = closest_approach_offset / spawn_distance
    radial_magnitude = torch.sqrt(
        torch.clamp(1.0 - torch.square(offset_ratio), min=0.0)
    )
    signed_radial_magnitude = torch.where(
        incoming,
        -radial_magnitude,
        radial_magnitude,
    )
    direction = (
        signed_radial_magnitude.unsqueeze(-1) * radial
        + offset_ratio.unsqueeze(-1) * tangent
    )

    spawn_position = spawn_distance.unsqueeze(-1) * radial
    velocity = base_speed.unsqueeze(-1) * direction
    return spawn_position, velocity
