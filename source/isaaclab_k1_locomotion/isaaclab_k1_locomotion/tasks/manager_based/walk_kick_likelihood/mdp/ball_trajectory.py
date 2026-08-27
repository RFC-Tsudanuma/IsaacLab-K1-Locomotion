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


def build_incoming_trajectory_near_robot(
    path_length: torch.Tensor,
    approach_heading: torch.Tensor,
    closest_approach_radius: torch.Tensor,
    closest_approach_side: torch.Tensor,
    speed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a straight incoming path whose closest point is around the robot.

    The closest point is sampled in the reset robot's XY frame.  A uniformly
    sampled heading plus a random normal-side sign covers front, rear, left,
    right, and diagonal closest points instead of limiting misses to the
    lateral axis.  ``path_length`` is the distance travelled before reaching
    that point, so every non-zero-speed sample initially approaches the robot.
    """
    direction = torch.stack(
        (torch.cos(approach_heading), torch.sin(approach_heading)),
        dim=-1,
    )
    normal = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    closest_point = (
        closest_approach_radius.unsqueeze(-1)
        * closest_approach_side.unsqueeze(-1)
        * normal
    )
    spawn_position = closest_point - path_length.unsqueeze(-1) * direction
    velocity = speed.unsqueeze(-1) * direction
    return spawn_position, velocity
