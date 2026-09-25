# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/ball_trajectory.py
"""Trajectory construction for DirectKicking ball resets."""

import torch


def build_ball_trajectory(
    spawn_distance,
    spawn_bearing,
    closest_approach_offset,
    base_speed,
    incoming,
):
    """Build local-frame spawn positions and velocities around the robot.

    The signed closest-approach offset prevents every incoming trajectory from
    passing through the robot center.  ``incoming`` selects whether velocity
    points toward or away from the robot; spawn position is determined only by
    the supplied distance and bearing.  All returned velocity directions have
    unit length before applying the sampled speed.
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
