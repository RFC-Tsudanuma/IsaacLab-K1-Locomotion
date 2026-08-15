"""Moving-ball reset and material events for the likelihood task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .ball_trajectory import build_ball_trajectory

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_VISION_FOV_YAW_RAD = 3.49065850
_VISION_MIN_DISTANCE_M = 0.05
_VISION_MAX_DISTANCE_M = 6.0


def _validated_range(
    values: tuple[float, float],
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    lower = float(values[0])
    upper = float(values[1])
    if lower > upper:
        raise ValueError(f"{name} must be ordered")
    if positive and lower <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and lower < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return lower, upper


def _sample_uniform(
    value_range: tuple[float, float],
    count: int,
    device: str | torch.device,
) -> torch.Tensor:
    lower, upper = value_range
    return lower + (upper - lower) * torch.rand(count, device=device)


def reset_moving_ball_trajectory(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_radius: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    speed_range_mps: tuple[float, float] = (0.0, 1.0),
    incoming_probability: float = 0.5,
    incoming_spawn_distance_range_m: tuple[float, float] = (1.5, 3.0),
    outgoing_spawn_distance_range_m: tuple[float, float] = (1.5, 3.0),
    closest_approach_offset_range_m: tuple[float, float] = (-0.25, 0.25),
    spawn_bearing_range_rad: tuple[float, float] = (-0.87266463, 0.87266463),
) -> None:
    """Reset balls on source-compatible incoming or outgoing trajectories.

    Spawn positions and velocities are sampled in the reset robot's yaw frame.
    The vertical position uses the local flat-environment origin, and the
    initial angular velocity is the no-slip spin for the sampled XY velocity.
    """
    if len(env_ids) == 0:
        return

    speed_range = _validated_range(
        speed_range_mps,
        "speed_range_mps",
        non_negative=True,
    )
    incoming_distance_range = _validated_range(
        incoming_spawn_distance_range_m,
        "incoming_spawn_distance_range_m",
        positive=True,
    )
    outgoing_distance_range = _validated_range(
        outgoing_spawn_distance_range_m,
        "outgoing_spawn_distance_range_m",
        positive=True,
    )
    offset_range = _validated_range(
        closest_approach_offset_range_m,
        "closest_approach_offset_range_m",
    )
    bearing_range = _validated_range(
        spawn_bearing_range_rad,
        "spawn_bearing_range_rad",
    )
    incoming_probability = float(incoming_probability)
    if not 0.0 <= incoming_probability <= 1.0:
        raise ValueError("incoming_probability must be in [0, 1]")

    minimum_spawn_distance = min(
        incoming_distance_range[0],
        outgoing_distance_range[0],
    )
    maximum_offset = max(abs(value) for value in offset_range)
    if maximum_offset >= minimum_spawn_distance:
        raise ValueError(
            "closest_approach_offset_range_m must stay inside the minimum "
            "spawn distance"
        )

    half_fov = 0.5 * _VISION_FOV_YAW_RAD
    if bearing_range[0] <= -half_fov or bearing_range[1] >= half_fov:
        raise ValueError("spawn_bearing_range_rad must stay inside vision fov_yaw")
    maximum_spawn_distance = max(
        incoming_distance_range[1],
        outgoing_distance_range[1],
    )
    if minimum_spawn_distance <= _VISION_MIN_DISTANCE_M:
        raise ValueError("spawn distance must be greater than vision min_distance")
    if maximum_spawn_distance >= _VISION_MAX_DISTANCE_M:
        raise ValueError("spawn distance must be less than vision max_distance")

    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    count = len(env_ids)

    incoming = torch.rand(count, device=env.device) < incoming_probability
    incoming_distance = _sample_uniform(
        incoming_distance_range,
        count,
        env.device,
    )
    outgoing_distance = _sample_uniform(
        outgoing_distance_range,
        count,
        env.device,
    )
    spawn_distance = torch.where(
        incoming,
        incoming_distance,
        outgoing_distance,
    )
    spawn_bearing = _sample_uniform(bearing_range, count, env.device)
    closest_approach_offset = _sample_uniform(offset_range, count, env.device)
    base_speed = _sample_uniform(speed_range, count, env.device)
    local_spawn_xy, local_velocity_xy = build_ball_trajectory(
        spawn_distance,
        spawn_bearing,
        closest_approach_offset,
        base_speed,
        incoming,
    )

    robot_pos_w = robot.data.root_pos_w[env_ids]
    robot_quat_w = robot.data.root_quat_w[env_ids]
    qw, qx, qy, qz = (
        robot_quat_w[:, 0],
        robot_quat_w[:, 1],
        robot_quat_w[:, 2],
        robot_quat_w[:, 3],
    )
    robot_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy.square() + qz.square()),
    )
    cosine = torch.cos(robot_yaw)
    sine = torch.sin(robot_yaw)

    world_spawn_xy = torch.stack(
        (
            cosine * local_spawn_xy[:, 0] - sine * local_spawn_xy[:, 1],
            sine * local_spawn_xy[:, 0] + cosine * local_spawn_xy[:, 1],
        ),
        dim=-1,
    )
    world_velocity_xy = torch.stack(
        (
            cosine * local_velocity_xy[:, 0] - sine * local_velocity_xy[:, 1],
            sine * local_velocity_xy[:, 0] + cosine * local_velocity_xy[:, 1],
        ),
        dim=-1,
    )

    state = ball.data.default_root_state[env_ids].clone()
    state[:, :2] = robot_pos_w[:, :2] + world_spawn_xy
    state[:, 2] = env.scene.env_origins[env_ids, 2] + ball_radius
    state[:, 3:7] = state.new_tensor((1.0, 0.0, 0.0, 0.0))
    state[:, 7:] = 0.0
    state[:, 7:9] = world_velocity_xy
    state[:, 10] = -world_velocity_xy[:, 1] / ball_radius
    state[:, 11] = world_velocity_xy[:, 0] / ball_radius
    ball.write_root_state_to_sim(state, env_ids=env_ids)


class RandomizeBallFriction(ManagerTermBase):
    """Assign one continuous startup friction sample to each ball instance."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        _validated_range(
            cfg.params["friction_range"],
            "friction_range",
            positive=True,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del asset_cfg
        friction_range = _validated_range(
            friction_range,
            "friction_range",
            positive=True,
        )
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids.numel() == 0:
            return

        materials = self.asset.root_physx_view.get_material_properties()
        friction = _sample_uniform(
            friction_range,
            len(env_ids),
            materials.device,
        ).to(dtype=materials.dtype)
        materials[env_ids, :, 0] = friction.unsqueeze(-1)
        materials[env_ids, :, 1] = friction.unsqueeze(-1)
        self.asset.root_physx_view.set_material_properties(materials, env_ids)
