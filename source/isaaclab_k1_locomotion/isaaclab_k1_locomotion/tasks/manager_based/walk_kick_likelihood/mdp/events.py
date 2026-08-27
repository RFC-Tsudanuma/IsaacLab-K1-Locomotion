"""Moving-ball reset and material events for the likelihood task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .ball_trajectory import (
    build_ball_trajectory,
    build_incoming_trajectory_near_robot,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_VISION_FOV_YAW_RAD = 3.49065850
_VISION_MIN_DISTANCE_M = 0.05
_VISION_MAX_DISTANCE_M = 10.0

_INITIAL_BALL_SPEED_ATTR = "_walk_kick_likelihood_initial_ball_speed"
_BALL_INCOMING_ATTR = "_walk_kick_likelihood_ball_incoming"
_BALL_SPEED_CAP_ATTR = "_walk_kick_likelihood_ball_speed_cap"
_CLOSEST_APPROACH_RADIUS_CAP_ATTR = (
    "_walk_kick_likelihood_closest_approach_radius_cap"
)
_BALL_RESET_METADATA_VALID_ATTR = "_walk_kick_likelihood_ball_reset_metadata_valid"


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


def _episode_metadata_buffer(
    env: ManagerBasedRLEnv,
    attribute_name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = (env.scene.num_envs,)
    buffer = getattr(env, attribute_name, None)
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.shape != shape
        or buffer.device != torch.device(env.device)
        or buffer.dtype != dtype
    ):
        buffer = torch.zeros(shape, device=env.device, dtype=dtype)
        setattr(env, attribute_name, buffer)
    return buffer


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
    nominal_ttc_range_s: tuple[float, float] | None = None,
) -> None:
    """Reset balls on source-compatible incoming or outgoing trajectories.

    Spawn positions and velocities are sampled in the reset robot's yaw frame.
    The vertical position uses the local flat-environment origin, and the
    initial angular velocity is the no-slip spin for the sampled XY velocity.
    When ``nominal_ttc_range_s`` is set, speed is derived from the path length
    to closest approach divided by the sampled nominal time-to-contact, then
    clipped to ``speed_range_mps``.
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
    ttc_range = (
        None
        if nominal_ttc_range_s is None
        else _validated_range(
            nominal_ttc_range_s,
            "nominal_ttc_range_s",
            positive=True,
        )
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
    if ttc_range is None:
        base_speed = _sample_uniform(speed_range, count, env.device)
    else:
        nominal_ttc = _sample_uniform(ttc_range, count, env.device)
        path_length = torch.sqrt(
            torch.clamp(
                spawn_distance.square() - closest_approach_offset.square(),
                min=0.0,
            )
        )
        base_speed = torch.clamp(
            path_length / nominal_ttc,
            min=speed_range[0],
            max=speed_range[1],
        )

    # Keep the sampled difficulty attached to each episode.  CurriculumManager runs
    # before the next reset event, so it can evaluate the episode that just ended
    # without reconstructing the reset distribution from the final ball state.
    metadata_env_ids = env_ids.to(device=env.device, dtype=torch.long)
    speed_buffer = _episode_metadata_buffer(
        env, _INITIAL_BALL_SPEED_ATTR, base_speed.dtype
    )
    incoming_buffer = _episode_metadata_buffer(env, _BALL_INCOMING_ATTR, torch.bool)
    cap_buffer = _episode_metadata_buffer(env, _BALL_SPEED_CAP_ATTR, base_speed.dtype)
    valid_buffer = _episode_metadata_buffer(
        env, _BALL_RESET_METADATA_VALID_ATTR, torch.bool
    )
    speed_buffer[metadata_env_ids] = base_speed
    incoming_buffer[metadata_env_ids] = incoming
    cap_buffer[metadata_env_ids] = speed_range[1]
    valid_buffer[metadata_env_ids] = True

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


def reset_incoming_ball_near_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_radius: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    speed_range_mps: tuple[float, float] = (0.0, 0.0),
    path_length_range_m: tuple[float, float] = (1.5, 3.0),
    closest_approach_radius_range_m: tuple[float, float] = (0.0, 0.0),
    approach_heading_range_rad: tuple[float, float] = (-3.14159265, 3.14159265),
    nominal_ttc_range_s: tuple[float, float] | None = (1.5, 5.0),
) -> None:
    """Reset a ball on an incoming path near the robot in every XY direction.

    The path's true closest point is sampled uniformly by area from an annulus
    in the reset robot frame.  Its polar direction therefore spans front,
    rear, left, right, and diagonals.  The ball never receives an initially
    outgoing velocity; after an unsuccessful pass it may naturally move away
    once it has crossed the closest point.
    """
    if len(env_ids) == 0:
        return

    speed_range = _validated_range(
        speed_range_mps,
        "speed_range_mps",
        non_negative=True,
    )
    path_range = _validated_range(
        path_length_range_m,
        "path_length_range_m",
        positive=True,
    )
    radius_range = _validated_range(
        closest_approach_radius_range_m,
        "closest_approach_radius_range_m",
        non_negative=True,
    )
    heading_range = _validated_range(
        approach_heading_range_rad,
        "approach_heading_range_rad",
    )
    ttc_range = (
        None
        if nominal_ttc_range_s is None
        else _validated_range(
            nominal_ttc_range_s,
            "nominal_ttc_range_s",
            positive=True,
        )
    )
    maximum_spawn_distance = (
        path_range[1] ** 2 + radius_range[1] ** 2
    ) ** 0.5
    if maximum_spawn_distance >= _VISION_MAX_DISTANCE_M:
        raise ValueError("incoming trajectory spawn distance must stay inside vision range")

    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    count = len(env_ids)

    path_length = _sample_uniform(path_range, count, env.device)
    approach_heading = _sample_uniform(heading_range, count, env.device)
    radius_squared = _sample_uniform(
        (radius_range[0] ** 2, radius_range[1] ** 2),
        count,
        env.device,
    )
    closest_approach_radius = torch.sqrt(radius_squared)
    closest_approach_side = torch.where(
        torch.rand(count, device=env.device) < 0.5,
        -torch.ones(count, device=env.device),
        torch.ones(count, device=env.device),
    )
    if ttc_range is None:
        speed = _sample_uniform(speed_range, count, env.device)
    else:
        nominal_ttc = _sample_uniform(ttc_range, count, env.device)
        speed = torch.clamp(
            path_length / nominal_ttc,
            min=speed_range[0],
            max=speed_range[1],
        )

    metadata_env_ids = env_ids.to(device=env.device, dtype=torch.long)
    speed_buffer = _episode_metadata_buffer(
        env, _INITIAL_BALL_SPEED_ATTR, speed.dtype
    )
    incoming_buffer = _episode_metadata_buffer(env, _BALL_INCOMING_ATTR, torch.bool)
    speed_cap_buffer = _episode_metadata_buffer(
        env, _BALL_SPEED_CAP_ATTR, speed.dtype
    )
    radius_cap_buffer = _episode_metadata_buffer(
        env, _CLOSEST_APPROACH_RADIUS_CAP_ATTR, speed.dtype
    )
    valid_buffer = _episode_metadata_buffer(
        env, _BALL_RESET_METADATA_VALID_ATTR, torch.bool
    )
    speed_buffer[metadata_env_ids] = speed
    incoming_buffer[metadata_env_ids] = True
    speed_cap_buffer[metadata_env_ids] = speed_range[1]
    radius_cap_buffer[metadata_env_ids] = radius_range[1]
    valid_buffer[metadata_env_ids] = True

    local_spawn_xy, local_velocity_xy = build_incoming_trajectory_near_robot(
        path_length,
        approach_heading,
        closest_approach_radius,
        closest_approach_side,
        speed,
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


class ApplyBallRollingResistance(ManagerTermBase):
    """Apply domain-randomized pseudo rolling resistance on flat ground.

    Only the world-frame horizontal angular velocity is damped.  The torque is
    cleared while the ball is above the flat-ground contact tolerance, so this
    term does not introduce air drag or damp vertical spin.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self._ball_radius = float(cfg.params["ball_radius"])
        self._ball_mass = float(cfg.params["ball_mass"])
        self._contact_tolerance = float(cfg.params["contact_tolerance_m"])
        decay_rate_range = _validated_range(
            cfg.params["decay_rate_range_s_inv"],
            "decay_rate_range_s_inv",
            positive=True,
        )
        if self._ball_radius <= 0.0:
            raise ValueError("ball_radius must be positive")
        if self._ball_mass <= 0.0:
            raise ValueError("ball_mass must be positive")
        if self._contact_tolerance < 0.0:
            raise ValueError("contact_tolerance_m must be non-negative")

        self._decay_rate = _sample_uniform(
            decay_rate_range,
            env.scene.num_envs,
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        ball_radius: float,
        ball_mass: float,
        decay_rate_range_s_inv: tuple[float, float],
        contact_tolerance_m: float,
    ) -> None:
        del (
            asset_cfg,
            ball_radius,
            ball_mass,
            decay_rate_range_s_inv,
            contact_tolerance_m,
        )
        if env_ids is None:
            env_ids = torch.arange(
                env.scene.num_envs,
                device=env.device,
                dtype=torch.long,
            )
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        angular_velocity = self.asset.data.root_ang_vel_w[env_ids]
        ground_height = env.scene.env_origins[env_ids, 2]
        ball_height = self.asset.data.root_pos_w[env_ids, 2]
        on_flat_ground = ball_height <= (
            ground_height + self._ball_radius + self._contact_tolerance
        )

        # For a solid sphere, I=2/5*m*r^2.  With no slip, I+m*r^2 gives
        # 7/5*m*r^2, making lambda the ideal linear-speed decay rate.
        effective_inertia = 1.4 * self._ball_mass * self._ball_radius**2
        damping = effective_inertia * self._decay_rate[env_ids]
        torque = torch.zeros(
            (env_ids.numel(), 1, 3),
            device=angular_velocity.device,
            dtype=angular_velocity.dtype,
        )
        torque[:, 0, :2] = (
            -damping.unsqueeze(-1)
            * angular_velocity[:, :2]
            * on_flat_ground.unsqueeze(-1)
        )
        force = torch.zeros_like(torque)
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force,
            torques=torque,
            env_ids=env_ids,
            is_global=True,
        )


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
