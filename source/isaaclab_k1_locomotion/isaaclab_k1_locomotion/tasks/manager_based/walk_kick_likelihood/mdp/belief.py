"""CVKF ball-belief observation for the walk-kick likelihood task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .ball_kalman_filter import (
    BatchedConstantVelocityKalmanFilter,
    distance_scaled_measurement_std,
)
from .ego_motion import (
    forecast_constant_body_twist,
    forecast_constant_body_twist_variance,
    relative_velocity_from_world,
)
from .observation_contract import (
    BELIEF_STATUS_SIZE,
    BELIEF_VELOCITY_SIZE,
    HORIZON_TOKEN_SIZE,
    build_horizon_tokens,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PREDICTION_HORIZONS_S = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    2.50,
    3.00,
)
CVKF_BELIEF_OBSERVATION_SIZE = (
    len(PREDICTION_HORIZONS_S) * HORIZON_TOKEN_SIZE
    + BELIEF_VELOCITY_SIZE
    + BELIEF_STATUS_SIZE
)

# DirectKicking commit 3af2acc configuration.
_STD_REFERENCE_M = 0.01
_LOG_STD_CLIP = (-4.0, 4.0)
_MEASUREMENT_AGE_NORMALIZER_S = 0.5
_UNCERTAINTY_LOG_STD_SCALE = 0.25
_PROCESS_ACCELERATION_STD_MPS2 = 0.8
_INITIAL_VELOCITY_STD_MPS = 0.5
_PROCESS_NOISE_SCALE_RANGE = (0.8, 1.2)
_MEASUREMENT_NOISE_SCALE_RANGE = (0.8, 1.2)
_NIS_THRESHOLD = 13.82
_MAX_MISSING_TIME_S = 0.5
_MEASUREMENT_NOISE_STD_RANGE_M = (0.005, 0.015)
_MEASUREMENT_NOISE_REFERENCE_DISTANCE_M = 1.0
_MEASUREMENT_NOISE_MAX_SCALE = 8.0
_CAMERA_FPS_RANGE = (25.0, 30.0)
_CAMERA_FPS_JITTER = 0.15
_LATENCY_RANGE_S = (0.0, 0.06)
_DROPOUT_BURST_PROBABILITY = 0.05
_DROPOUT_BURST_FRAMES = (1, 3)
_OUTLIER_PROBABILITY = 0.02
_OUTLIER_DISTANCE_RANGE_M = (0.10, 0.30)
_VISION_FOV_YAW_RAD = 3.49065850
_VISION_MIN_DISTANCE_M = 0.05
_VISION_MAX_DISTANCE_M = 6.0
_BASE_LINEAR_VELOCITY_NOISE_STD_MPS = 0.10
_BASE_ANGULAR_VELOCITY_NOISE_STD_RPS = 0.15
_EGO_VELOCITY_BIAS_STD_MPS = 0.03
_EGO_VELOCITY_DRIFT_STD_MPS_PER_SQRT_S = 0.01
_EGO_YAW_RATE_BIAS_STD_RPS = 0.05
_EGO_YAW_RATE_DRIFT_STD_RPS_PER_SQRT_S = 0.02
_EGO_POSITION_NOISE_STD_M = 0.0
_EGO_POSITION_BIAS_STD_M = 0.0
_EGO_YAW_NOISE_STD_RAD = 0.0
_EGO_YAW_BIAS_STD_RAD = 0.0

_STATE_ATTRIBUTE = "_walk_kick_likelihood_belief_state"


class CVKFBeliefState:
    """Own the per-environment delayed perception and CVKF state.

    The state advances at most once for each ``common_step_counter`` value.
    Rows whose ``episode_length_buf`` transitions to zero are reset without
    disturbing the remaining environments.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        robot_name: str,
        ball_name: str,
    ) -> None:
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.dt = float(env.step_dt)
        self.robot_name = robot_name
        self.ball_name = ball_name
        self.last_step = int(env.common_step_counter)
        self.last_ego_advance_step = self.last_step
        self.last_episode_length = env.episode_length_buf.clone()
        self.ego_generation = 0
        self.ego_cache_key: tuple[int, int] | None = None
        self.observation_dirty = True

        self.filter = BatchedConstantVelocityKalmanFilter(
            self.num_envs,
            self.dt,
            self.device,
        )
        self.measurement_noise_std = torch.zeros(self.num_envs, device=self.device)
        self.filter_measurement_std = torch.zeros_like(self.measurement_noise_std)
        self.process_acceleration_std = torch.zeros_like(self.measurement_noise_std)
        self.camera_fps = torch.zeros_like(self.measurement_noise_std)
        self.camera_timer = torch.zeros_like(self.measurement_noise_std)
        self.last_measurement_age = torch.zeros_like(self.measurement_noise_std)
        self.measurement_updated = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self.belief_valid = torch.zeros_like(self.measurement_updated)
        self.perception_just_reset = torch.zeros_like(self.measurement_updated)
        self.pending_reset = torch.zeros_like(self.measurement_updated)
        self.dropout_remaining = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.perception_latency_steps = torch.zeros_like(self.dropout_remaining)
        self.perception_latency = torch.zeros_like(self.measurement_noise_std)
        self.perception_history_valid_steps = torch.zeros_like(self.dropout_remaining)

        self.ego_velocity_bias = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.ego_velocity_drift = torch.zeros_like(self.ego_velocity_bias)
        self.ego_yaw_rate_bias = torch.zeros_like(self.measurement_noise_std)
        self.ego_yaw_rate_drift = torch.zeros_like(self.measurement_noise_std)
        self.ego_position_bias = torch.zeros_like(self.ego_velocity_bias)
        self.ego_yaw_bias = torch.zeros_like(self.measurement_noise_std)
        self.observed_base_position = torch.zeros_like(self.ego_velocity_bias)
        self.observed_base_yaw = torch.zeros_like(self.measurement_noise_std)
        self.observed_base_lin_vel_yaw = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        self.observed_base_ang_vel = torch.zeros_like(self.observed_base_lin_vel_yaw)

        max_latency_steps = int(math.ceil(_LATENCY_RANGE_S[1] / self.dt))
        self.perception_history_length = max_latency_steps + 2
        self.ball_position_history = torch.zeros(
            self.num_envs,
            self.perception_history_length,
            2,
            device=self.device,
        )
        self.base_position_history = torch.zeros_like(self.ball_position_history)
        self.base_yaw_history = torch.zeros(
            self.num_envs,
            self.perception_history_length,
            device=self.device,
        )
        self.perception_history_cursor = 0
        self.cached_observation = torch.zeros(
            self.num_envs,
            CVKF_BELIEF_OBSERVATION_SIZE,
            device=self.device,
        )

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset(env, all_env_ids)
        self._sample_observed_velocities(env)
        self.ego_cache_key = (self.last_step, self.ego_generation)
        self.cached_observation = self._build_observation(env)
        self.observation_dirty = False

    def request_reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Defer row reinitialization until reset state has reached data buffers."""
        if env_ids is None:
            self.pending_reset[:] = True
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.pending_reset[ids] = True

    def observe(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Advance if needed, apply partial resets, and return the cached 83D belief."""
        self.prepare_ego_observation(env)
        step = int(env.common_step_counter)

        if step != self.last_step:
            # prepare_ego_observation resets rows before actor-prefix terms are
            # read. perception_just_reset prevents those rows from consuming
            # the preceding episode while the other rows advance normally.
            self._advance(env)
            self.last_step = step
            self.observation_dirty = True

        if self.observation_dirty:
            self.cached_observation = self._build_observation(env)
            self.observation_dirty = False

        # Reset observations leave the perception state active for the first
        # following control step. _advance clears this too when a reset and a
        # normal transition share the same common-step change.
        self.perception_just_reset.zero_()

        # OnPolicyRunner may randomize episode lengths after its initial
        # observation while common_step_counter is still zero. Keep this
        # snapshot current without advancing the perception state so a later
        # transition back to zero remains detectable as a partial reset.
        self.last_episode_length.copy_(env.episode_length_buf)

        return self.cached_observation

    def prepare_ego_observation(self, env: ManagerBasedRLEnv) -> None:
        """Refresh the shared noisy ego sample at most once per control step."""
        step = int(env.common_step_counter)
        episode_length = env.episode_length_buf
        newly_reset = (episode_length == 0) & (self.last_episode_length != 0)
        reset_mask = newly_reset | self.pending_reset
        if reset_mask.any():
            reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
            self._reset(env, reset_ids)
            self.pending_reset[reset_ids] = False

        if step != self.last_ego_advance_step:
            self._update_ego_motion_sensor_noise()
            self._update_observed_base_pose(env)
            self.last_ego_advance_step = step
            self.observation_dirty = True

        cache_key = (step, self.ego_generation)
        if self.ego_cache_key != cache_key:
            self._sample_observed_velocities(env)
            self.ego_cache_key = cache_key
            self.observation_dirty = True

        # Keep the reset detector synchronized even when a prefix/target term
        # is the first state consumer for this observation pass.
        self.last_episode_length.copy_(episode_length)

    def _reset(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        if count == 0:
            return

        self.ego_velocity_bias[env_ids] = (
            torch.randn(count, 2, device=self.device) * _EGO_VELOCITY_BIAS_STD_MPS
        )
        self.ego_velocity_drift[env_ids] = 0.0
        self.ego_yaw_rate_bias[env_ids] = (
            torch.randn(count, device=self.device) * _EGO_YAW_RATE_BIAS_STD_RPS
        )
        self.ego_yaw_rate_drift[env_ids] = 0.0
        self.ego_position_bias[env_ids] = (
            torch.randn(count, 2, device=self.device) * _EGO_POSITION_BIAS_STD_M
        )
        self.ego_yaw_bias[env_ids] = (
            torch.randn(count, device=self.device) * _EGO_YAW_BIAS_STD_RAD
        )

        self.measurement_noise_std[env_ids] = self._sample_uniform(
            _MEASUREMENT_NOISE_STD_RANGE_M,
            count,
        )
        r_scale = self._sample_uniform(_MEASUREMENT_NOISE_SCALE_RANGE, count)
        self.filter_measurement_std[env_ids] = self.measurement_noise_std[env_ids] * r_scale
        q_scale = self._sample_uniform(_PROCESS_NOISE_SCALE_RANGE, count)
        # WalkKick does not expose the source task's sampled rolling/ground
        # friction state. Preserve the target physics and apply only the
        # independent Q calibration draw here.
        self.process_acceleration_std[env_ids] = _PROCESS_ACCELERATION_STD_MPS2 * q_scale
        self.camera_fps[env_ids] = self._sample_uniform(_CAMERA_FPS_RANGE, count)

        sampled_latency = self._sample_uniform(_LATENCY_RANGE_S, count)
        latency_steps = torch.round(sampled_latency / self.dt).to(torch.long)
        latency_steps.clamp_(max=self.perception_history_length - 2)
        self.perception_latency_steps[env_ids] = latency_steps
        self.perception_latency[env_ids] = latency_steps.to(torch.float) * self.dt
        self.camera_timer[env_ids] = self.perception_latency[env_ids] + (
            torch.rand(count, device=self.device) / self.camera_fps[env_ids]
        )
        self.last_measurement_age[env_ids] = _MAX_MISSING_TIME_S
        self.measurement_updated[env_ids] = False
        self.belief_valid[env_ids] = False
        self.perception_just_reset[env_ids] = True
        self.dropout_remaining[env_ids] = 0
        self.perception_history_valid_steps[env_ids] = 0

        ball = env.scene[self.ball_name]
        robot = env.scene[self.robot_name]
        ball_xy = ball.data.root_pos_w[env_ids, :2]
        base_xy = robot.data.root_pos_w[env_ids, :2]
        base_yaw = _yaw_from_quaternion(robot.data.root_quat_w[env_ids])
        self.observed_base_position[env_ids] = (
            base_xy
            + self.ego_position_bias[env_ids]
            + torch.randn(count, 2, device=self.device) * _EGO_POSITION_NOISE_STD_M
        )
        self.observed_base_yaw[env_ids] = (
            base_yaw
            + self.ego_yaw_bias[env_ids]
            + torch.randn(count, device=self.device) * _EGO_YAW_NOISE_STD_RAD
        )
        self.ball_position_history[env_ids] = ball_xy.unsqueeze(1)
        self.base_position_history[env_ids] = self.observed_base_position[env_ids].unsqueeze(1)
        self.base_yaw_history[env_ids] = self.observed_base_yaw[env_ids].unsqueeze(1)
        self.filter.invalidate(env_ids)
        self.ego_generation += 1
        self.ego_cache_key = None
        self.observation_dirty = True

    def _advance(self, env: ManagerBasedRLEnv) -> None:
        just_reset = self.perception_just_reset.clone()
        perception_active = ~just_reset
        filter_active = self.filter.initialized & perception_active

        ball = env.scene[self.ball_name]
        self.perception_history_cursor = (
            self.perception_history_cursor + 1
        ) % self.perception_history_length
        cursor = self.perception_history_cursor
        self.ball_position_history[:, cursor] = ball.data.root_pos_w[:, :2]
        self.base_position_history[:, cursor] = self.observed_base_position
        self.base_yaw_history[:, cursor] = self.observed_base_yaw
        self.perception_history_valid_steps[perception_active] += 1

        self.filter.predict(self.process_acceleration_std, mask=filter_active)
        self.last_measurement_age[filter_active] += self.dt
        self.camera_timer[perception_active] -= self.dt
        history_ready = self.perception_history_valid_steps >= self.perception_latency_steps
        due = perception_active & history_ready & (self.camera_timer <= 0.0)
        due_ids = due.nonzero(as_tuple=False).flatten()
        self.measurement_updated.zero_()

        if due_ids.numel() > 0:
            jitter = 1.0 + _CAMERA_FPS_JITTER * (
                2.0 * torch.rand(due_ids.numel(), device=self.device) - 1.0
            )
            next_period = jitter / self.camera_fps[due_ids]
            self.camera_timer[due_ids] += next_period
            self.camera_timer[due_ids] = torch.clamp(
                self.camera_timer[due_ids],
                min=0.25 * self.dt,
            )

            dropout = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            continuing_dropout = self.dropout_remaining[due_ids] > 0
            continuing_ids = due_ids[continuing_dropout]
            dropout[continuing_ids] = True
            self.dropout_remaining[continuing_ids] -= 1

            eligible_ids = due_ids[~continuing_dropout]
            if eligible_ids.numel() > 0:
                starts_dropout = (
                    torch.rand(eligible_ids.numel(), device=self.device)
                    < _DROPOUT_BURST_PROBABILITY
                )
                start_ids = eligible_ids[starts_dropout]
                if start_ids.numel() > 0:
                    burst_lengths = torch.randint(
                        _DROPOUT_BURST_FRAMES[0],
                        _DROPOUT_BURST_FRAMES[1] + 1,
                        (start_ids.numel(),),
                        device=self.device,
                    )
                    dropout[start_ids] = True
                    self.dropout_remaining[start_ids] = burst_lengths - 1

            latency_steps = self.perception_latency_steps[due_ids]
            measurement_age = self.perception_latency[due_ids]
            history_index = (cursor - latency_steps) % self.perception_history_length
            captured_ball = self.ball_position_history[due_ids, history_index]
            captured_base = self.base_position_history[due_ids, history_index]
            captured_yaw = self.base_yaw_history[due_ids, history_index]
            captured_local = _world_point_to_local(captured_ball, captured_base, captured_yaw)

            distance = torch.norm(captured_local, dim=-1)
            visible_local = (
                _horizontal_fov_mask(captured_local, _VISION_FOV_YAW_RAD)
                & (distance > _VISION_MIN_DISTANCE_M)
                & (distance < _VISION_MAX_DISTANCE_M)
            )
            available_local = visible_local & ~dropout[due_ids]
            available_ids = due_ids[available_local]
            measurements = torch.zeros(self.num_envs, 2, device=self.device)
            update_mask = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )
            ages = torch.zeros(self.num_envs, device=self.device)
            frame_filter_measurement_std = self.filter_measurement_std.clone()

            if available_ids.numel() > 0:
                measured_local = captured_local[available_local].clone()
                sensor_measurement_std = distance_scaled_measurement_std(
                    self.measurement_noise_std[available_ids],
                    distance[available_local],
                    _MEASUREMENT_NOISE_REFERENCE_DISTANCE_M,
                    _MEASUREMENT_NOISE_MAX_SCALE,
                )
                frame_filter_measurement_std[available_ids] = (
                    distance_scaled_measurement_std(
                        self.filter_measurement_std[available_ids],
                        distance[available_local],
                        _MEASUREMENT_NOISE_REFERENCE_DISTANCE_M,
                        _MEASUREMENT_NOISE_MAX_SCALE,
                    )
                )
                measured_local += (
                    torch.randn_like(measured_local)
                    * sensor_measurement_std.unsqueeze(-1)
                )
                outlier_candidate = (
                    torch.rand(available_ids.numel(), device=self.device)
                    < _OUTLIER_PROBABILITY
                )
                initialized_track = self.filter.initialized[available_ids]
                rejected_initial_outlier = outlier_candidate & ~initialized_track
                outlier = outlier_candidate & initialized_track
                outlier_distance = self._sample_uniform(
                    _OUTLIER_DISTANCE_RANGE_M,
                    available_ids.numel(),
                )
                outlier_angle = 2.0 * math.pi * torch.rand(
                    available_ids.numel(),
                    device=self.device,
                )
                outlier_scale = outlier.float() * outlier_distance
                measured_local[:, 0] += outlier_scale * torch.cos(outlier_angle)
                measured_local[:, 1] += outlier_scale * torch.sin(outlier_angle)
                measurements[available_ids] = _local_point_to_world(
                    measured_local,
                    captured_base[available_local],
                    captured_yaw[available_local],
                )
                usable_ids = available_ids[~rejected_initial_outlier]
                update_mask[usable_ids] = True
                ages[usable_ids] = measurement_age[available_local][~rejected_initial_outlier]

            was_initialized = self.filter.initialized.clone()
            initializing = update_mask & ~was_initialized
            initializing_ids = initializing.nonzero(as_tuple=False).flatten()
            if initializing_ids.numel() > 0:
                velocity_std = torch.full(
                    (initializing_ids.numel(),),
                    _INITIAL_VELOCITY_STD_MPS,
                    device=self.device,
                )
                self.filter.reset(
                    initializing_ids,
                    measurements[initializing_ids],
                    frame_filter_measurement_std[initializing_ids],
                    velocity_std,
                )

            correcting = update_mask & was_initialized
            accepted = self.filter.update(
                measurements,
                correcting,
                frame_filter_measurement_std,
                nis_threshold=_NIS_THRESHOLD,
            )
            accepted |= initializing
            self.measurement_updated[:] = accepted
            self.last_measurement_age[accepted] = ages[accepted]

        self.belief_valid[:] = self.filter.initialized & (
            self.last_measurement_age <= _MAX_MISSING_TIME_S
        )
        self.perception_just_reset.zero_()

    def _update_ego_motion_sensor_noise(self) -> None:
        drift_scale = math.sqrt(self.dt)
        self.ego_velocity_drift += (
            torch.randn_like(self.ego_velocity_drift)
            * _EGO_VELOCITY_DRIFT_STD_MPS_PER_SQRT_S
            * drift_scale
        )
        self.ego_yaw_rate_drift += (
            torch.randn_like(self.ego_yaw_rate_drift)
            * _EGO_YAW_RATE_DRIFT_STD_RPS_PER_SQRT_S
            * drift_scale
        )

    def _update_observed_base_pose(self, env: ManagerBasedRLEnv) -> None:
        robot = env.scene[self.robot_name]
        true_base_position = robot.data.root_pos_w[:, :2]
        true_base_yaw = _yaw_from_quaternion(robot.data.root_quat_w)
        self.observed_base_position[:] = (
            true_base_position
            + self.ego_position_bias
            + torch.randn_like(true_base_position) * _EGO_POSITION_NOISE_STD_M
        )
        self.observed_base_yaw[:] = (
            true_base_yaw
            + self.ego_yaw_bias
            + torch.randn_like(true_base_yaw) * _EGO_YAW_NOISE_STD_RAD
        )

    def _sample_observed_velocities(self, env: ManagerBasedRLEnv) -> None:
        robot = env.scene[self.robot_name]
        rotation = _world_to_local_rotation(self.observed_base_yaw)
        base_velocity_yaw_xy = torch.matmul(
            rotation,
            robot.data.root_lin_vel_w[:, :2].unsqueeze(-1),
        ).squeeze(-1)
        base_velocity_yaw = torch.cat(
            (base_velocity_yaw_xy, robot.data.root_lin_vel_w[:, 2:3]),
            dim=-1,
        )
        self.observed_base_lin_vel_yaw[:] = (
            base_velocity_yaw
            + torch.randn_like(base_velocity_yaw) * _BASE_LINEAR_VELOCITY_NOISE_STD_MPS
        )
        self.observed_base_lin_vel_yaw[:, :2] += (
            self.ego_velocity_bias + self.ego_velocity_drift
        )
        self.observed_base_ang_vel[:] = (
            robot.data.root_ang_vel_b
            + torch.randn_like(robot.data.root_ang_vel_b)
            * _BASE_ANGULAR_VELOCITY_NOISE_STD_RPS
        )
        self.observed_base_ang_vel[:, 2] += (
            self.ego_yaw_rate_bias + self.ego_yaw_rate_drift
        )

    def _build_observation(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        horizons = torch.as_tensor(
            PREDICTION_HORIZONS_S,
            device=self.device,
            dtype=self.filter.state.dtype,
        )
        forecast_offsets = self.perception_latency.unsqueeze(1) + horizons.unsqueeze(0)
        means_world, covariance_world = self.filter.forecast_offsets(
            forecast_offsets,
            self.process_acceleration_std,
        )

        # The filter lives on the delayed measurement timeline, while the ego
        # pose is current. Thus only the ball forecast includes latency.
        ego_forecast_offsets = horizons.unsqueeze(0).expand(self.num_envs, -1)
        base_position_forecast, base_yaw_forecast = forecast_constant_body_twist(
            self.observed_base_position,
            self.observed_base_yaw,
            self.observed_base_lin_vel_yaw[:, :2],
            self.observed_base_ang_vel[:, 2],
            ego_forecast_offsets,
        )
        rotation = _world_to_local_rotation(base_yaw_forecast)
        relative_position = torch.matmul(
            rotation,
            (means_world[:, :, :2] - base_position_forecast).unsqueeze(-1),
        ).squeeze(-1)
        position_covariance = covariance_world[:, :, :2, :2]
        local_covariance = torch.matmul(
            torch.matmul(rotation, position_covariance),
            rotation.transpose(-1, -2),
        )
        ego_position_variance, ego_yaw_variance = forecast_constant_body_twist_variance(
            ego_forecast_offsets,
            _EGO_POSITION_NOISE_STD_M,
            _EGO_POSITION_BIAS_STD_M,
            _BASE_LINEAR_VELOCITY_NOISE_STD_MPS,
            _EGO_VELOCITY_BIAS_STD_MPS,
            _EGO_VELOCITY_DRIFT_STD_MPS_PER_SQRT_S,
            _EGO_YAW_NOISE_STD_RAD,
            _EGO_YAW_BIAS_STD_RAD,
            _BASE_ANGULAR_VELOCITY_NOISE_STD_RPS,
            _EGO_YAW_RATE_BIAS_STD_RPS,
            _EGO_YAW_RATE_DRIFT_STD_RPS_PER_SQRT_S,
        )
        local_covariance = local_covariance + torch.diag_embed(
            ego_position_variance.unsqueeze(-1).expand(-1, -1, 2)
        )
        yaw_gradient = torch.stack(
            (-relative_position[:, :, 1], relative_position[:, :, 0]),
            dim=-1,
        )
        local_covariance = local_covariance + (
            ego_yaw_variance.unsqueeze(-1).unsqueeze(-1)
            * torch.matmul(yaw_gradient.unsqueeze(-1), yaw_gradient.unsqueeze(-2))
        )
        variance = torch.diagonal(local_covariance, dim1=-2, dim2=-1).clamp(min=1.0e-12)
        standard_deviation = torch.sqrt(variance)
        correlation = (
            local_covariance[:, :, 0, 1]
            / (standard_deviation[:, :, 0] * standard_deviation[:, :, 1])
        ).clamp(-0.999, 0.999)
        current_relative_position = relative_position[:, 0].clone()

        log_std = torch.log(standard_deviation / _STD_REFERENCE_M).clamp(
            _LOG_STD_CLIP[0],
            _LOG_STD_CLIP[1],
        )
        log_std *= _UNCERTAINTY_LOG_STD_SCALE
        max_log_std = _LOG_STD_CLIP[1] * _UNCERTAINTY_LOG_STD_SCALE
        horizon_features = build_horizon_tokens(
            relative_position,
            log_std,
            correlation,
            PREDICTION_HORIZONS_S,
            self.belief_valid,
            max_log_std,
        ).reshape(self.num_envs, -1)

        relative_velocity = relative_velocity_from_world(
            means_world[:, 0, 2:4],
            self.observed_base_lin_vel_yaw[:, :2],
            base_yaw_forecast[:, 0],
            self.observed_base_ang_vel[:, 2],
            current_relative_position,
        )
        relative_velocity *= self.belief_valid.unsqueeze(-1).float()

        normalized_age = torch.clamp(
            self.last_measurement_age / _MEASUREMENT_AGE_NORMALIZER_S,
            min=0.0,
            max=1.0,
        )
        status = torch.stack(
            (
                normalized_age,
                self.measurement_updated.float(),
                self.belief_valid.float(),
            ),
            dim=-1,
        )
        return torch.cat((horizon_features, relative_velocity, status), dim=-1)

    def _sample_uniform(self, value_range: tuple[float, float], count: int) -> torch.Tensor:
        lower, upper = value_range
        return lower + (upper - lower) * torch.rand(count, device=self.device)


class CVKFBeliefObservation(ManagerTermBase):
    """Reset-aware observation term owning the batched CVKF state."""

    def __init__(self, cfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        ball_cfg: SceneEntityCfg = cfg.params["ball_cfg"]
        self.state = _get_or_create_state(env, robot_cfg.name, ball_cfg.name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        del robot_cfg, ball_cfg
        return self.state.observe(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # ObservationManager.reset runs before SimulationContext.forward.  Mark
        # the rows here and read the new robot/ball state on the next compute.
        self.state.request_reset(env_ids)


def observed_base_ang_vel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """Return the per-step angular-velocity sample shared with the CVKF."""
    state = _get_or_create_state(env, robot_cfg.name, ball_cfg.name)
    state.prepare_ego_observation(env)
    return state.observed_base_ang_vel


def observed_kick_direction(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """Express the world kick direction in the shared observed-yaw frame."""
    state = _get_or_create_state(env, robot_cfg.name, ball_cfg.name)
    state.prepare_ego_observation(env)
    command = env.command_manager.get_command(command_name)
    direction_world = torch.stack((command[:, 1], command[:, 0]), dim=-1)
    return torch.matmul(
        _world_to_local_rotation(state.observed_base_yaw),
        direction_world.unsqueeze(-1),
    ).squeeze(-1)


def _get_or_create_state(
    env: ManagerBasedRLEnv,
    robot_name: str,
    ball_name: str,
) -> CVKFBeliefState:
    state = getattr(env, _STATE_ATTRIBUTE, None)
    if state is None:
        state = CVKFBeliefState(env, robot_name, ball_name)
        setattr(env, _STATE_ATTRIBUTE, state)
    elif state.robot_name != robot_name or state.ball_name != ball_name:
        raise ValueError("walk-kick likelihood observation terms must share robot and ball assets")
    return state


def _yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract yaw from Isaac Lab's ``(w, x, y, z)`` quaternion layout."""
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _world_to_local_rotation(yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    rotation = torch.zeros(*yaw.shape, 2, 2, device=yaw.device, dtype=yaw.dtype)
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 1] = sine
    rotation[..., 1, 0] = -sine
    rotation[..., 1, 1] = cosine
    return rotation


def _horizontal_fov_mask(local_xy: torch.Tensor, fov_yaw: float) -> torch.Tensor:
    """Return whether each local XY point lies inside the horizontal FOV."""
    if local_xy.ndim != 2 or local_xy.shape[-1] != 2:
        raise ValueError("local_xy must have shape (batch, 2)")
    yaw = torch.atan2(local_xy[:, 1], local_xy[:, 0])
    return torch.abs(yaw) < 0.5 * float(fov_yaw)


def _world_point_to_local(
    point: torch.Tensor,
    origin: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    rotation = _world_to_local_rotation(yaw)
    return torch.matmul(rotation, (point - origin).unsqueeze(-1)).squeeze(-1)


def _local_point_to_world(
    point: torch.Tensor,
    origin: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    rotation = _world_to_local_rotation(yaw)
    return origin + torch.matmul(
        rotation.transpose(-1, -2),
        point.unsqueeze(-1),
    ).squeeze(-1)
