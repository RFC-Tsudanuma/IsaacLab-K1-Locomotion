"""Batched constant-velocity Kalman filtering for ball tracking.

The implementation is intentionally independent from Isaac Lab so the same
state estimator can be reused by training and deployment code. State and
covariance tensors are kept per environment and updated in one Torch batch.
"""

from typing import Optional, Sequence, Tuple

import torch


def distance_scaled_measurement_std(
    base_std: torch.Tensor,
    distances: torch.Tensor,
    reference_distance: float,
    maximum_scale: float,
) -> torch.Tensor:
    """Grow measurement standard deviation linearly with observation range."""
    if reference_distance <= 0.0:
        raise ValueError("reference_distance must be positive")
    if maximum_scale < 1.0:
        raise ValueError("maximum_scale must be at least one")
    distance_scale = torch.clamp(
        distances / float(reference_distance),
        min=1.0,
        max=float(maximum_scale),
    )
    return base_std * distance_scale


class BatchedConstantVelocityKalmanFilter:
    """Constant-velocity Kalman filters with state ``[x, y, vx, vy]``.

    The filter itself always updates at the measurement timestamp. A caller
    with delayed observations must keep the filter on that delayed timeline
    and forecast from there; treating an old measurement as a current one
    would make covariance inconsistent when process noise is non-zero.
    """

    state_dim = 4
    measurement_dim = 2

    def __init__(
        self,
        num_filters: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_filters <= 0:
            raise ValueError("num_filters must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.num_filters = int(num_filters)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.state = torch.zeros(
            self.num_filters,
            self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.covariance = torch.zeros(
            self.num_filters,
            self.state_dim,
            self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.initialized = torch.zeros(
            self.num_filters,
            device=self.device,
            dtype=torch.bool,
        )
        self._identity = torch.eye(
            self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def reset(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        position_std: torch.Tensor,
        velocity_std: torch.Tensor,
    ) -> None:
        """Initialize selected filters from position-only measurements."""
        env_ids = self._env_ids(env_ids)
        count = env_ids.numel()
        if count == 0:
            return

        positions = self._rows(positions, count, 2, "positions")
        position_std = self._values(position_std, count, "position_std")
        velocity_std = self._values(velocity_std, count, "velocity_std")
        self.state[env_ids] = 0.0
        self.state[env_ids, :2] = positions
        diagonal = torch.stack(
            (
                position_std.square(),
                position_std.square(),
                velocity_std.square(),
                velocity_std.square(),
            ),
            dim=-1,
        )
        self.covariance[env_ids] = torch.diag_embed(diagonal)
        self.initialized[env_ids] = True

    def invalidate(self, env_ids: torch.Tensor) -> None:
        """Mark selected filters as uninitialized and clear their belief."""
        env_ids = self._env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self.state[env_ids] = 0.0
        self.covariance[env_ids] = 0.0
        self.initialized[env_ids] = False

    def predict(
        self,
        process_acceleration_std: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance initialized filters by the configured control timestep."""
        active = self.initialized.clone()
        if mask is not None:
            active &= self._mask(mask, "mask")
        env_ids = active.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        acceleration_std = self._full_values(
            process_acceleration_std,
            "process_acceleration_std",
        )[env_ids]
        transition = self._transition(self.dt)
        state = self.state[env_ids]
        covariance = self.covariance[env_ids]
        self.state[env_ids] = torch.matmul(state, transition.transpose(0, 1))
        predicted_covariance = torch.matmul(
            torch.matmul(transition.unsqueeze(0), covariance),
            transition.transpose(0, 1).unsqueeze(0),
        )
        predicted_covariance += self._process_noise(acceleration_std, self.dt)
        self.covariance[env_ids] = self._symmetrize(predicted_covariance)

    def update(
        self,
        measurements: torch.Tensor,
        update_mask: torch.Tensor,
        measurement_std: torch.Tensor,
        nis_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Correct from position measurements and return the accepted mask.

        ``nis_threshold`` applies a normalized-innovation-squared gate. A
        rejected outlier leaves the predicted belief untouched.
        """
        measurements = self._rows(
            measurements,
            self.num_filters,
            self.measurement_dim,
            "measurements",
        )
        requested = self._mask(update_mask, "update_mask") & self.initialized
        requested &= torch.isfinite(measurements).all(dim=-1)
        env_ids = requested.nonzero(as_tuple=False).flatten()
        accepted = torch.zeros_like(requested)
        if env_ids.numel() == 0:
            return accepted

        std = self._full_values(measurement_std, "measurement_std")[env_ids]
        count = env_ids.numel()
        measurement_matrix = torch.zeros(
            count,
            self.measurement_dim,
            self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        measurement_matrix[:, 0, 0] = 1.0
        measurement_matrix[:, 1, 1] = 1.0

        state = self.state[env_ids]
        covariance = self.covariance[env_ids]
        measurement = measurements[env_ids]
        predicted_measurement = torch.matmul(
            measurement_matrix,
            state.unsqueeze(-1),
        ).squeeze(-1)
        innovation = measurement - predicted_measurement
        measurement_noise = torch.diag_embed(
            torch.stack((std.square(), std.square()), dim=-1)
        )
        covariance_measurement = torch.matmul(
            covariance,
            measurement_matrix.transpose(-1, -2),
        )
        innovation_covariance = torch.matmul(
            measurement_matrix,
            covariance_measurement,
        ) + measurement_noise
        innovation_covariance = self._symmetrize(innovation_covariance)

        solved_innovation = torch.linalg.solve(
            innovation_covariance,
            innovation.unsqueeze(-1),
        ).squeeze(-1)
        nis = torch.sum(innovation * solved_innovation, dim=-1)
        local_accept = torch.isfinite(nis)
        if nis_threshold is not None:
            if nis_threshold <= 0.0:
                raise ValueError("nis_threshold must be positive")
            local_accept &= nis <= float(nis_threshold)

        kalman_gain = torch.linalg.solve(
            innovation_covariance,
            covariance_measurement.transpose(-1, -2),
        ).transpose(-1, -2)
        corrected_state = state + torch.matmul(
            kalman_gain,
            innovation.unsqueeze(-1),
        ).squeeze(-1)

        identity = self._identity.unsqueeze(0).expand(count, -1, -1)
        innovation_update = identity - torch.matmul(kalman_gain, measurement_matrix)
        corrected_covariance = torch.matmul(
            torch.matmul(innovation_update, covariance),
            innovation_update.transpose(-1, -2),
        )
        corrected_covariance += torch.matmul(
            torch.matmul(kalman_gain, measurement_noise),
            kalman_gain.transpose(-1, -2),
        )
        corrected_covariance = self._symmetrize(corrected_covariance)
        self.state[env_ids] = torch.where(
            local_accept.unsqueeze(-1),
            corrected_state,
            state,
        )
        self.covariance[env_ids] = torch.where(
            local_accept.view(-1, 1, 1),
            corrected_covariance,
            covariance,
        )
        accepted[env_ids] = local_accept
        return accepted

    def forecast(
        self,
        horizons: Sequence[float],
        process_acceleration_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return state means and covariances at each future horizon."""
        horizon_values = tuple(float(horizon) for horizon in horizons)
        if not horizon_values:
            raise ValueError("horizons must be a non-empty one-dimensional sequence")
        if any(horizon < 0.0 for horizon in horizon_values):
            raise ValueError("forecast horizons must be non-negative")
        horizon_tensor = torch.as_tensor(
            horizon_values,
            device=self.device,
            dtype=self.dtype,
        )
        offsets = horizon_tensor.unsqueeze(0).expand(self.num_filters, -1)
        return self.forecast_offsets(offsets, process_acceleration_std)

    def forecast_offsets(
        self,
        offsets: torch.Tensor,
        process_acceleration_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forecast each filter at its own set of non-negative offsets."""
        offsets = torch.as_tensor(offsets, device=self.device, dtype=self.dtype)
        if offsets.ndim != 2 or offsets.shape[0] != self.num_filters:
            raise ValueError("offsets must have shape (num_filters, num_horizons)")

        transition = self._transition(offsets)
        means = torch.matmul(
            transition,
            self.state.unsqueeze(1).unsqueeze(-1),
        ).squeeze(-1)
        covariance = torch.matmul(
            torch.matmul(
                transition,
                self.covariance.unsqueeze(1),
            ),
            transition.transpose(-1, -2),
        )
        acceleration_std = self._full_values(
            process_acceleration_std,
            "process_acceleration_std",
        )
        covariance += self._accumulated_process_noise(
            acceleration_std,
            offsets,
        )
        return means, self._symmetrize(covariance)

    def _transition(self, dt: torch.Tensor) -> torch.Tensor:
        dt_tensor = torch.as_tensor(dt, device=self.device, dtype=self.dtype)
        if dt_tensor.ndim == 0:
            transition = self._identity.clone()
            transition[0, 2] = dt_tensor
            transition[1, 3] = dt_tensor
            return transition

        transition = self._identity.expand(*dt_tensor.shape, -1, -1).clone()
        transition[..., 0, 2] = dt_tensor
        transition[..., 1, 3] = dt_tensor
        return transition

    def _process_noise(
        self,
        acceleration_std: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        dt_tensor = torch.full_like(acceleration_std, float(dt))
        return self._process_noise_terms(acceleration_std, dt_tensor)

    def _accumulated_process_noise(
        self,
        acceleration_std: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate the same discrete Q used by repeated control steps."""
        step = self.dt
        full_steps = torch.floor(offsets / step + 1.0e-6)
        remainder = torch.clamp(offsets - full_steps * step, min=0.0)
        variance = acceleration_std.square().unsqueeze(1)

        q_xx = (
            variance
            * (step**4)
            * full_steps
            * (4.0 * full_steps.square() - 1.0)
            / 12.0
        )
        q_xv = variance * (step**3) * full_steps.square() * 0.5
        q_vv = variance * (step**2) * full_steps

        accumulated_xx = (
            q_xx
            + 2.0 * remainder * q_xv
            + remainder.square() * q_vv
            + 0.25 * remainder.pow(4) * variance
        )
        accumulated_xv = (
            q_xv
            + remainder * q_vv
            + 0.5 * remainder.pow(3) * variance
        )
        accumulated_vv = q_vv + remainder.square() * variance

        noise = torch.zeros(
            self.num_filters,
            offsets.shape[1],
            self.state_dim,
            self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        noise[:, :, 0, 0] = accumulated_xx
        noise[:, :, 1, 1] = accumulated_xx
        noise[:, :, 0, 2] = accumulated_xv
        noise[:, :, 2, 0] = accumulated_xv
        noise[:, :, 1, 3] = accumulated_xv
        noise[:, :, 3, 1] = accumulated_xv
        noise[:, :, 2, 2] = accumulated_vv
        noise[:, :, 3, 3] = accumulated_vv
        return noise

    def _process_noise_terms(
        self,
        acceleration_std: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        dt2 = dt.square()
        dt3 = dt2 * dt
        dt4 = dt2.square()
        variance = acceleration_std.square()
        noise = torch.zeros(
            acceleration_std.numel(),
            self.state_dim,
            self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        noise[:, 0, 0] = 0.25 * dt4 * variance
        noise[:, 1, 1] = 0.25 * dt4 * variance
        noise[:, 0, 2] = 0.5 * dt3 * variance
        noise[:, 2, 0] = noise[:, 0, 2]
        noise[:, 1, 3] = 0.5 * dt3 * variance
        noise[:, 3, 1] = noise[:, 1, 3]
        noise[:, 2, 2] = dt2 * variance
        noise[:, 3, 3] = dt2 * variance
        return noise

    def _env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        return env_ids

    def _mask(self, mask: torch.Tensor, name: str) -> torch.Tensor:
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_filters,):
            raise ValueError("{} must have shape ({},)".format(name, self.num_filters))
        return mask

    def _rows(
        self,
        values: torch.Tensor,
        rows: int,
        columns: int,
        name: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if values.shape != (rows, columns):
            raise ValueError("{} must have shape ({}, {})".format(name, rows, columns))
        return values

    def _values(self, values: torch.Tensor, count: int, name: str) -> torch.Tensor:
        values = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if values.ndim == 0:
            return values.expand(count)
        if values.shape != (count,):
            raise ValueError("{} must be scalar or have shape ({},)".format(name, count))
        return values

    def _full_values(self, values: torch.Tensor, name: str) -> torch.Tensor:
        return self._values(values, self.num_filters, name)

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.transpose(-1, -2))
