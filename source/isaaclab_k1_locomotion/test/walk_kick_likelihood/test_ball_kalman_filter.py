import sys
import unittest
from pathlib import Path

import torch


MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick_likelihood"
    / "mdp"
)
sys.path.insert(0, str(MDP_DIR))

from ball_kalman_filter import (  # noqa: E402
    BatchedConstantVelocityKalmanFilter,
    distance_scaled_measurement_std,
)


class BatchedConstantVelocityKalmanFilterTest(unittest.TestCase):
    def _filter(self, count=2, dt=0.1):
        kalman_filter = BatchedConstantVelocityKalmanFilter(
            count,
            dt,
            torch.device("cpu"),
        )
        env_ids = torch.arange(count)
        kalman_filter.reset(
            env_ids,
            torch.zeros(count, 2),
            torch.full((count,), 0.01),
            torch.full((count,), 0.2),
        )
        return kalman_filter

    def test_predict_advances_constant_velocity_state(self):
        kalman_filter = self._filter(count=1)
        kalman_filter.state[0, 2:] = torch.tensor([1.0, -0.5])

        kalman_filter.predict(torch.zeros(1))

        self.assertTrue(
            torch.allclose(
                kalman_filter.state[0],
                torch.tensor([0.1, -0.05, 1.0, -0.5]),
            )
        )

    def test_nis_gate_rejects_large_outlier(self):
        kalman_filter = self._filter(count=1)
        state_before = kalman_filter.state.clone()
        covariance_before = kalman_filter.covariance.clone()

        accepted = kalman_filter.update(
            torch.tensor([[10.0, -10.0]]),
            torch.tensor([True]),
            torch.tensor([0.01]),
            nis_threshold=13.82,
        )

        self.assertFalse(accepted.item())
        self.assertTrue(torch.allclose(kalman_filter.state, state_before))
        self.assertTrue(torch.allclose(kalman_filter.covariance, covariance_before))

    def test_forecast_covariance_matches_repeated_control_predictions(self):
        forecast_filter = self._filter(count=1)
        repeated_filter = self._filter(count=1)
        acceleration_std = torch.tensor([0.8])

        forecast_mean, forecast_covariance = forecast_filter.forecast(
            [0.2],
            acceleration_std,
        )
        repeated_filter.predict(acceleration_std)
        repeated_filter.predict(acceleration_std)

        self.assertTrue(
            torch.allclose(
                forecast_mean[:, 0],
                repeated_filter.state,
                rtol=1.0e-5,
                atol=1.0e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                forecast_covariance[:, 0],
                repeated_filter.covariance,
                rtol=1.0e-5,
                atol=1.0e-7,
            )
        )

    def test_per_filter_offsets_are_supported(self):
        kalman_filter = self._filter(count=2)
        kalman_filter.state[:, 2] = 1.0

        means, covariance = kalman_filter.forecast_offsets(
            torch.tensor([[0.0, 0.1], [0.2, 0.3]]),
            torch.zeros(2),
        )

        self.assertTrue(torch.allclose(means[0, :, 0], torch.tensor([0.0, 0.1])))
        self.assertTrue(torch.allclose(means[1, :, 0], torch.tensor([0.2, 0.3])))
        self.assertEqual(tuple(covariance.shape), (2, 2, 4, 4))

    def test_measurement_std_scales_with_distance_and_caps_at_eight(self):
        scaled = distance_scaled_measurement_std(
            torch.tensor([0.01, 0.01, 0.01]),
            torch.tensor([0.5, 2.0, 10.0]),
            reference_distance=1.0,
            maximum_scale=8.0,
        )

        torch.testing.assert_close(scaled, torch.tensor([0.01, 0.02, 0.08]))


if __name__ == "__main__":
    unittest.main()
