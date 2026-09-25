"""Full-state uncertainty, reflection, and exclusive LSTM ball-input contract."""
import math
import unittest
from unittest.mock import patch

import torch

from test_source_parity import HORIZONS, load_port


ego = load_port("ego_motion")
observation = load_port("direct_kicking_observation")
symmetry = load_port("direct_kicking_symmetry")


class StateCovarianceTest(unittest.TestCase):
    def test_moving_frame_covariance_matches_independent_finite_difference(self):
        dtype = torch.float64
        ball = torch.tensor([1.7, .8, .6, -.25], dtype=dtype)
        base_position = torch.tensor([.2, -.1], dtype=dtype)
        yaw, rate = .7, .4
        velocity = torch.tensor([.5, -.2], dtype=dtype)
        # Correlated position/velocity errors exercise all four covariance blocks.
        factor = torch.tensor([[.2, 0, 0, 0], [.03, .25, 0, 0],
                               [.1, -.05, .3, 0], [-.02, .06, .07, .4]], dtype=dtype)
        ball_p = factor @ factor.T
        t = torch.tensor([[.75]], dtype=dtype)
        ego_p = ego.forecast_ego_state_covariance(t, .01, .02, .03, .04, .05, .02, .03, .04, .05, .06)
        # Independently check each integrated random-walk covariance, including
        # position/rate correlations rather than using the production marginal.
        expected_ego = torch.zeros(6, 6, dtype=dtype)
        for position_index, velocity_index, p0, v0, q in (
            (0, 3, .01**2 + .02**2, .03**2 + .04**2, .05**2),
            (1, 4, .01**2 + .02**2, .03**2 + .04**2, .05**2),
            (2, 5, .02**2 + .03**2, .04**2 + .05**2, .06**2),
        ):
            expected_ego[position_index, position_index] = p0 + .75**2*v0 + .75**3*q/3
            expected_ego[velocity_index, velocity_index] = v0 + .75*q
            expected_ego[position_index, velocity_index] = .75*v0 + .75**2*q/2
            expected_ego[velocity_index, position_index] = .75*v0 + .75**2*q/2
        torch.testing.assert_close(ego_p[0, 0], expected_ego)

        def state_from_perturbed_inputs(z):
            theta = yaw + z[6]
            c, s = torch.cos(theta), torch.sin(theta)
            # Position-error components are expressed in the nominal local frame.
            px = base_position[0] + math.cos(yaw)*z[4] - math.sin(yaw)*z[5]
            py = base_position[1] + math.sin(yaw)*z[4] + math.cos(yaw)*z[5]
            x = c*(z[0]-px) + s*(z[1]-py)
            y = -s*(z[0]-px) + c*(z[1]-py)
            vx = c*z[2] + s*z[3] - velocity[0] - z[7] + (rate+z[9])*y
            vy = -s*z[2] + c*z[3] - velocity[1] - z[8] - (rate+z[9])*x
            return torch.stack((x, y, vx, vy))

        z = torch.cat((ball, torch.zeros(6, dtype=dtype)))
        jacobian = torch.empty(4, 10, dtype=dtype)
        for i in range(10):
            delta = torch.zeros_like(z)
            delta[i] = 1e-6
            jacobian[:, i] = (state_from_perturbed_inputs(z+delta)-state_from_perturbed_inputs(z-delta))/2e-6
        expected_p = jacobian @ torch.block_diag(ball_p, expected_ego) @ jacobian.T
        state, covariance = ego.relative_state_and_covariance(
            ball[None, None], ball_p[None, None], base_position[None, None],
            torch.tensor([[yaw]], dtype=dtype), velocity[None, None],
            torch.tensor([[rate]], dtype=dtype), ego_p,
        )
        torch.testing.assert_close(state[0, 0], state_from_perturbed_inputs(z))
        torch.testing.assert_close(covariance[0, 0], expected_p, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(covariance, covariance.transpose(-1, -2))
        self.assertTrue((torch.linalg.eigvalsh(covariance) >= 0).all())

    def test_mirror_transforms_all_state_covariance_elements_and_is_involution(self):
        torch.manual_seed(180)
        obs = torch.randn(2, 325)
        forecast = obs[:, 47:320].reshape(2, 13, 21)
        covariance = forecast[..., 4:20].reshape(2, 13, 4, 4)
        mirror = symmetry.mirror_direct_kicking_observation(obs, 13)
        mirrored_forecast = mirror[:, 47:320].reshape(2, 13, 21)
        reflected_covariance = mirrored_forecast[..., 4:20].reshape(2, 13, 4, 4)
        signs = [1, -1, 1, -1]
        for row in range(4):
            torch.testing.assert_close(mirrored_forecast[..., row], forecast[..., row]*signs[row])
            for col in range(4):
                torch.testing.assert_close(reflected_covariance[..., row, col], covariance[..., row, col]*signs[row]*signs[col])
        torch.testing.assert_close(mirrored_forecast[..., 20], forecast[..., 20])
        torch.testing.assert_close(mirror[:, 320:323], obs[:, 320:323])
        torch.testing.assert_close(symmetry.mirror_direct_kicking_observation(mirror, 13), obs)

    def test_ball_state_and_all_covariance_entries_only_reach_mlp_through_lstm(self):
        torch.manual_seed(102)
        model = load_port("model").DirectKickingActorCritic(12, 325, 20, HORIZONS)
        obs = torch.randn(2, 325)
        seen = []
        hook = model.actor.encoder.lstm.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        encoded = model.actor.encoder(obs)
        hook.remove()
        self.assertEqual(tuple(seen[0].shape), (2, 13, 21))
        torch.testing.assert_close(seen[0], obs[:, 47:320].reshape(2, 13, 21))
        self.assertEqual(tuple(encoded.shape), (2, 116))
        torch.testing.assert_close(encoded[:, :52], torch.cat((obs[:, :47], obs[:, 320:]), dim=-1))
        changed = obs.clone()
        changed[:, 47:320] += 3.0
        zero_state = torch.zeros(1, 2, 64)
        zero_output = torch.zeros(2, 13, 64)
        with patch.object(model.actor.encoder.lstm, "forward", return_value=(zero_output, (zero_state, zero_state))):
            torch.testing.assert_close(model.actor(obs), model.actor(changed))
        self.assertEqual(model.checkpoint_metadata()["horizon_token_size"], 21)
        self.assertEqual(model.checkpoint_metadata()["num_observations"], 325)
        self.assertEqual(model.checkpoint_metadata()["num_privileged_observations"], 20)


if __name__ == "__main__":
    unittest.main()
