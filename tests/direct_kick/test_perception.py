"""Real camera/filter/Actor325 integration without an IsaacLab process.

The tensor backend only replaces physics writes.  Sensor scheduling, history,
VisionFilter acquisition and public-snapshot forecasting run production code.
"""

import math
import unittest

import torch

from test_source_parity import load_port
from test_task_logic import DirectKickingLogic, TensorBackend


class PerceptionBackend(TensorBackend):
    _belief_observation = DirectKickingLogic._belief_observation

    def _init_direct_perception_buffers(self):
        # Exercise the approved VisionFilter configuration with real perception.
        self.direct_cfg["filter"] = load_port("config").load_config()["direct_kicking"]["filter"]
        self._perception_step_requested = False
        DirectKickingLogic._init_direct_perception_buffers(self)

    def __init__(self, count=2):
        super().__init__(count)
        for name in (
            "ego_velocity_bias_std", "ego_velocity_drift_std", "ego_velocity_noise_std",
            "ego_yaw_rate_bias_std", "ego_yaw_rate_drift_std", "ego_yaw_rate_noise_std",
            "ego_position_noise_std", "ego_position_bias_std", "ego_yaw_noise_std", "ego_yaw_bias_std",
        ):
            setattr(self, name, 0.0)
        self.measurement_noise_std_range = (0.0, 0.0)
        self.latency_range = (0.0, 0.0)
        self.camera_fps_range = (25.0, 25.0)
        self.camera_fps_jitter = 0.0
        self.dropout_burst_probability = 0.0
        self.outlier_probability = 0.0
        self._reset_direct_perception(torch.arange(count))
        # Reset observations do not deliver a camera frame.
        self._compute_observations()

    def tick(self, steps=1, force_frame=False):
        self.common_step_counter += steps
        if force_frame:
            self.camera_timer.fill_(-0.001)
        self._perception_step_requested = True
        try:
            self._compute_observations()
        finally:
            self._perception_step_requested = False
        return self.obs_buf


class PerceptionTest(unittest.TestCase):
    def assert_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype),
                                   rtol=kwargs.get("rtol", 1e-5), atol=kwargs.get("atol", 1e-6))

    def assert_invalid_actor(self, env, index):
        tokens = env.obs_buf[index, 47:320].reshape(13, 21)
        self.assert_close(tokens[:, :4], torch.zeros(13, 4))
        invalid_variance = (.01 * math.exp(4))**2
        expected_covariance = torch.diag(torch.tensor([invalid_variance, invalid_variance, 0., 0.]))
        self.assert_close(tokens[:, 4:20].reshape(13, 4, 4), expected_covariance.expand(13, -1, -1))
        self.assert_close(tokens[:, 20], torch.tensor(env.prediction_horizons) / 3.0)
        self.assert_close(env.obs_buf[index, 321:323], [0., 0.])
        self.assertTrue(torch.isfinite(env.obs_buf[index]).all())

    def test_camera_schedule_fixed_lag_uses_capture_pose_and_timestamps(self):
        env = PerceptionBackend()
        env.perception_latency_steps[:] = torch.tensor([0, 2])
        env.perception_latency[:] = torch.tensor([0., 0.04])
        env.camera_timer[:] = torch.tensor([0.019, 0.039])
        calls = []
        process = env.ball_filter.process

        def record(ids, stamp, global_xy, local_xy, observed):
            calls.append(tuple(value.clone() for value in (ids, stamp, global_xy, local_xy, observed)))
            return process(ids, stamp, global_xy, local_xy, observed)

        env.ball_filter.process = record
        # Robot translation makes using present-time instead of capture-time
        # pose observable. Global ball position must still reconstruct exactly.
        for step in range(1, 7):
            env.base_pos[:, 0] = 0.005 * step
            env.ball_pos[:, 0] = 0.3 + 0.02 * step
            env.tick()
        by_env = {0: [], 1: []}
        for ids, stamps, glob, local, observed in calls:
            for row, index in enumerate(ids.tolist()):
                by_env[index].append(int(stamps[row]))
                capture_step = int(stamps[row]) / 20_000_000
                self.assert_close(glob[row], [0.3 + 0.02 * capture_step, 0.096])
                self.assert_close(local[row], [0.3 + 0.015 * capture_step, 0.096])
                self.assertTrue(observed[row])
        self.assertEqual(by_env, {0: [20_000_000, 60_000_000, 100_000_000],
                                  1: [0, 40_000_000, 80_000_000]})
        self.assert_close(env.last_measurement_age, [0.02, 0.04])
        self.assertEqual(env.ball_filter.stamp_ns.tolist(), [100_000_000, 80_000_000])
        self.assertEqual(tuple(env.obs_buf.shape), (2, 325))
        self.assertEqual(env.obs_buf[:, 321:323].tolist(), [[0., 1.], [1., 1.]])

    def test_missing_frames_clear_tentative_then_confirmed_timeout_and_reacquire(self):
        env = PerceptionBackend(1)
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.capture_count.item(), 1)
        self.assert_invalid_actor(env, 0)
        env.dropout_remaining.fill_(1)
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.capture_count.item(), 0)
        env.tick(force_frame=True)
        self.assertFalse(env.belief_valid.item())
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.status.item(), 1)
        self.assert_close(env.obs_buf[0, 320:323], [0., 1., 1.])

        confirmed_x = env.ball_filter.confirmed.x.clone()
        stamp = env.ball_filter.stamp_ns.clone()
        env.camera_timer.fill_(1.0)
        env.tick()
        self.assert_close(env.ball_filter.confirmed.x, confirmed_x)
        self.assertTrue(torch.equal(env.ball_filter.stamp_ns, stamp))
        self.assert_close(env.obs_buf[0, 320:323], [0.04, 0., 1.])
        env.dropout_remaining.fill_(1)
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.status.item(), 2)
        self.assert_close(env.obs_buf[0, 321:323], [0., 1.])

        # Source timeout is strict: exactly 3 seconds retains the prediction.
        last = int(env.ball_filter.observation_ns.item()) // 20_000_000
        env.dropout_remaining.fill_(1)
        env.tick(steps=last + 150 - env.common_step_counter, force_frame=True)
        self.assertEqual(env.ball_filter.status.item(), 2)
        env.dropout_remaining.fill_(1)
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.status.item(), 0)
        self.assert_invalid_actor(env, 0)
        self.assert_close(env.ball_filter.state, torch.zeros(1, 4))
        self.assert_close(env.ball_filter.covariance, torch.zeros(1, 4, 4))
        env.tick(force_frame=True)
        self.assert_invalid_actor(env, 0)
        env.tick(force_frame=True)
        self.assertTrue(env.belief_valid.item())
        self.assert_close(env.obs_buf[0, 320:323], [0., 1., 1.])

    def test_partial_reset_clears_history_and_filter_only_for_selected_environment(self):
        env = PerceptionBackend()
        env.tick(force_frame=True)
        env.tick(force_frame=True)
        other = {}
        for name in ("state", "covariance", "status", "stamp_ns", "observation_ns", "active"):
            other[name] = getattr(env.ball_filter, name)[1].clone()
        history = env.ball_position_history[1].clone()
        env.ball_pos[0, :2] = torch.tensor([0.7, -0.2])
        env._reset_direct_perception(torch.tensor([0]))
        self.assertFalse(env.ball_filter.initialized[0])
        self.assertEqual(env.perception_history_valid_steps[0], 0)
        self.assert_close(env.ball_position_history[0], env.ball_pos[0, :2].expand(5, -1))
        self.assert_close(env.ball_position_history[1], history)
        for name, value in other.items():
            self.assertTrue(torch.equal(getattr(env.ball_filter, name)[1], value), name)
        env.tick(force_frame=True)
        self.assert_invalid_actor(env, 0)
        self.assertEqual(env.ball_filter.capture_count[0], 0)
        self.assertEqual(env.perception_history_valid_steps[0], 0)
        self.assertTrue(env.belief_valid[1])
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.capture_count[0], 1)
        env.tick(force_frame=True)
        self.assertTrue(env.belief_valid.all())

    def test_fov_and_initial_outlier_are_missing_camera_messages(self):
        env = PerceptionBackend()
        env.ball_pos[1, 0] = -0.3
        env.outlier_probability = 1.0
        env.tick(force_frame=True)
        self.assertEqual(env.ball_filter.capture_count.tolist(), [0, 0])
        env.outlier_probability = 0.0
        env.tick(force_frame=True)
        env.tick(force_frame=True)
        self.assertEqual(env.belief_valid.tolist(), [True, False])
        self.assert_invalid_actor(env, 1)

    def test_full_public_covariance_reaches_tokens_with_consistent_scaling(self):
        env = PerceptionBackend(1)
        env.common_step_counter = 2
        env.ball_filter.stamp_ns.zero_()
        state = torch.tensor([1.2, -.3, .8, .2], dtype=torch.float64)
        factor = torch.tensor([[.2, 0., 0., 0.], [.03, .25, 0., 0.],
                               [.1, -.05, .3, 0.], [-.02, .06, .07, .4]], dtype=torch.float64)
        p = factor @ factor.T
        env.ball_filter.state[0] = state.float()
        env.ball_filter.covariance[0] = p.float()
        env.belief_valid[:] = True
        env.observed_base_position[:] = torch.tensor([.2, -.1])
        env.observed_base_yaw.zero_()
        env.task_cfg["normalization"].update(ball_pos=2., ball_vel=3.)
        belief = env._belief_observation(torch.tensor([[.4, -.2, 0.]]), torch.zeros(1, 3))
        scale = torch.diag(torch.tensor([2., 2., 3., 3.], dtype=torch.float64))
        for i, h in enumerate(env.prediction_horizons):
            t = .04 + h
            f = torch.eye(4, dtype=torch.float64)
            f[0, 2] = f[1, 3] = t
            g = torch.tensor([[t*t/2, 0.], [0., t*t/2], [t, 0.], [0., t]], dtype=torch.float64)
            expected_p = scale @ (f @ p @ f.T + .64 * g @ g.T) @ scale
            relative = f @ state
            relative[:2] -= torch.tensor([.2 + .4*h, -.1 - .2*h], dtype=torch.float64)
            relative[2:] -= torch.tensor([.4, -.2], dtype=torch.float64)
            token = belief[0, :273].reshape(13, 21)[i]
            self.assert_close(token[:4], scale @ relative)
            self.assert_close(token[4:20].reshape(4, 4), expected_p)
        # Large future uncertainty must not be saturated to the old log-std cap.
        self.assertGreater(belief[0, :273].reshape(13, 21)[-1, 4], 10.)

    def test_horizons_use_public_snapshot_age_and_ego_horizon_only(self):
        env = PerceptionBackend(1)
        env.common_step_counter = 100  # Present time = 2 s, snapshot age = 0.04 s.
        env.ball_filter.stamp_ns[:] = 1_960_000_000
        state = torch.tensor([1.7, 0.8, 0.6, -0.25], dtype=torch.float64)
        p = torch.tensor([[.012, .001, .003, -.002], [.001, .018, .001, .004],
                          [.003, .001, .03, .002], [-.002, .004, .002, .04]], dtype=torch.float64)
        env.ball_filter.state[0] = state.float()
        env.ball_filter.covariance[0] = p.float()
        # Core internals deliberately disagree: the deployment contract exposes
        # only the chosen float32 snapshot, never the hidden hypothesis bank.
        env.ball_filter.confirmed.x.fill_(99.)
        env.ball_filter.confirmed.p.fill_(77.)
        env.belief_valid[:] = True
        env.observed_base_position[:] = torch.tensor([0.2, -0.1])
        env.observed_base_yaw[:] = 0.7
        env.ego_position_noise_std = 0.01
        env.ego_position_bias_std = 0.02
        env.ego_velocity_noise_std = 0.03
        env.ego_velocity_bias_std = 0.04
        env.ego_velocity_drift_std = 0.05
        env.ego_yaw_noise_std = 0.02
        env.ego_yaw_bias_std = 0.03
        env.ego_yaw_rate_noise_std = 0.04
        env.ego_yaw_rate_bias_std = 0.05
        env.ego_yaw_rate_drift_std = 0.06
        belief = env._belief_observation(torch.tensor([[.5, -.2, 0.]]), torch.tensor([[0., 0., .4]]))
        expected = []
        for h in env.prediction_horizons:
            t = h + 0.04
            f = torch.eye(4, dtype=torch.float64)
            f[0, 2] = f[1, 3] = t
            ball = f @ state
            covariance = (f @ p @ f.T)[:2, :2] + torch.eye(2, dtype=torch.float64) * (.64 * t**4 / 4)
            dx = math.sin(.4*h)/.4 * .5 + (math.cos(.4*h)-1)/.4 * -.2
            dy = (1-math.cos(.4*h))/.4 * .5 + math.sin(.4*h)/.4 * -.2
            bx = .2 + math.cos(.7)*dx - math.sin(.7)*dy
            by = -.1 + math.sin(.7)*dx + math.cos(.7)*dy
            angle = .7 + .4*h
            rotation = torch.tensor([[math.cos(angle), math.sin(angle)],
                                     [-math.sin(angle), math.cos(angle)]], dtype=torch.float64)
            relative = rotation @ (ball[:2] - torch.tensor([bx, by], dtype=torch.float64))
            variance_xy = .01**2 + .02**2 + h*h*(.03**2+.04**2) + h**3*.05**2/3
            variance_yaw = .02**2 + .03**2 + h*h*(.04**2+.05**2) + h**3*.06**2/3
            g = torch.stack((-relative[1], relative[0]))
            local_p = rotation @ covariance @ rotation.T + torch.eye(2)*variance_xy + variance_yaw*g[:, None]*g[None, :]
            relative_velocity = rotation @ state[2:] - torch.tensor([.5, -.2]) - .4*g
            token = belief[0, :273].reshape(13, 21)[len(expected)]
            self.assert_close(token[:4], torch.cat((relative, relative_velocity)))
            self.assert_close(token[4:20].reshape(4, 4)[:2, :2], local_p)
            self.assert_close(token[20], h/3)
            expected.append(token)
        self.assertEqual(tuple(belief.shape), (1, 276))


if __name__ == "__main__":
    unittest.main()
