import importlib.util
import math
import sys
import types
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
PACKAGE_NAME = "_walk_kick_likelihood_mdp_test"


def _load_belief_module():
    isaaclab = types.ModuleType("isaaclab")
    managers = types.ModuleType("isaaclab.managers")

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    managers.ManagerTermBase = ManagerTermBase
    managers.SceneEntityCfg = SceneEntityCfg
    isaaclab.managers = managers

    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MDP_DIR)]
    sys.modules.setdefault(PACKAGE_NAME, package)
    module_name = f"{PACKAGE_NAME}.belief"
    spec = importlib.util.spec_from_file_location(module_name, MDP_DIR / "belief.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_isaaclab = sys.modules.get("isaaclab")
    previous_managers = sys.modules.get("isaaclab.managers")
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.managers"] = managers
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_isaaclab is None:
            sys.modules.pop("isaaclab", None)
        else:
            sys.modules["isaaclab"] = previous_isaaclab
        if previous_managers is None:
            sys.modules.pop("isaaclab.managers", None)
        else:
            sys.modules["isaaclab.managers"] = previous_managers
    return module


belief = _load_belief_module()


class BeliefLifecycleTest(unittest.TestCase):
    def _state(self, step, episode_lengths, pending_reset):
        state = belief.CVKFBeliefState.__new__(belief.CVKFBeliefState)
        state.last_step = step
        state.last_ego_advance_step = step
        state.last_episode_length = episode_lengths.clone()
        state.pending_reset = pending_reset.clone()
        state.perception_just_reset = torch.zeros_like(pending_reset)
        state.cached_observation = torch.zeros(2, belief.CVKF_BELIEF_OBSERVATION_SIZE)
        state.ego_generation = 0
        state.ego_cache_key = (step, 0)
        state.observation_dirty = False
        state._update_ego_motion_sensor_noise = lambda: None
        state._update_observed_base_pose = lambda env: None
        state._sample_observed_velocities = lambda env: None
        return state

    def test_same_step_episode_length_change_is_synced_for_later_partial_reset(self):
        state = self._state(
            0,
            torch.tensor([0, 0]),
            torch.tensor([False, False]),
        )

        reset_calls = []
        def reset(env, env_ids):
            del env
            reset_calls.append(env_ids.clone())
            state.ego_generation += 1
            state.ego_cache_key = None
            state.observation_dirty = True
            state.perception_just_reset[env_ids] = True

        state._reset = reset
        state._build_observation = lambda env: state.cached_observation
        state._advance = lambda env: self.fail("same-step synchronization advanced the filter")

        env = types.SimpleNamespace(
            common_step_counter=0,
            episode_length_buf=torch.tensor([5, 7]),
        )
        state.observe(env)

        self.assertTrue(torch.equal(state.last_episode_length, torch.tensor([5, 7])))
        self.assertEqual(reset_calls, [])

        env.episode_length_buf = torch.tensor([0, 8])
        state.observe(env)

        self.assertEqual(len(reset_calls), 1)
        self.assertTrue(torch.equal(reset_calls[0], torch.tensor([0])))
        self.assertTrue(torch.equal(state.last_episode_length, torch.tensor([0, 8])))

    def test_explicit_reset_is_consumed_on_same_step(self):
        state = self._state(
            4,
            torch.tensor([3, 3]),
            torch.tensor([False, True]),
        )

        reset_calls = []
        def reset(env, env_ids):
            del env
            reset_calls.append(env_ids.clone())
            state.ego_generation += 1
            state.ego_cache_key = None
            state.observation_dirty = True
            state.perception_just_reset[env_ids] = True

        state._reset = reset
        state._build_observation = lambda env: state.cached_observation
        state._advance = lambda env: self.fail("same-step reset advanced the filter")

        env = types.SimpleNamespace(
            common_step_counter=4,
            episode_length_buf=torch.tensor([3, 0]),
        )
        state.observe(env)

        self.assertEqual(len(reset_calls), 1)
        self.assertTrue(torch.equal(reset_calls[0], torch.tensor([1])))
        self.assertFalse(state.pending_reset.any())
        self.assertFalse(state.perception_just_reset.any())

    def test_reset_row_skips_only_the_transition_that_contains_reset(self):
        state = self._state(
            4,
            torch.tensor([3, 3]),
            torch.tensor([False, True]),
        )

        def reset(env, env_ids):
            del env
            state.ego_generation += 1
            state.ego_cache_key = None
            state.observation_dirty = True
            state.perception_just_reset[env_ids] = True

        active_masks = []
        def advance(env):
            del env
            active_masks.append((~state.perception_just_reset).clone())
            state.perception_just_reset.zero_()

        state._reset = reset
        state._advance = advance
        state._build_observation = lambda env: state.cached_observation

        env = types.SimpleNamespace(
            common_step_counter=5,
            episode_length_buf=torch.tensor([4, 0]),
        )
        state.observe(env)
        env.common_step_counter = 6
        env.episode_length_buf = torch.tensor([5, 1])
        state.observe(env)

        self.assertTrue(torch.equal(active_masks[0], torch.tensor([True, False])))
        self.assertTrue(torch.equal(active_masks[1], torch.tensor([True, True])))
        self.assertFalse(state.perception_just_reset.any())

    def test_ego_sample_is_cached_within_a_step(self):
        state = self._state(
            2,
            torch.tensor([1, 1]),
            torch.tensor([False, False]),
        )
        sample_count = 0

        def sample(env):
            nonlocal sample_count
            del env
            sample_count += 1

        state._sample_observed_velocities = sample
        env = types.SimpleNamespace(
            common_step_counter=3,
            episode_length_buf=torch.tensor([2, 2]),
        )

        state.prepare_ego_observation(env)
        state.prepare_ego_observation(env)

        self.assertEqual(sample_count, 1)

    def test_observed_pose_uses_episode_bias(self):
        state = belief.CVKFBeliefState.__new__(belief.CVKFBeliefState)
        state.robot_name = "robot"
        state.ego_position_bias = torch.tensor([[0.25, -0.50]])
        state.ego_yaw_bias = torch.tensor([0.30])
        state.observed_base_position = torch.zeros(1, 2)
        state.observed_base_yaw = torch.zeros(1)
        robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 0.5]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
        env = types.SimpleNamespace(scene={"robot": robot})

        state._update_observed_base_pose(env)

        torch.testing.assert_close(
            state.observed_base_position,
            torch.tensor([[1.25, 1.50]]),
        )
        torch.testing.assert_close(state.observed_base_yaw, torch.tensor([0.30]))

    def test_two_hundred_degree_fov_extends_behind_frontal_half_plane(self):
        angles = torch.deg2rad(torch.tensor([95.0, -95.0, 105.0, -105.0]))
        points = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        visible = belief._horizontal_fov_mask(points, belief._VISION_FOV_YAW_RAD)

        self.assertTrue(
            torch.equal(visible, torch.tensor([True, True, False, False]))
        )
        self.assertAlmostEqual(belief._VISION_FOV_YAW_RAD, math.radians(200.0), places=6)
        self.assertEqual(belief._VISION_MAX_DISTANCE_M, 6.0)

    def test_actor_angular_velocity_and_target_use_shared_ego_state(self):
        observed_ang_vel = torch.tensor([[0.1, -0.2, 0.3]])
        state = types.SimpleNamespace(
            robot_name="robot",
            ball_name="soccer_ball",
            observed_base_ang_vel=observed_ang_vel,
            observed_base_yaw=torch.tensor([math.pi / 2.0]),
            prepare_ego_observation=lambda env: None,
        )
        command_manager = types.SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.0, 1.0, 2.0]])
        )
        env = types.SimpleNamespace(command_manager=command_manager)
        setattr(env, belief._STATE_ATTRIBUTE, state)

        angular = belief.observed_base_ang_vel(env)
        target = belief.observed_kick_direction(env)

        self.assertIs(angular, observed_ang_vel)
        torch.testing.assert_close(target, torch.tensor([[0.0, -1.0]]), atol=1.0e-6, rtol=0.0)

    def test_belief_forecasts_ego_translation_and_rotating_frame_velocity(self):
        horizon_count = len(belief.PREDICTION_HORIZONS_S)
        means = torch.zeros(1, horizon_count, 4)
        means[:, :, 0] = 1.0
        covariance = torch.eye(4).view(1, 1, 4, 4).repeat(1, horizon_count, 1, 1)
        covariance *= 1.0e-4

        class Filter:
            state = torch.zeros(1, 4)

            def forecast_offsets(self, offsets, process_acceleration_std):
                del offsets, process_acceleration_std
                return means, covariance

        state = belief.CVKFBeliefState.__new__(belief.CVKFBeliefState)
        state.num_envs = 1
        state.device = torch.device("cpu")
        state.filter = Filter()
        state.process_acceleration_std = torch.zeros(1)
        state.perception_latency = torch.zeros(1)
        state.observed_base_position = torch.zeros(1, 2)
        state.observed_base_yaw = torch.zeros(1)
        state.observed_base_lin_vel_yaw = torch.tensor([[1.0, 0.0, 0.0]])
        state.observed_base_ang_vel = torch.zeros(1, 3)
        state.belief_valid = torch.tensor([True])
        state.last_measurement_age = torch.zeros(1)
        state.measurement_updated = torch.zeros(1, dtype=torch.bool)

        observation = state._build_observation(types.SimpleNamespace())
        one_second_index = belief.PREDICTION_HORIZONS_S.index(1.0)
        one_second_token_start = one_second_index * 6

        # The base advances to the stationary ball over one second.
        torch.testing.assert_close(
            observation[0, one_second_token_start],
            torch.tensor(0.0),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            observation[0, horizon_count * 6 : horizon_count * 6 + 2],
            torch.tensor([-1.0, 0.0]),
            atol=1.0e-6,
            rtol=0.0,
        )

        state.observed_base_lin_vel_yaw.zero_()
        state.observed_base_ang_vel[:, 2] = 1.0
        rotating_observation = state._build_observation(types.SimpleNamespace())
        # At horizon zero, +1 rad/s yaw makes a stationary ball one metre
        # ahead appear to move laterally at -1 m/s in the rotating frame.
        torch.testing.assert_close(
            rotating_observation[0, horizon_count * 6 : horizon_count * 6 + 2],
            torch.tensor([0.0, -1.0]),
            atol=1.0e-6,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
