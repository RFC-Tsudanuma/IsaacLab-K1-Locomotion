"""CPU rollout, checkpoint and export integration using a mutating fake env."""

import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from test_source_parity import load_port


Runner = load_port("runner").DirectKickRunner
KickEpisodeMetrics = load_port("episode_metrics").KickEpisodeMetrics


def small_config():
    cfg = load_port("config").load_config()
    cfg["runner"].update(horizon_length=3, mini_epochs=2, optimization_chunk_size=4,
                         save_interval=1, use_wandb=False)
    return cfg


class ReusedTensorEnv:
    """Keep all observation/extra containers alive while emulating auto-reset."""

    def __init__(self):
        self.unwrapped = self
        self.num_envs = 2
        self.observations = {"policy": torch.zeros(2, 325), "critic": torch.zeros(2, 20)}
        self.extras = {"post_kick_phase_target": torch.zeros(2)}
        self.seen_actions = []
        self.tick = 0
        self.episode_metrics = KickEpisodeMetrics(self.num_envs, "cpu")

    def start_episodes(self, ids):
        self.episode_metrics.start(ids, torch.tensor([0., 3.])[ids],
                                   torch.tensor([True, False])[ids], torch.tensor([0., 0.4])[ids])

    def write_observations(self):
        self.observations["policy"][0].fill_(self.tick * 0.1)
        self.observations["policy"][1].fill_(self.tick * 0.1 + 0.01)
        self.observations["policy"][:, 322] = 1.0
        self.observations["policy"][:, 323] = 1.0
        self.observations["policy"][:, 324] = 0.0
        self.observations["critic"].fill_(self.tick * 0.2)

    def reset(self):
        self.tick = 0
        self.seen_actions.clear()
        self.start_episodes(torch.arange(self.num_envs))
        self.write_observations()
        self.extras["post_kick_phase_target"].copy_(torch.tensor([0.0, 1.0]))
        return self.observations, self.extras

    def step(self, actions):
        if actions.shape != (2, 12):
            raise AssertionError("Only the 12 joint actions belong to env.step")
        self.seen_actions.append(actions.detach().clone())
        self.tick += 1
        self.write_observations()
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        if self.tick == 1:
            # Environment 1 ends and is reset; its new phase is false.
            terminated[1] = True
            labels = [1.0, 0.0]
        elif self.tick == 2:
            # Environment 0 times out and is reset; its new phase is false.
            truncated[0] = True
            labels = [0.0, 1.0]
        else:
            labels = [1.0, 1.0]
        finished = (terminated | truncated).nonzero(as_tuple=False).flatten()
        self.episode_metrics.finish(finished, terminated[finished], torch.zeros_like(terminated[finished]),
                                    truncated[finished], terminated[finished])
        self.start_episodes(finished)
        self.extras["post_kick_phase_target"].copy_(torch.tensor(labels))
        rewards = torch.tensor([0.2 * self.tick, -0.1 * self.tick])
        return self.observations, rewards, terminated, truncated, self.extras


class RunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(500)
        self.directory = tempfile.TemporaryDirectory(prefix="direct_kick_runner_test_")
        self.addCleanup(self.directory.cleanup)
        self.log_dir = Path(self.directory.name)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype))

    def test_collect_snapshots_phase_before_step_and_auto_reset(self):
        env = ReusedTensorEnv()
        runner = Runner(env, small_config(), self.log_dir, "cpu")
        observations, extras = env.reset()
        original_extras = extras
        original_phase_tensor = extras["post_kick_phase_target"]
        rollout, final_observations, final_extras = runner.collect(observations, extras)
        self.assertIs(final_extras, original_extras)
        self.assertIs(final_extras["post_kick_phase_target"], original_phase_tensor)
        self.assertIs(final_observations, observations)
        self.assert_close(rollout["post_kick_phase_targets"], [[0, 1], [1, 0], [0, 1]])
        self.assert_close(rollout["obses"][:, :, 0], [[0, 0.01], [0.1, 0.11], [0.2, 0.21]])
        self.assert_close(rollout["privileged_obses"][:, :, 0], [[0, 0], [0.2, 0.2], [0.4, 0.4]])
        self.assertEqual(rollout["dones"].tolist(), [[False, True], [True, False], [False, False]])
        self.assertEqual(rollout["time_outs"].tolist(), [[False, False], [True, False], [False, False]])
        self.assert_close(rollout["actions"], torch.stack(env.seen_actions))
        self.assert_close(final_extras["post_kick_phase_target"], [1, 1])
        self.assertFalse(rollout["obses"].requires_grad)

    def test_one_iteration_updates_saves_resumes_and_exports(self):
        env = ReusedTensorEnv()
        runner = Runner(env, small_config(), self.log_dir / "train", "cpu")
        initial = {key: value.clone() for key, value in runner.model.state_dict().items()}
        path = runner.train(max_iterations=1)
        self.assertTrue(path.exists())
        self.assertEqual(runner.iteration, 1)
        self.assertEqual(runner.total_steps, 6)
        self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in runner.model.state_dict().items()))
        self.assertTrue(all(state["step"].item() == 2 for state in runner.ppo.optimizer.state.values()))
        with (runner.log_dir / "learning.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["iteration"], "1")
        self.assertEqual(rows[0]["total_steps"], "6")
        self.assertTrue(all(math.isfinite(float(rows[0][key])) for key in ("value_loss", "phase_loss", "kl")))

        with (runner.log_dir / "episode_metrics.csv").open() as stream:
            episodes = {row["group"]: row for row in csv.DictReader(stream)}
        self.assertEqual(episodes["all"]["episodes"], "2")
        self.assertEqual(episodes["all"]["kick_rate"], "0.5")
        self.assertEqual(episodes["stationary"]["kick_rate"], "0.0")
        self.assertEqual(episodes["moving"]["kick_rate"], "1.0")
        self.assertEqual(episodes["speed_5_6_mps"]["kick_rate"], "")
        self.assertEqual(runner.env.episode_metrics.summary()["all"]["episodes"], 0)

        resumed = Runner(ReusedTensorEnv(), small_config(), self.log_dir / "resume", "cpu")
        checkpoint = resumed.load(path)
        self.assertEqual(resumed.iteration, 1)
        self.assertEqual(resumed.total_steps, 6)
        self.assertEqual(checkpoint["port_metadata"], runner.metadata)
        for key, value in runner.model.state_dict().items():
            self.assert_close(resumed.model.state_dict()[key], value)
        original_optimizer = runner.ppo.optimizer.state_dict()
        restored_optimizer = resumed.ppo.optimizer.state_dict()
        self.assertEqual(restored_optimizer["param_groups"], original_optimizer["param_groups"])
        for parameter, state in original_optimizer["state"].items():
            for key, value in state.items():
                self.assert_close(restored_optimizer["state"][parameter][key], value)

        weights_only = Runner(ReusedTensorEnv(), small_config(), self.log_dir / "weights", "cpu")
        weights_only.load(path, resume=False)
        self.assertEqual(weights_only.iteration, 0)
        self.assertEqual(weights_only.total_steps, 0)
        self.assertFalse(weights_only.ppo.optimizer.state)
        self.assertEqual(weights_only.ppo.learning_rate, small_config()["algorithm"]["learning_rate"])
        self.assertEqual(weights_only.ppo.optimizer.param_groups[0]["lr"], weights_only.ppo.learning_rate)
        for key, value in runner.model.state_dict().items():
            self.assert_close(weights_only.model.state_dict()[key], value)

        export_path = self.log_dir / "export" / "policy.pt"
        observations = torch.randn(3, 325)
        with torch.no_grad():
            expected = runner.model.actor(observations)
        runner.export(export_path)
        scripted = torch.jit.load(str(export_path), map_location="cpu")
        actual = scripted(observations)
        self.assertEqual(tuple(actual.shape), (3, 13))
        self.assert_close(actual, expected)
        self.assert_close(actual[:, :12], runner.model.act(observations).loc)
        self.assertTrue(((actual[:, 12] >= 0) & (actual[:, 12] <= 1)).all())
        self.assertEqual(json.loads(export_path.with_suffix(".json").read_text()), runner.metadata)

    def test_resume_matches_uninterrupted_adaptive_lr_and_two_further_updates(self):
        cfg = small_config()
        cfg["algorithm"].update(learning_rate=1e-4, desired_kl=1e6)
        original = Runner(ReusedTensorEnv(), cfg, self.log_dir / "original", "cpu")
        path = original.train(max_iterations=1)
        self.assertNotEqual(original.ppo.learning_rate, cfg["algorithm"]["learning_rate"])
        restored = Runner(ReusedTensorEnv(), copy.deepcopy(cfg), self.log_dir / "restored", "cpu")
        restored.load(path)
        self.assertEqual(restored.ppo.learning_rate, original.ppo.learning_rate)
        observations, extras = original.env.reset()
        for _ in range(2):
            rollout, observations, extras = original.collect(observations, extras)
            for runner in (original, restored):
                # update() replaces timeout rewards in place.
                runner.ppo.update(copy.deepcopy(rollout), observations["policy"].clone(),
                                  observations["critic"].clone())
            self.assertEqual(restored.ppo.learning_rate, original.ppo.learning_rate)
            for key, value in original.model.state_dict().items():
                self.assert_close(restored.model.state_dict()[key], value)
            expected = original.ppo.optimizer.state_dict()
            actual = restored.ppo.optimizer.state_dict()
            self.assertEqual(actual["param_groups"], expected["param_groups"])
            for parameter, state in expected["state"].items():
                for key, value in state.items():
                    self.assert_close(actual["state"][parameter][key], value)

    def test_port_metadata_gate_rejects_missing_and_changed_perception_before_load(self):
        source = Runner(ReusedTensorEnv(), small_config(), self.log_dir / "source", "cpu")
        checkpoint = torch.load(source.save(), map_location="cpu", weights_only=False)
        target = Runner(ReusedTensorEnv(), small_config(), self.log_dir / "target", "cpu")
        before = {key: value.clone() for key, value in target.model.state_dict().items()}
        for variant in ("missing", "changed_perception", "changed_model", "old_position_only"):
            with self.subTest(variant=variant):
                incompatible = copy.deepcopy(checkpoint)
                if variant == "missing":
                    incompatible.pop("port_metadata")
                elif variant == "changed_perception":
                    incompatible["port_metadata"]["migration"]["perception_schema"] = "source_cv_kalman_filter"
                elif variant == "old_position_only":
                    incompatible["port_metadata"]["model"]["observation_schema"] = "direct_kicking_horizon_lstm_direction_only_v2"
                    incompatible["port_metadata"]["model"]["num_observations"] = 132
                    incompatible["port_metadata"]["model"]["horizon_token_size"] = 6
                else:
                    incompatible["port_metadata"]["model"]["num_policy_outputs"] = 12
                path = self.log_dir / f"{variant}.pth"
                torch.save(incompatible, path)
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    target.load(path)
                for key, value in before.items():
                    self.assert_close(target.model.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
