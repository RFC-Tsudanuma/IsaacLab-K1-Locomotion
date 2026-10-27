"""MLP forecast encoding and compatibility with the existing LSTM contract."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_source_parity import HORIZONS, load_port
from test_runner_integration import ReusedTensorEnv, small_config


Model = load_port("model").DirectKickingActorCritic
Runner = load_port("runner").DirectKickRunner


class ObservationEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(723)
        self.directory = tempfile.TemporaryDirectory(prefix="direct_kick_encoder_test_")
        self.addCleanup(self.directory.cleanup)
        self.log_dir = Path(self.directory.name)

    def test_default_mlp_keeps_ball_features_exclusive_to_encoder(self):
        model = Model.from_config(12, 325, 20, small_config())
        self.assertEqual(model.encoder_type, "mlp")
        obs = torch.randn(2, 325, requires_grad=True)
        for branch in (model.actor, model.critic):
            with self.subTest(branch=type(branch).__name__):
                encoder = branch.encoder
                self.assertFalse(hasattr(encoder, "lstm"))
                layers = [module for module in encoder.mlp if isinstance(module, torch.nn.Linear)]
                self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                                 [(273, 128), (128, 64)])
                self.assertEqual(sum(parameter.numel() for parameter in encoder.parameters()), 43328)
                seen = []
                hook = encoder.mlp.register_forward_pre_hook(
                    lambda module, args: seen.append(args[0].detach().clone()))
                encoded = encoder(obs)
                hook.remove()
                torch.testing.assert_close(seen[0], obs[:, 47:320])
                self.assertEqual(tuple(encoded.shape), (2, 116))
                torch.testing.assert_close(encoded[:, :52], torch.cat((obs[:, :47], obs[:, 320:]), dim=-1))
                gradient = torch.autograd.grad(encoded[:, 52:].sum(), obs)[0]
                self.assertTrue((gradient[:, 47:320] != 0).all())
                self.assertEqual(torch.count_nonzero(gradient[:, :47]).item(), 0)
                self.assertEqual(torch.count_nonzero(gradient[:, 320:]).item(), 0)

        changed = obs.detach().clone()
        changed[:, 47:320] += 3.0
        privileged = torch.randn(2, 20)
        for branch in (model.actor, model.critic):
            with patch.object(branch.encoder.mlp, "forward", return_value=torch.zeros(2, 64)):
                if branch is model.actor:
                    torch.testing.assert_close(branch(obs), branch(changed))
                else:
                    torch.testing.assert_close(branch(obs, privileged), branch(changed, privileged))

    def test_legacy_config_preserves_lstm_metadata_and_state_dict(self):
        cfg = small_config()
        observation = cfg["direct_kicking"]["observation"]
        for key in ("encoder_type", "mlp_hidden_size", "mlp_output_size"):
            observation.pop(key)
        model = Model.from_config(12, 325, 20, cfg)
        self.assertEqual(model.encoder_type, "lstm")
        self.assertEqual(model.checkpoint_metadata(), {
            "model_class": "DirectKickingActorCritic",
            "observation_schema": "direct_kicking_horizon_lstm_state_covariance_v3",
            "num_actions": 12,
            "num_observations": 325,
            "num_privileged_observations": 20,
            "prediction_horizons_s": HORIZONS,
            "horizon_token_size": 21,
            "lstm_hidden_size": 64,
            "lstm_num_layers": 1,
            "policy_output_schema": "joint_action_12_post_kick_phase_probability_v1",
            "num_policy_outputs": 13,
            "post_kick_phase_output_index": 12,
        })
        original = Model(12, 325, 20, HORIZONS)
        model.load_state_dict(original.state_dict(), strict=True)
        obs = torch.randn(3, 325)
        privileged = torch.randn(3, 20)
        torch.testing.assert_close(model.actor(obs), original.actor(obs), rtol=0, atol=0)
        torch.testing.assert_close(model.est_value(obs, privileged), original.est_value(obs, privileged), rtol=0, atol=0)
        self.assertFalse(any(".mlp." in name for name in model.state_dict()))

    def test_old_lstm_config_resumes_with_explicit_lstm_and_exports(self):
        legacy_cfg = small_config()
        for key in ("encoder_type", "mlp_hidden_size", "mlp_output_size"):
            legacy_cfg["direct_kicking"]["observation"].pop(key)
        original = Runner(ReusedTensorEnv(), legacy_cfg, self.log_dir / "legacy", "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            path = original.train(1)
        cfg = small_config()
        cfg["direct_kicking"]["observation"]["encoder_type"] = "lstm"
        restored = Runner(ReusedTensorEnv(), cfg, self.log_dir / "restored", "cpu")
        restored.load(path)
        self.assertEqual(restored.metadata, original.metadata)
        self.assertEqual(restored.iteration, original.iteration)
        self.assertEqual(restored.total_steps, original.total_steps)
        for key, value in original.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
        expected_optimizer = original.ppo.optimizer.state_dict()
        actual_optimizer = restored.ppo.optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        for parameter, state in expected_optimizer["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(actual_optimizer["state"][parameter][key], value, rtol=0, atol=0)
        export_path = self.log_dir / "legacy.pt"
        restored.export(export_path)
        scripted = torch.jit.load(str(export_path))
        for batch_size in (1, 3):
            obs = torch.randn(batch_size, 325)
            torch.testing.assert_close(scripted(obs), original.model.actor(obs))

    def test_cross_encoder_checkpoint_rejected_before_loading(self):
        mlp = Runner(ReusedTensorEnv(), small_config(), self.log_dir / "mlp", "cpu")
        lstm_cfg = small_config()
        lstm_cfg["direct_kicking"]["observation"]["encoder_type"] = "lstm"
        lstm = Runner(ReusedTensorEnv(), lstm_cfg, self.log_dir / "lstm", "cpu")
        for source, target in ((mlp, lstm), (lstm, mlp)):
            with self.subTest(source=source.model.encoder_type):
                before = copy.deepcopy(target.model.state_dict())
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    target.load(source.save())
                for key, value in before.items():
                    torch.testing.assert_close(target.model.state_dict()[key], value, rtol=0, atol=0)
                self.assertFalse(target.ppo.optimizer.state)

    def test_mlp_training_updates_both_forecast_encoders(self):
        runner = Runner(ReusedTensorEnv(), small_config(), self.log_dir, "cpu")
        before = copy.deepcopy(runner.model.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            runner.train(1)
        for prefix in ("actor.encoder.mlp.", "critic.encoder.mlp."):
            self.assertTrue(any(not torch.equal(value, before[key])
                                for key, value in runner.model.state_dict().items()
                                if key.startswith(prefix)))
        self.assertEqual(runner.model.checkpoint_metadata()["encoder_type"], "mlp")
        self.assertEqual(runner.model.checkpoint_metadata()["mlp_hidden_size"], 128)
        self.assertEqual(runner.model.checkpoint_metadata()["mlp_output_size"], 64)


if __name__ == "__main__":
    unittest.main()
