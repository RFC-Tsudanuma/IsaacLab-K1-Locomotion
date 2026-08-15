import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick_likelihood"
)
sys.path.insert(0, str(TASK_DIR))

from agents.model import DirectKickingActorCritic  # noqa: E402
from agents.runner import DirectKickingOnPolicyRunner  # noqa: E402


def make_model(*, prediction_horizons_s=None):
    observations = {
        "policy": torch.zeros(2, 132),
        "critic": torch.zeros(2, 20),
    }
    model_kwargs = {}
    if prediction_horizons_s is not None:
        model_kwargs["prediction_horizons_s"] = prediction_horizons_s
    return DirectKickingActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        12,
        **model_kwargs,
    )


def make_uninitialized_runner(model):
    runner = object.__new__(DirectKickingOnPolicyRunner)
    runner.device = "cpu"
    runner.current_learning_iteration = 0
    runner.logger_type = "tensorboard"
    runner.disable_logs = True
    runner.alg = SimpleNamespace(
        policy=model,
        optimizer=torch.optim.Adam(model.parameters()),
    )
    return runner


class DirectKickingCheckpointCompatibilityTest(unittest.TestCase):
    def test_model_1000_weights_only_loads_strictly_when_present(self):
        checkpoint_path = REPOSITORY_ROOT / "model_1000.pth"
        if not checkpoint_path.exists():
            self.skipTest("model_1000.pth is not present in this checkout")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = make_model()

        resumed = model.load_state_dict(checkpoint["model"], strict=True)

        self.assertTrue(resumed)
        self.assertEqual(checkpoint["model_metadata"], model.checkpoint_metadata())
        self.assertEqual(len(model.state_dict()), 25)
        for key, expected in checkpoint["model"].items():
            torch.testing.assert_close(model.state_dict()[key], expected)

    def test_runner_loads_source_model_key_after_metadata_validation(self):
        checkpoint_path = REPOSITORY_ROOT / "model_1000.pth"
        if not checkpoint_path.exists():
            self.skipTest("model_1000.pth is not present in this checkout")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = make_model()
        runner = make_uninitialized_runner(model)

        runner.load(checkpoint_path, load_optimizer=False, map_location="cpu")

        for key, expected in checkpoint["model"].items():
            torch.testing.assert_close(model.state_dict()[key], expected)
        self.assertEqual(runner.current_learning_iteration, 1000)

    def test_runner_accepts_direct_weights_checkpoint_without_optimizer(self):
        source_model = make_model()
        checkpoint = {
            "model": source_model.state_dict(),
            "model_metadata": source_model.checkpoint_metadata(),
            "iteration": 7,
        }
        target_model = make_model()
        runner = make_uninitialized_runner(target_model)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "weights_only.pth"
            torch.save(checkpoint, checkpoint_path)
            runner.load(checkpoint_path, load_optimizer=False, map_location="cpu")

        self.assertEqual(runner.current_learning_iteration, 7)
        for key, expected in source_model.state_dict().items():
            torch.testing.assert_close(target_model.state_dict()[key], expected)

    def test_native_save_preserves_rsl_keys_and_model_metadata(self):
        model = make_model()
        runner = make_uninitialized_runner(model)
        runner.current_learning_iteration = 11

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "native.pt"
            runner.save(checkpoint_path, infos={"origin": "test"})
            checkpoint = torch.load(
                checkpoint_path,
                weights_only=True,
                map_location="cpu",
            )

        self.assertEqual(
            set(checkpoint),
            {
                "model_state_dict",
                "optimizer_state_dict",
                "iter",
                "infos",
                "model_metadata",
            },
        )
        self.assertEqual(checkpoint["model_metadata"], model.checkpoint_metadata())
        self.assertEqual(checkpoint["iter"], 11)
        self.assertEqual(checkpoint["infos"], {"origin": "test"})

    def test_native_load_validates_same_shape_horizon_metadata(self):
        source_horizons = [
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
            4.00,
        ]
        source_runner = make_uninitialized_runner(
            make_model(prediction_horizons_s=source_horizons)
        )
        target_runner = make_uninitialized_runner(make_model())

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "native.pt"
            source_runner.save(checkpoint_path)
            with self.assertRaisesRegex(ValueError, "model_metadata"):
                target_runner.load(
                    checkpoint_path,
                    load_optimizer=False,
                    map_location="cpu",
                )

    def test_native_load_restores_iteration_without_optimizer(self):
        source_runner = make_uninitialized_runner(make_model())
        source_runner.current_learning_iteration = 13
        target_runner = make_uninitialized_runner(make_model())

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "native.pt"
            source_runner.save(checkpoint_path)
            target_runner.load(
                checkpoint_path,
                load_optimizer=False,
                map_location="cpu",
            )

        self.assertEqual(target_runner.current_learning_iteration, 13)


if __name__ == "__main__":
    unittest.main()
