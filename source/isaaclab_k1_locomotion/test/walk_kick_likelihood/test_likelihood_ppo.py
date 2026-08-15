import sys
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict


TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick_likelihood"
)
sys.path.insert(0, str(TASK_DIR))

from agents.model import DirectKickingActorCritic  # noqa: E402
from agents.ppo import DirectKickingPPO  # noqa: E402


def make_observations(num_envs):
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 132),
            "critic": torch.randn(num_envs, 20),
        },
        batch_size=[num_envs],
    )


class DirectKickingPPOTest(unittest.TestCase):
    def test_weighted_mirror_term_runs_inside_rsl_update(self):
        num_envs = 4
        observations = make_observations(num_envs)
        model = DirectKickingActorCritic(
            observations,
            {"policy": ["policy"], "critic": ["policy", "critic"]},
            12,
        )
        algorithm = DirectKickingPPO(
            model,
            symmetric_coef=10.0,
            num_learning_epochs=1,
            num_mini_batches=1,
            desired_kl=None,
        )
        algorithm.init_storage("rl", num_envs, 2, observations, [12])

        for _ in range(2):
            algorithm.act(observations)
            next_observations = make_observations(num_envs)
            algorithm.process_env_step(
                next_observations,
                torch.randn(num_envs),
                torch.zeros(num_envs, dtype=torch.uint8),
                {},
            )
            observations = next_observations
        algorithm.compute_returns(observations)

        losses = algorithm.update()

        self.assertTrue(algorithm.use_direct_kicking_symmetry)
        self.assertEqual(losses["mirror_consistency_coefficient"], 10.0)
        self.assertGreaterEqual(losses["mirror_consistency_weight"], 0.0)
        self.assertLessEqual(losses["mirror_consistency_weight"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(losses["mirror_consistency"])))
        self.assertEqual(algorithm.storage.step, 0)

    def test_negative_coefficient_is_rejected_before_training(self):
        observations = make_observations(2)
        model = DirectKickingActorCritic(
            observations,
            {"policy": ["policy"], "critic": ["critic"]},
            12,
        )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            DirectKickingPPO(model, symmetric_coef=-1.0)


if __name__ == "__main__":
    unittest.main()
