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

from observation_contract import (  # noqa: E402
    HORIZON_TOKEN_SIZE,
    NORMALIZED_HORIZON_INDEX,
    build_horizon_tokens,
    expected_direct_kicking_observation_size,
)


HORIZONS = (
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


class ObservationContractTest(unittest.TestCase):
    def test_expected_observation_size_is_132(self):
        self.assertEqual(expected_direct_kicking_observation_size(len(HORIZONS)), 132)

    def test_tokens_include_normalized_horizon_in_fixed_order(self):
        relative_position = torch.zeros(1, len(HORIZONS), 2)
        relative_position[0, :, 0] = torch.arange(len(HORIZONS))
        log_std = torch.full_like(relative_position, 0.25)
        correlation = torch.full((1, len(HORIZONS)), -0.4)

        tokens = build_horizon_tokens(
            relative_position,
            log_std,
            correlation,
            HORIZONS,
            torch.tensor([True]),
            invalid_log_std=1.0,
        )

        self.assertEqual(tuple(tokens.shape), (1, len(HORIZONS), HORIZON_TOKEN_SIZE))
        self.assertTrue(torch.allclose(tokens[0, :, 0], relative_position[0, :, 0]))
        self.assertTrue(torch.allclose(tokens[0, :, 2:4], log_std[0]))
        self.assertTrue(torch.allclose(tokens[0, :, 4], correlation[0]))
        expected_time = torch.tensor(HORIZONS) / HORIZONS[-1]
        self.assertTrue(
            torch.allclose(tokens[0, :, NORMALIZED_HORIZON_INDEX], expected_time)
        )

    def test_invalid_belief_keeps_time_and_max_uncertainty(self):
        relative_position = torch.randn(1, len(HORIZONS), 2)
        log_std = torch.randn_like(relative_position)
        correlation = torch.randn(1, len(HORIZONS))

        tokens = build_horizon_tokens(
            relative_position,
            log_std,
            correlation,
            HORIZONS,
            torch.tensor([False]),
            invalid_log_std=2.0,
        )

        self.assertTrue(torch.all(tokens[0, :, :2] == 0.0))
        self.assertTrue(torch.all(tokens[0, :, 2:4] == 2.0))
        self.assertTrue(torch.all(tokens[0, :, 4] == 0.0))
        expected_time = torch.tensor(HORIZONS) / HORIZONS[-1]
        self.assertTrue(
            torch.allclose(tokens[0, :, NORMALIZED_HORIZON_INDEX], expected_time)
        )


if __name__ == "__main__":
    unittest.main()
