import sys
import unittest
from pathlib import Path

import torch


TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick_likelihood"
)
sys.path.insert(0, str(TASK_DIR))

from agents.symmetry import (  # noqa: E402
    BELIEF_STATUS_SIZE,
    BELIEF_VELOCITY_SIZE,
    HORIZON_TOKEN_SIZE,
    LOCOMOTION_OBSERVATION_SIZE,
    expected_direct_kicking_observation_size,
    kick_feasibility_ambiguity_weight,
    mirror_direct_kicking_observation,
    mirror_leg_actions,
    weighted_mirror_consistency_loss,
)


class DirectKickingSymmetryTest(unittest.TestCase):
    def test_action_mirror_is_an_involution_for_batched_actions(self):
        actions = torch.randn(3, 5, 12)

        mirrored_twice = mirror_leg_actions(mirror_leg_actions(actions))

        torch.testing.assert_close(mirrored_twice, actions)

    def test_observation_mirror_is_an_involution(self):
        horizon_count = 13
        observation = torch.randn(
            4,
            expected_direct_kicking_observation_size(horizon_count),
        )

        mirrored_twice = mirror_direct_kicking_observation(
            mirror_direct_kicking_observation(observation, horizon_count),
            horizon_count,
        )

        torch.testing.assert_close(mirrored_twice, observation)

    def test_ambiguity_weight_has_exact_source_transition_and_validity(self):
        horizon_count = 1
        observation = torch.zeros(
            4,
            expected_direct_kicking_observation_size(horizon_count),
        )
        forecast_start = LOCOMOTION_OBSERVATION_SIZE
        forecast_end = forecast_start + horizon_count * HORIZON_TOKEN_SIZE
        valid_index = forecast_end + BELIEF_VELOCITY_SIZE + BELIEF_STATUS_SIZE - 1
        observation[:, valid_index] = 1.0

        # Equal foot costs, clearly-left feasible, transition t=.75, then invalid.
        observation[0, forecast_start : forecast_start + 2] = torch.tensor([0.185, 0.0])
        observation[1, forecast_start : forecast_start + 2] = torch.tensor([0.185, 0.096])
        observation[2, forecast_start : forecast_start + 2] = torch.tensor([0.185, 0.040])
        observation[3] = observation[1]
        observation[3, valid_index] = 0.0

        weight = kick_feasibility_ambiguity_weight(
            observation,
            horizon_count,
            nominal_strike_point_m=(0.185, 0.096),
            cost_gap_m=(0.02, 0.10),
            ball_position_observation_scale=1.0,
        )

        torch.testing.assert_close(
            weight,
            torch.tensor([0.0, 1.0, 0.84375, 0.0]),
        )

    def test_ambiguity_weight_is_mirror_invariant(self):
        horizon_count = 13
        observation = torch.randn(
            7,
            expected_direct_kicking_observation_size(horizon_count),
        )
        forecast_end = LOCOMOTION_OBSERVATION_SIZE + horizon_count * HORIZON_TOKEN_SIZE
        valid_index = forecast_end + BELIEF_VELOCITY_SIZE + BELIEF_STATUS_SIZE - 1
        observation[:, valid_index] = torch.linspace(0.0, 1.0, observation.shape[0])

        weight = kick_feasibility_ambiguity_weight(
            observation,
            horizon_count,
            (0.185, 0.096),
            (0.02, 0.10),
            1.0,
        )
        mirrored_weight = kick_feasibility_ambiguity_weight(
            mirror_direct_kicking_observation(observation, horizon_count),
            horizon_count,
            (0.185, 0.096),
            (0.02, 0.10),
            1.0,
        )

        torch.testing.assert_close(mirrored_weight, weight)

    def test_weighted_loss_uses_action_sum_then_batch_mean(self):
        action_mean = torch.zeros(2, 12)
        mirrored_observation_action_mean = torch.ones(2, 12)
        weight = torch.tensor([1.0, 0.0])

        loss = weighted_mirror_consistency_loss(
            action_mean,
            mirrored_observation_action_mean,
            weight,
        )

        # The source loss sums 12 unit errors per active sample, then averages
        # [12, 0] over the batch.  This scale pairs with symmetric_coef=10.
        torch.testing.assert_close(loss, torch.tensor(6.0))


if __name__ == "__main__":
    unittest.main()
