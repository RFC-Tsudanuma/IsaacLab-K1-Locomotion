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

from ego_motion import (  # noqa: E402
    forecast_constant_body_twist,
    forecast_constant_body_twist_variance,
    relative_velocity_from_world,
)


class EgoMotionTest(unittest.TestCase):
    def test_translation_without_yaw_rate(self):
        position, yaw = forecast_constant_body_twist(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([0.0]),
            torch.tensor([[2.0, -1.0]]),
            torch.tensor([0.0]),
            torch.tensor([[0.0, 0.5]]),
        )
        torch.testing.assert_close(position[0, 1], torch.tensor([2.0, 1.5]))
        torch.testing.assert_close(yaw[0], torch.tensor([0.0, 0.0]))

    def test_constant_twist_follows_circular_arc(self):
        position, yaw = forecast_constant_body_twist(
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1.0]),
            torch.tensor([[torch.pi / 2.0]]),
        )
        torch.testing.assert_close(position[0, 0], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(yaw[0, 0], torch.tensor(torch.pi / 2.0))

    def test_relative_velocity_includes_rotating_frame_term(self):
        relative_velocity = relative_velocity_from_world(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.ones(1),
            torch.tensor([[1.0, 0.0]]),
        )
        torch.testing.assert_close(relative_velocity[0], torch.tensor([0.0, -1.0]))

    def test_ego_variance_grows_with_horizon(self):
        position_variance, yaw_variance = forecast_constant_body_twist_variance(
            torch.tensor([[0.0, 1.0]]),
            0.01,
            0.02,
            0.1,
            0.03,
            0.04,
            0.005,
            0.01,
            0.15,
            0.05,
            0.02,
        )
        self.assertTrue(torch.all(position_variance[:, 1] > position_variance[:, 0]))
        self.assertTrue(torch.all(yaw_variance[:, 1] > yaw_variance[:, 0]))


if __name__ == "__main__":
    unittest.main()
