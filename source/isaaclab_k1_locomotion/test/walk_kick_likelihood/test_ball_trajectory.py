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

from ball_trajectory import build_ball_trajectory  # noqa: E402


class BallTrajectoryTest(unittest.TestCase):
    def test_incoming_trajectory_keeps_base_speed_and_passes_at_offset(self):
        spawn, velocity = build_ball_trajectory(
            spawn_distance=torch.tensor([2.0]),
            spawn_bearing=torch.tensor([0.3]),
            closest_approach_offset=torch.tensor([0.2]),
            base_speed=torch.tensor([0.8]),
            incoming=torch.tensor([True]),
        )

        radial_dot = torch.sum(spawn * velocity, dim=-1)
        speed = torch.norm(velocity, dim=-1)
        signed_closest_approach = (
            spawn[:, 0] * velocity[:, 1] - spawn[:, 1] * velocity[:, 0]
        ) / speed

        self.assertLess(radial_dot.item(), 0.0)
        torch.testing.assert_close(speed, torch.tensor([0.8]))
        torch.testing.assert_close(
            signed_closest_approach,
            torch.tensor([0.2]),
        )

    def test_outgoing_trajectory_keeps_base_speed(self):
        spawn, velocity = build_ball_trajectory(
            spawn_distance=torch.tensor([1.5, 6.0]),
            spawn_bearing=torch.tensor([-0.2, 0.4]),
            closest_approach_offset=torch.tensor([0.25, -0.25]),
            base_speed=torch.tensor([0.6, 1.0]),
            incoming=torch.tensor([False, False]),
        )

        radial_dot = torch.sum(spawn * velocity, dim=-1)
        self.assertTrue(torch.all(radial_dot > 0.0))
        torch.testing.assert_close(
            torch.norm(velocity, dim=-1),
            torch.tensor([0.6, 1.0]),
        )

    def test_spawn_bearing_and_distance_are_preserved(self):
        distance = torch.tensor([1.5, 6.0, 1.5, 6.0])
        bearing = torch.tensor([-0.87266463, 0.87266463, -0.4, 0.4])

        spawn, _ = build_ball_trajectory(
            spawn_distance=distance,
            spawn_bearing=bearing,
            closest_approach_offset=torch.zeros(4),
            base_speed=torch.ones(4),
            incoming=torch.tensor([True, True, False, False]),
        )

        torch.testing.assert_close(torch.norm(spawn, dim=-1), distance)
        torch.testing.assert_close(torch.atan2(spawn[:, 1], spawn[:, 0]), bearing)


if __name__ == "__main__":
    unittest.main()
