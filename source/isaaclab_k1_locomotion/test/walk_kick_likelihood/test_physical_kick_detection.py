import importlib.util
import types
import unittest
from pathlib import Path

import torch


KICK_STATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "isaaclab_k1_locomotion"
    / "tasks"
    / "manager_based"
    / "walk_kick"
    / "mdp"
    / "kick_state.py"
)


def _load_kick_state_module():
    spec = importlib.util.spec_from_file_location("_walk_kick_state_test", KICK_STATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kick_state_module = _load_kick_state_module()


class _FakeRobot:
    def __init__(self, num_envs):
        self.data = types.SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3),
            root_lin_vel_w=torch.zeros(num_envs, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            body_pos_w=torch.zeros(num_envs, 2, 3),
        )

    def find_bodies(self, name):
        body_id = 0 if name == "left_foot_link" else 1
        return [body_id], [name]


class _FakeEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.step_dt = 0.02
        self.common_step_counter = 1
        self.episode_length_buf = torch.ones(num_envs, dtype=torch.long)
        self.robot = _FakeRobot(num_envs)
        self.ball = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=torch.tensor([[0.2, 0.0, 0.11]]).repeat(num_envs, 1),
                root_lin_vel_w=torch.zeros(num_envs, 3),
            )
        )
        self.scene = {"robot": self.robot, "soccer_ball": self.ball}
        command = torch.tensor([[0.0, 1.0, 2.0]]).repeat(num_envs, 1)
        self.command_manager = types.SimpleNamespace(get_command=lambda _name: command)
        self.robot.data.body_pos_w[:, 0] = torch.tensor([0.0, 0.0, 0.11])
        self.robot.data.body_pos_w[:, 1] = torch.tensor([0.0, 1.0, 0.11])


def _state(env, physical_kick_detection):
    return kick_state_module.kick_state(
        env,
        r_stance=0.25,
        alpha=0.5,
        v_thresh=0.8,
        physical_kick_detection=physical_kick_detection,
        kick_detection_foot_distance_threshold=0.23,
        kick_detection_min_foot_speed_towards_ball=0.2,
        kick_detection_velocity_change_threshold=0.5,
        kick_detection_warmup_steps=5,
    )


class PhysicalKickDetectionTest(unittest.TestCase):
    def test_untouched_one_mps_rolling_ball_does_not_latch(self):
        env = _FakeEnv()
        env.ball.data.root_lin_vel_w[0, 0] = 1.0

        state = _state(env, physical_kick_detection=True)
        self.assertFalse(state["kick_done"].item())

        for episode_step in range(2, 7):
            env.common_step_counter += 1
            env.episode_length_buf[0] = episode_step
            env.ball.data.root_pos_w[0, 0] += env.step_dt
            state = _state(env, physical_kick_detection=True)

        self.assertFalse(state["kick_done"].item())

    def test_qualifying_kick_latches_after_five_step_warmup(self):
        env = _FakeEnv()
        env.ball.data.root_lin_vel_w[0, :2] = torch.tensor([1.0, 0.0])
        # 右足は現在距離だけなら近いが静止しており、接触候補ではない。
        env.robot.data.body_pos_w[0, 1] = torch.tensor([0.15, 0.0, 0.11])
        _state(env, physical_kick_detection=True)

        # 接触条件と 0.6 m/s の XY 速度ベクトル変化を満たすが、step 4 は warmup 中。
        env.common_step_counter += 1
        env.episode_length_buf[0] = 4
        env.robot.data.body_pos_w[0, 0, 0] = 0.01
        env.ball.data.root_pos_w[0, 0] = 0.22
        env.ball.data.root_lin_vel_w[0, :2] = torch.tensor([0.8, 0.6])
        state = _state(env, physical_kick_detection=True)
        self.assertFalse(state["kick_done"].item())

        # step 5 では速さ 1.0 m/s のまま方向だけ変わる。scalar speed 差は 0 だが、
        # XY 速度ベクトル差は 0.5 m/s を超えるため latch する。
        env.common_step_counter += 1
        env.episode_length_buf[0] = 5
        env.robot.data.body_pos_w[0, 0, 0] = 0.02
        env.robot.data.body_pos_w[0, 0, 2] = 0.15
        env.ball.data.root_pos_w[0, 0] = 0.24
        env.ball.data.root_lin_vel_w[0, :2] = torch.tensor([0.28, 0.96])
        state = _state(env, physical_kick_detection=True)

        self.assertTrue(state["kick_done"].item())
        self.assertEqual(state["kick_foot_frozen"].item(), 0.0)
        self.assertEqual(state["touch_count"].item(), 1.0)
        self.assertAlmostEqual(state["sole_height_at_kick"].item(), 0.112, places=6)

    def test_partial_reset_rebases_only_reset_row(self):
        env = _FakeEnv(num_envs=2)
        env.ball.data.root_lin_vel_w[:, 0] = 1.0
        _state(env, physical_kick_detection=True)

        env.common_step_counter += 1
        env.episode_length_buf[:] = torch.tensor([1, 5])

        # env 0: reset と同時に大きく状態を変える。履歴差分として扱ってはならない。
        env.ball.data.root_pos_w[0] = torch.tensor([2.0, 0.0, 0.11])
        env.ball.data.root_lin_vel_w[0, :2] = torch.tensor([-1.0, 2.0])
        env.robot.data.body_pos_w[0, 0] = torch.tensor([1.9, 0.0, 0.11])

        # env 1: reset されていない履歴から正当な接触・速度変化を検出する。
        env.robot.data.body_pos_w[1, 0, 0] = 0.01
        env.ball.data.root_pos_w[1, 0] = 0.22
        env.ball.data.root_lin_vel_w[1, :2] = torch.tensor([1.0, 0.6])
        state = _state(env, physical_kick_detection=True)

        self.assertTrue(torch.equal(state["kick_done"], torch.tensor([False, True])))
        torch.testing.assert_close(state["prev_ball_pos_w"][0], env.ball.data.root_pos_w[0])

    def test_default_absolute_speed_trigger_is_unchanged(self):
        env = _FakeEnv()
        env.ball.data.root_lin_vel_w[0, 0] = 1.0

        state = _state(env, physical_kick_detection=False)

        self.assertTrue(state["kick_done"].item())
        self.assertEqual(state["v_ball_frozen"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
