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
PACKAGE_NAME = "_walk_kick_likelihood_events_test"


def _load_events_module():
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
    module_name = f"{PACKAGE_NAME}.events"
    spec = importlib.util.spec_from_file_location(module_name, MDP_DIR / "events.py")
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


events = _load_events_module()


class _FakeBall:
    def __init__(self, default_root_state):
        self.data = types.SimpleNamespace(default_root_state=default_root_state)
        self.written_state = None
        self.written_env_ids = None

    def write_root_state_to_sim(self, state, env_ids):
        self.written_state = state.clone()
        self.written_env_ids = env_ids.clone()


class _FakeScene(dict):
    def __init__(self, *args, env_origins, num_envs):
        super().__init__(*args)
        self.env_origins = env_origins
        self.num_envs = num_envs


class _FakePhysxView:
    def __init__(self, materials):
        self.materials = materials.clone()
        self.set_env_ids = None

    def get_material_properties(self):
        return self.materials.clone()

    def set_material_properties(self, materials, env_ids):
        self.materials = materials.clone()
        self.set_env_ids = env_ids.clone()


class MovingBallEventsTest(unittest.TestCase):
    def test_reset_uses_source_trajectory_in_robot_yaw_frame(self):
        env_ids = torch.tensor([0, 2])
        radius = 0.11
        default_root_state = torch.full((3, 13), 9.0)
        ball = _FakeBall(default_root_state)
        root_pos_w = torch.tensor(
            [
                [1.0, 2.0, 0.5],
                [3.0, 4.0, 0.5],
                [-1.0, 0.5, 0.5],
            ]
        )
        half_sqrt_two = math.sqrt(0.5)
        root_quat_w = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
            ]
        )
        robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=root_pos_w,
                root_quat_w=root_quat_w,
            )
        )
        env_origins = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, -0.1],
            ]
        )
        env = types.SimpleNamespace(
            device="cpu",
            scene=_FakeScene(
                {"robot": robot, "soccer_ball": ball},
                env_origins=env_origins,
                num_envs=3,
            ),
        )

        torch.manual_seed(0)
        events.reset_moving_ball_trajectory(
            env,
            env_ids,
            ball_radius=radius,
        )

        torch.manual_seed(0)
        incoming = torch.rand(2) < 0.5
        incoming_distance = 1.5 + 1.5 * torch.rand(2)
        outgoing_distance = 1.5 + 1.5 * torch.rand(2)
        spawn_distance = torch.where(incoming, incoming_distance, outgoing_distance)
        spawn_bearing = -0.87266463 + 1.74532926 * torch.rand(2)
        closest_offset = -0.25 + 0.5 * torch.rand(2)
        speed = torch.rand(2)
        local_spawn, local_velocity = events.build_ball_trajectory(
            spawn_distance,
            spawn_bearing,
            closest_offset,
            speed,
            incoming,
        )
        yaw = torch.tensor([0.0, math.pi / 2])
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        world_spawn = torch.stack(
            (
                cosine * local_spawn[:, 0] - sine * local_spawn[:, 1],
                sine * local_spawn[:, 0] + cosine * local_spawn[:, 1],
            ),
            dim=-1,
        )
        world_velocity = torch.stack(
            (
                cosine * local_velocity[:, 0] - sine * local_velocity[:, 1],
                sine * local_velocity[:, 0] + cosine * local_velocity[:, 1],
            ),
            dim=-1,
        )

        state = ball.written_state
        torch.testing.assert_close(
            state[:, :2],
            root_pos_w[env_ids, :2] + world_spawn,
        )
        torch.testing.assert_close(
            state[:, 2],
            env_origins[env_ids, 2] + radius,
        )
        torch.testing.assert_close(
            state[:, 3:7],
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        )
        torch.testing.assert_close(state[:, 7:9], world_velocity)
        torch.testing.assert_close(state[:, 9], torch.zeros(2))
        torch.testing.assert_close(state[:, 10], -world_velocity[:, 1] / radius)
        torch.testing.assert_close(state[:, 11], world_velocity[:, 0] / radius)
        torch.testing.assert_close(state[:, 12], torch.zeros(2))
        self.assertTrue(torch.equal(ball.written_env_ids, env_ids))

    def test_startup_friction_is_continuous_per_env_and_preserves_restitution(self):
        materials = torch.tensor(
            [
                [[0.1, 0.2, 0.01], [0.3, 0.4, 0.02]],
                [[0.5, 0.6, 0.11], [0.7, 0.8, 0.12]],
                [[0.9, 1.0, 0.21], [1.1, 1.2, 0.22]],
                [[1.3, 1.4, 0.31], [1.5, 1.6, 0.32]],
            ]
        )
        view = _FakePhysxView(materials)
        ball = types.SimpleNamespace(root_physx_view=view)
        scene = _FakeScene(
            {"soccer_ball": ball},
            env_origins=torch.zeros(4, 3),
            num_envs=4,
        )
        env = types.SimpleNamespace(scene=scene)
        asset_cfg = events.SceneEntityCfg("soccer_ball")
        cfg = types.SimpleNamespace(
            params={
                "asset_cfg": asset_cfg,
                "friction_range": (0.9, 1.3),
            }
        )
        term = events.RandomizeBallFriction(cfg, env)
        env_ids = torch.tensor([1, 3])

        torch.manual_seed(4)
        term(
            env,
            env_ids,
            friction_range=(0.9, 1.3),
            asset_cfg=asset_cfg,
        )

        sampled = view.materials[env_ids, :, 0]
        torch.testing.assert_close(view.materials[env_ids, :, 1], sampled)
        self.assertTrue(torch.all(sampled >= 0.9))
        self.assertTrue(torch.all(sampled <= 1.3))
        self.assertNotEqual(sampled[0, 0].item(), sampled[1, 0].item())
        torch.testing.assert_close(view.materials[:, :, 2], materials[:, :, 2])
        torch.testing.assert_close(view.materials[[0, 2]], materials[[0, 2]])
        self.assertTrue(torch.equal(view.set_env_ids, env_ids))


if __name__ == "__main__":
    unittest.main()
