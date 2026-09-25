"""CPU checks of native boundary methods without starting either simulator.

The wrench lifetime check executes the installed Lab write method with tensor
doubles. These tests do not establish the legacy Gym body's pose convention.
"""

import ast
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from test_source_parity import ROOT


ENV_PATH = (
    ROOT / "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion"
    / "tasks/direct/direct_kick/env.py"
)


def extract_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    scope = {"torch": torch}
    exec(compile(module, str(path), "exec"), scope)
    return scope[method_name]


class WrenchBuffer:
    """Tensor storage double; lifetime is controlled by the real Lab method."""

    def __init__(self, shape):
        self.composed_force = torch.zeros(shape)
        self.composed_torque = torch.zeros(shape)
        self.active = False
        self.calls = []

    @property
    def composed_force_as_torch(self):
        return self.composed_force

    @property
    def composed_torque_as_torch(self):
        return self.composed_torque

    def set_forces_and_torques(self, *, forces, torques, is_global, positions=None):
        if positions is not None or is_global:
            raise AssertionError("Expected the explicitly transported local wrench")
        self.composed_force.copy_(forces)
        self.composed_torque.copy_(torques)
        self.active = True
        self.calls.append((forces.clone(), torques.clone()))

    def add_forces_and_torques(self, *, forces, torques, body_ids, env_ids):
        self.composed_force.add_(forces)
        self.composed_torque.add_(torques)

    def reset(self):
        self.composed_force.zero_()
        self.composed_torque.zero_()
        self.active = False


class BackendBoundaryTest(unittest.TestCase):
    write_forces = staticmethod(extract_method(ENV_PATH, "DirectKickEnv", "_write_external_forces"))
    write_roots = staticmethod(extract_method(ENV_PATH, "DirectKickEnv", "_write_root_states"))

    def force_fixture(self):
        # Last body is the ball and must not be passed to the robot view.
        forces = torch.tensor([[[0., 3., 0.], [0., 0., 5.], [91., 92., 93.]],
                               [[7., 0., 0.], [0., 11., 0.], [94., 95., 96.]]])
        torques = torch.tensor([[[13., 17., 19.], [23., 29., 31.], [97., 98., 99.]],
                                [[37., 41., 43.], [47., 53., 59.], [100., 101., 102.]]])
        com = torch.tensor([[[2., 0., 0.], [0., 4., 0.]],
                            [[0., 0., 6.], [8., 0., 0.]]])
        buffer = WrenchBuffer((2, 2, 3))
        env = SimpleNamespace(
            num_bodies=2, pushing_forces=forces, pushing_torques=torques,
            robot_body_local_com=com,
            robot=SimpleNamespace(instantaneous_wrench_composer=buffer),
        )
        return env, buffer

    def test_wrench_preserves_force_and_moment_about_com_without_mutating_source(self):
        env, buffer = self.force_fixture()
        source_forces, source_torques = env.pushing_forces.clone(), env.pushing_torques.clone()
        self.write_forces(env)
        force, torque_at_origin = buffer.calls[0]
        torch.testing.assert_close(force, source_forces[:, :2])
        # Known lever arms produce z=6, x=20, y=42, z=88 Nm respectively.
        expected_offset = torch.tensor([[[0., 0., 6.], [20., 0., 0.]],
                                        [[0., 42., 0.], [0., 0., 88.]]])
        torch.testing.assert_close(torque_at_origin, source_torques[:, :2] + expected_offset)
        moment_about_com = torque_at_origin + torch.cross(-env.robot_body_local_com, force, dim=-1)
        torch.testing.assert_close(moment_about_com, source_torques[:, :2])
        torch.testing.assert_close(env.pushing_forces, source_forces)
        torch.testing.assert_close(env.pushing_torques, source_torques)

    def test_installed_lab_consumes_wrench_once_during_ten_physics_writes(self):
        spec = find_spec("isaaclab")
        if spec is None:
            self.skipTest("IsaacLab must be installed to check its wrench consumption")
        package = Path(spec.origin).parent
        candidates = [package, package / "source/isaaclab/isaaclab"]
        path = next((p / "assets/articulation/articulation.py" for p in candidates
                     if (p / "assets/articulation/articulation.py").is_file()), None)
        self.assertIsNotNone(path, "Installed IsaacLab articulation source is unavailable")
        write_to_sim = extract_method(path, "Articulation", "write_data_to_sim")
        env, buffer = self.force_fixture()
        self.write_forces(env)
        emitted = []

        def apply_wrench(**kwargs):
            emitted.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})

        robot = SimpleNamespace(
            _instantaneous_wrench_composer=buffer,
            _permanent_wrench_composer=WrenchBuffer((2, 2, 3)),
            _ALL_BODY_INDICES_WP=None, _ALL_INDICES_WP=None,
            _ALL_INDICES=torch.arange(2), _joint_effort_target_sim=torch.zeros(2, 12),
            _has_implicit_actuators=False, _apply_actuator_model=lambda: None,
            root_physx_view=SimpleNamespace(
                apply_forces_and_torques_at_position=apply_wrench,
                set_dof_actuation_forces=lambda *args: None,
            ),
        )
        for _ in range(10):
            write_to_sim(robot)
        self.assertEqual(len(emitted), 1)
        self.assertFalse(buffer.active)
        self.assertIsNone(emitted[0]["position_data"])
        self.assertFalse(emitted[0]["is_global"])
        torch.testing.assert_close(emitted[0]["force_data"], buffer.calls[0][0].reshape(-1, 3))
        torch.testing.assert_close(emitted[0]["torque_data"], buffer.calls[0][1].reshape(-1, 3))

    def test_root_write_adds_scene_origin_and_reorders_quaternion_without_mutating_task(self):
        state = torch.arange(52, dtype=torch.float32).reshape(2, 2, 13)
        state[:, :, 3:7] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        original = state.clone()
        origins = torch.tensor([[10., 20., 30.], [-40., -50., -60.]])
        calls = {"robot": [], "ball": []}

        def asset(name):
            return SimpleNamespace(write_root_state_to_sim=lambda value, env_ids:
                                   calls[name].append((value.clone(), env_ids.clone())))

        env = SimpleNamespace(root_states=state, scene=SimpleNamespace(env_origins=origins),
                              robot=asset("robot"), ball=asset("ball"))
        ids = torch.tensor([1])
        self.write_roots(env, ids, robot=True, ball=True)
        for index, name in enumerate(("robot", "ball")):
            written, written_ids = calls[name][0]
            torch.testing.assert_close(written_ids, ids)
            torch.testing.assert_close(written[:, :3], original[ids, index, :3] + origins[ids])
            torch.testing.assert_close(written[:, 3:7], torch.tensor([[0.4, 0.1, 0.2, 0.3]]))
            torch.testing.assert_close(written[:, 7:], original[ids, index, 7:])
        torch.testing.assert_close(state, original)


if __name__ == "__main__":
    unittest.main()
