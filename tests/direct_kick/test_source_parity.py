"""CPU parity against pinned original source, without importing either simulator.

The oracle compiles the original Runner update AST. It does not call the
ported PPO/helpers, so changed reductions or update ordering remain observable.
Both learners use the current state/covariance model; its new observation
contract is tested independently, not asserted identical to the old model.
"""

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference")
PORT = ROOT / "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/direct_kick"
HORIZONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00]


def load_reference(name):
    """Only rewrite import namespace; preserve original executable statements."""
    namespace = "_direct_kick_parity_reference"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(REFERENCE)]
        sys.modules[namespace] = package
    qualified = f"{namespace}.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    path = REFERENCE / f"{name}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("utils."):
            dependency = node.module.removeprefix("utils.")
            load_reference(dependency)
            node.module = f"{namespace}.{dependency}"
    module = types.ModuleType(qualified)
    module.__file__ = str(path)
    sys.modules[qualified] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def load_port(name):
    namespace = "_direct_kick_parity_port"
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace, PORT / "__init__.py", submodule_search_locations=[str(PORT)]
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{namespace}.{name}")


def original_update_function():
    path = REFERENCE / "runner.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    train = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    loop = next(n for n in train.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == "it")
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "flat_obses" for t in n.targets))
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign)
               and isinstance(n.target, ast.Name) and n.target.id == "mean_post_kick_phase_loss"
               and isinstance(n.op, ast.Div))
    function = ast.parse("def update(self, obs, privileged_obs):\n    pass\n").body[0]
    function.body = copy.deepcopy(loop.body[start:end + 1]) + ast.parse("return locals()").body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    helpers = load_reference("utils")
    phase = load_reference("post_kick_phase")
    scope = {
        "torch": torch, "F": F,
        "discount_values": helpers.discount_values,
        "surrogate_loss": helpers.surrogate_loss,
        "class_balanced_phase_multipliers": phase.class_balanced_phase_multipliers,
        "weighted_phase_binary_cross_entropy": phase.weighted_phase_binary_cross_entropy,
    }
    exec(compile(module, str(path), "exec"), scope)
    return scope["update"]


def config(chunk):
    return {
        "runner": {"mini_epochs": 2, "optimization_chunk_size": chunk},
        "algorithm": {
            "learning_rate": 1e-5, "gamma": 0.995, "lam": 0.95,
            "bound_coef": 1.0, "entropy_coef": -0.01, "symmetric_coef": 10.0,
            "desired_kl": 0.01, "min_entropy": None, "max_entropy": None,
            "post_kick_phase_auxiliary": {
                "enabled": True, "coefficient": 0.1,
                "premature_weight": 3.0, "delayed_weight": 1.0,
            },
        },
    }


class SourceParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        manifest = json.loads((REFERENCE / "provenance.json").read_text())
        for name, metadata in manifest["files"].items():
            actual = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
            if actual != metadata["sha256"]:
                raise AssertionError(f"Original oracle fixture was modified: {name}")
        cls.port_model_type = load_port("model").DirectKickingActorCritic
        cls.ppo_type = load_port("learning").DirectKickPPO
        cls.original_update = staticmethod(original_update_function())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def models_and_rollout(self):
        torch.manual_seed(2049)
        original = self.port_model_type(12, 325, 20, HORIZONS)
        # Activate the bound penalty instead of comparing its all-zero path.
        with torch.no_grad():
            original.actor.network[-1].bias[0] = 1.2
            original.actor.network[-1].bias[1] = -1.2
        port = self.port_model_type(12, 325, 20, HORIZONS)
        port.load_state_dict(original.state_dict(), strict=True)
        observations = torch.randn(4, 3, 325) * 0.2
        forecast = observations[..., 47:320].reshape(4, 3, 13, 21)
        forecast[..., 0] = 0.185
        forecast[..., 1] = 0.096
        forecast[:, 0, :, 1] = 0.0  # Equal-cost samples must contribute zero symmetry weight.
        observations[..., 322] = 1.0
        privileged = torch.randn(4, 3, 20) * 0.2
        with torch.no_grad():
            actions = original.act(observations).sample()
        dones = torch.zeros(4, 3, dtype=torch.bool)
        dones[1, 0] = True
        dones[3, 0] = True
        timeouts = torch.zeros_like(dones)
        timeouts[1, 1] = True
        timeouts[2, 2] = True
        rollout = {
            "obses": observations, "privileged_obses": privileged,
            "actions": actions, "rewards": torch.randn(4, 3),
            "dones": dones, "time_outs": timeouts,
            "post_kick_phase_targets": torch.tensor(
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float32
            ).reshape(4, 3),
        }
        return original, port, rollout, torch.randn(3, 325), torch.randn(3, 20)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected), rtol=2e-5, atol=2e-6)

    def test_original_full_batch_matches_port_full_and_uneven_chunks(self):
        for chunk in (None, 5):
            with self.subTest(chunk=chunk):
                original, port, rollout, last_obs, last_privileged = self.models_and_rollout()
                original_rollout = {k: v.clone() for k, v in rollout.items()}
                port_rollout = {k: v.clone() for k, v in rollout.items()}
                oracle = types.SimpleNamespace(
                    model=original, cfg=config(None), device=torch.device("cpu"),
                    buffer=original_rollout, learning_rate=1e-5,
                    env=types.SimpleNamespace(num_obs=325, num_privileged_obs=20, num_actions=12),
                    optimizer=torch.optim.Adam(original.parameters(), lr=1e-5),
                    post_kick_phase_auxiliary_enabled=True,
                    post_kick_phase_loss_coefficient=0.1,
                    post_kick_phase_premature_weight=3.0,
                    post_kick_phase_delayed_weight=1.0,
                )
                learner = self.ppo_type(port, config(chunk), "cpu")
                expected = self.original_update(oracle, last_obs, last_privileged)
                actual = learner.update(port_rollout, last_obs, last_privileged)
                metric_names = {
                    "value_loss": "mean_value_loss", "actor_loss": "mean_actor_loss",
                    "bound_loss": "mean_bound_loss", "entropy": "mean_entropy",
                    "symmetry_loss": "mean_symmetry_loss", "symmetry_weight": "mean_symmetry_weight",
                    "phase_loss": "mean_post_kick_phase_loss", "kl": "kl_mean",
                    "phase_probability": "post_kick_phase_probability_mean",
                    "phase_ready_probability": "post_kick_phase_ready_probability_mean",
                }
                for key, source_key in metric_names.items():
                    with self.subTest(metric=key):
                        self.assert_close(actual[key], expected[source_key])
                self.assertGreater(actual["symmetry_loss"], 0)
                self.assertGreater(actual["bound_loss"], 0)
                self.assertGreater(actual["phase_loss"], 0)
                self.assert_close(actual["symmetry_weight"], 2 / 3)
                self.assertEqual(learner.learning_rate, oracle.learning_rate)
                self.assert_close(port_rollout["rewards"], original_rollout["rewards"])
                raw_advantages = expected["returns"] - expected["old_values"]
                self.assert_close(raw_advantages[rollout["time_outs"]], torch.zeros(2))
                for key, value in original.state_dict().items():
                    with self.subTest(parameter=key):
                        self.assert_close(port.state_dict()[key], value)
                original_state = oracle.optimizer.state_dict()
                port_state = learner.optimizer.state_dict()
                self.assertEqual(port_state["param_groups"], original_state["param_groups"])
                for parameter, state in original_state["state"].items():
                    self.assertEqual(port_state["state"][parameter]["step"].item(), 2)
                    for key, value in state.items():
                        with self.subTest(optimizer_parameter=parameter, optimizer_key=key):
                            self.assert_close(port_state["state"][parameter][key], value)

    def test_model_output_and_export_match_eager_clone(self):
        original, port, rollout, _, _ = self.models_and_rollout()
        obs = rollout["obses"]
        priv = rollout["privileged_obses"]
        self.assertEqual(port.checkpoint_metadata(), original.checkpoint_metadata())
        with torch.no_grad():
            self.assert_close(port.act(obs).loc, original.act(obs).loc)
            self.assert_close(port.act(obs).scale, original.act(obs).scale)
            self.assert_close(port.est_value(obs, priv), original.est_value(obs, priv))
            self.assert_close(port.post_kick_phase_logit(obs), original.post_kick_phase_logit(obs))
            flat = obs.reshape(-1, 325)
            self.assert_close(port.actor(flat), original.actor(flat))
            self.assertEqual(port.actor(flat).shape, (12, 13))
            self.assert_close(port.post_kick_phase_logit(flat), torch.full((12,), -4.0))
            scripted = torch.jit.script(port.actor)
            self.assert_close(scripted(flat), original.actor(flat))

    def test_timeout_gae_matches_hand_calculated_target(self):
        rewards = torch.tensor([[0.5], [2.0], [0.9]])
        done_with_timeout = torch.tensor([[False], [True], [False]])
        values = torch.tensor([[1.0], [2.0], [3.0]])
        for helpers in (load_reference("utils"), load_port("utils")):
            actual = helpers.discount_values(rewards, done_with_timeout, values, torch.tensor([4.0]), 0.995, 0.95)
            self.assert_close(actual, torch.tensor([[1.49], [0.0], [1.88]]))
            self.assert_close(values + actual, torch.tensor([[2.49], [2.0], [4.88]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
