"""CPU checks of the actual receiving math, observer, native boundary, and PPO."""
import ast
import contextlib
import copy
import importlib
import io
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion"
namespace = "_joga_receiving_test"
package = types.ModuleType(namespace)
package.__path__ = [str(PACKAGE)]
sys.modules[namespace] = package


def load(name):
    return importlib.import_module(f"{namespace}.{name}")


config = load("joga_receiving.config")
task_module = load("joga_receiving.task")
terrain_module = load("joga_receiving.terrain")
BallBelief = load("joga_receiving.perception").BallBelief
ReceivingRunner = load("joga_receiving.runner").ReceivingRunner
math_module = load("direct_kick.math_xyzw")


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(count=2):
    quaternion = torch.zeros(count, 2, 4)
    quaternion[..., 3] = 1.
    return {
        "ball_local": torch.tensor([[.2, 0., -.44]]).repeat(count, 1),
        "ball_position": torch.tensor([[.2, 0., .11]]).repeat(count, 1),
        "ball_velocity": torch.zeros(count, 3), "ball_ground_height": torch.full((count,), .11),
        "foot_position": torch.tensor([[[.2, .1, .02], [.2, -.1, .02]]]).repeat(count, 1, 1),
        "foot_quat": quaternion, "foot_local": torch.tensor([[[.2, .1, -.53], [.2, -.1, -.53]]]).repeat(count, 1, 1),
        "foot_ball_force": torch.zeros(count, 2), "self_contact_force": torch.zeros(count, 13),
        "upper_ball_force": torch.zeros(count, 1), "leg_ball_force": torch.zeros(count, 10),
        "visible": torch.ones(count, dtype=torch.bool), "base_velocity": torch.zeros(count, 3),
        "base_angular_velocity": torch.zeros(count, 3), "gravity": torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        "base_height": torch.full((count,), .55), "joint_position": torch.zeros(count, 12),
        "default_joint_position": torch.zeros(12), "joint_lower": torch.full((12,), -1.),
        "joint_upper": torch.ones(12), "ankle_pitch_indices": [4, 10],
        "action": torch.zeros(count, 12), "previous_action": torch.zeros(count, 12),
        "angular_momentum": torch.zeros(count, 3),
    }


def receiving(count=2):
    cfg = config.load_config()
    task = task_module.ReceivingTask(cfg, count, "cpu")
    stage = config.curriculum_at(cfg, 0)
    task.reset(torch.arange(count), stage, torch.full((count,), 3.))
    return task, state(count)


def test_exact_curriculum_iteration_boundaries():
    cfg = config.load_config()
    for iteration, speed, settle, side in (
        (0, [1.5, 2.8], .5, 0.), (4999, [1.5, 2.8], .5, 0.),
        (5000, [1.5, 2.8], .5, .35), (7999, [1.5, 2.8], .5, .35),
        (8000, [2., 3.5], 1., .35), (10000, [2., 3.5], 1., .55),
        (14000, [2.3, 4.5], 1.5, .55), (15000, [2.3, 4.5], 1.5, .70),
        (20000, [2.6, 5.5], 2., .70),
    ):
        actual = config.curriculum_at(cfg, iteration)
        assert actual["speed"] == speed
        assert actual["settle_s"] == settle
        assert actual["side_threshold"] == side
    assert cfg["physics_dt"] == .005 and cfg["decimation"] == 4
    assert cfg["model"]["actor_hidden_dims"] == [512, 256, 128]
    assert cfg["ppo"]["num_learning_epochs"] == 5
    assert cfg["ppo"]["num_mini_batches"] == 4


def test_pass_sampling_crosses_lateral_axis_at_sampled_offset():
    torch.manual_seed(71)
    cfg = config.load_config()
    stage = config.curriculum_at(cfg, 20000)
    position, velocity, speed, offset = task_module.sample_ground_pass(4096, stage, "cpu")
    time = -position[:, 0] / velocity[:, 0]
    assert (time > 0.).all()
    torch.testing.assert_close(position[:, 1] + time * velocity[:, 1], offset, atol=1.e-6, rtol=1.e-5)
    torch.testing.assert_close(velocity.norm(dim=-1), speed)
    assert (position.norm(dim=-1) >= 1.8 - 1.e-6).all()
    assert (position.norm(dim=-1) <= 5. + 1.e-6).all()
    assert (offset.abs() <= .6).all()
    assert (speed >= 2.6).all() and (speed <= 5.5).all()


def test_torque_speed_envelope_matches_equation_nine():
    target = torch.tensor([[100., -100., 3., -2.]])
    position = torch.tensor([[.2, .1, 0., 0.]])
    velocity = torch.tensor([[0., 2., 10., -12.]])
    limited, torque = task_module.torque_speed_limit(target, position, velocity, 50., 2., 20., 10.)
    expected_max = torch.tensor([[20., 16., 0., 0.]])
    torch.testing.assert_close(torque, torch.tensor([[20., -16., 0., 0.]]), atol=1.e-6, rtol=1.e-6)
    assert (torque.abs() <= expected_max + 1.e-6).all()
    torch.testing.assert_close(50. * (limited - position) - 2. * velocity, torque)


def test_angular_momentum_has_orbital_and_spin_components():
    mass = torch.ones(1, 2)
    position = torch.tensor([[[-1., 0., 0.], [1., 0., 0.]]])
    velocity = torch.tensor([[[0., -1., 0., 0., 0., 2.], [0., 1., 0., 0., 0., 3.]]])
    quat = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]])
    inertia = torch.eye(3).expand(1, 2, 3, 3)
    torch.testing.assert_close(task_module.angular_momentum(mass, position, velocity, quat, inertia),
                               torch.tensor([[0., 0., 7.]]))
    # A rotated anisotropic link uses actor-frame inertia, not principal-axis
    # orientation from the COM transform returned separately by the SDK.
    rotation = torch.tensor([[[0., math.sqrt(.5), 0., math.sqrt(.5)]]])
    spin_velocity = torch.tensor([[[0., 0., 0., 0., 0., 1.]]])
    torch.testing.assert_close(task_module.angular_momentum(
        torch.ones(1, 1), torch.zeros(1, 1, 3), spin_velocity, rotation,
        torch.diag(torch.tensor([1., 2., 3.]))[None, None]), torch.tensor([[0., 0., 1.]]))


def test_reward_weights_and_first_touch_are_not_repeated():
    task, data = receiving()
    data["foot_position"][..., 2] = .11
    data["foot_ball_force"][:, 0] = 10.
    data["ball_velocity"][:, 0] = -2.
    _, terms, success, fall = task.evaluate(data)
    dt = .02
    torch.testing.assert_close(terms["side_first_touch"], torch.full((2,), 6. * dt))
    torch.testing.assert_close(terms["velocity_kill"], torch.full((2,), 1.5 * dt))
    torch.testing.assert_close(terms["pose"], torch.full((2,), 2. * dt))
    torch.testing.assert_close(terms["upright"], torch.full((2,), dt))
    assert task.touch_count.tolist() == [1, 1]
    assert not success.any() and not fall.any()
    _, repeated, _, _ = task.evaluate(data)
    assert not repeated["side_first_touch"].any()
    assert not repeated["velocity_kill"].any()
    data["foot_ball_force"].zero_()
    task.evaluate(data)
    data["foot_ball_force"][:, 0] = 10.
    data["ball_velocity"][:, 0] = -1.
    _, second, _, _ = task.evaluate(data)
    assert task.touch_count.tolist() == [2, 2]
    torch.testing.assert_close(second["velocity_kill"], torch.full((2,), .5 * 1.5 * dt))


def test_side_contact_uses_full_foot_orientation():
    task, data = receiving(1)
    task.side_threshold[:] = .9
    data["foot_position"][0, 0] = data["ball_position"][0] - torch.tensor([0., 0., .1])
    data["foot_quat"][0, 0] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    data["foot_ball_force"][0, 0] = 10.
    _, terms, _, _ = task.evaluate(data)
    assert task.side_seen.item()
    torch.testing.assert_close(terms["side_first_touch"], torch.tensor([6. * .02]))


def test_settle_requires_continuous_valid_trap_and_fall_has_priority():
    task, data = receiving()
    task.side_seen[:] = True
    for _ in range(24):
        _, _, success, _ = task.evaluate(data)
        assert not success.any()
    data["ball_velocity"][0, 0] = .6
    task.evaluate(data)
    assert task.settle_time[0] == 0.
    assert task.settle_time[1] >= .5 - 1.e-6
    data["ball_velocity"].zero_()
    data["gravity"][1, 2] = 0.
    _, _, success, fall = task.evaluate(data)
    assert not success[1] and fall[1]


def test_force_penalty_thresholds_are_strict_and_reset_is_per_environment():
    task, data = receiving()
    data["self_contact_force"][0, :2] = torch.tensor([10., 11.])
    data["leg_ball_force"][0, :2] = torch.tensor([5., 6.])
    _, terms, _, _ = task.evaluate(data)
    torch.testing.assert_close(terms["self_collision"], torch.tensor([-.02, 0.]))
    torch.testing.assert_close(terms["leg_ball"], torch.tensor([-.1, 0.]))
    task.touch_count[:] = 4
    task.side_seen[:] = True
    task.reset(torch.tensor([0]), config.curriculum_at(task.cfg, 20000), torch.tensor([4.]))
    assert task.touch_count.tolist() == [0, 4]
    assert task.side_seen.tolist() == [False, True]
    assert task.target_settle.tolist() == [2., .5]


def test_terrain_height_matches_mesh_triangles_and_flat_exterior():
    heights = torch.tensor([[0., 1.], [2., 4.]])
    points = torch.tensor([[-.25, -.25], [.25, .25], [-.5, -.5], [-1., 0.]])
    torch.testing.assert_close(terrain_module.tile_height(points, heights, 1.), torch.tensor([.75, 2.75, 0., 0.]))


def test_real_observer_acquires_without_truth_velocity_and_partial_reset():
    cfg = config.load_config()
    torch.manual_seed(80)
    belief = BallBelief(cfg["perception"], 2, .02, "cpu")
    roots = torch.zeros(2, 2, 13)
    roots[..., 6] = 1.
    roots[:, 1, 0] = 2.
    belief.reset(torch.arange(2), roots)
    for step in range(40):
        feature, angular = belief.observe(roots, step, advance=step > 0)
    assert feature.shape == (2, 276) and angular.shape == (2, 3)
    assert torch.isfinite(feature).all()
    assert belief.belief_valid.all()
    previous = belief.ball_filter.state[1].clone()
    belief.reset(torch.tensor([0]), roots)
    assert not belief.belief_valid[0]
    torch.testing.assert_close(belief.ball_filter.state[1], previous)


def native_class():
    path = PACKAGE / "tasks/direct/joga_receiving/env.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in (
        "_get_observations", "_frame", "_truth_belief", "_get_dones", "_reset_idx", "drain_episode_metrics")]
    rebuilt = ast.ClassDef(name="NativeBoundary", bases=[ast.Name(id="object", ctx=ast.Load())],
                           keywords=[], body=methods, decorator_list=[])
    scope = {"torch": torch, "FRAME_SIZE": config.FRAME_SIZE,
             "HISTORY_LENGTH": config.HISTORY_LENGTH, "apply_randomization": load("direct_kick.utils").apply_randomization,
             "get_euler_xyz": math_module.get_euler_xyz, "quat_rotate_inverse": math_module.quat_rotate_inverse,
             "angular_momentum": task_module.angular_momentum,
             "build_horizon_tokens": load("direct_kick.direct_kicking_observation").build_horizon_tokens,
             "forecast_constant_body_twist": load("direct_kick.ego_motion").forecast_constant_body_twist,
             "relative_state_and_covariance": load("direct_kick.ego_motion").relative_state_and_covariance}
    scope.update({"math": math, "sample_ground_pass": task_module.sample_ground_pass,
                  "curriculum_at": config.curriculum_at,
                  "k1_foot_contact_points_root": load("direct_kick.k1_foot_kinematics").k1_foot_contact_points_root,
                  "quat_rotate": math_module.quat_rotate,
                  "DirectRLEnv": types.SimpleNamespace(
                      _reset_idx=lambda env, ids: env.episode_length_buf.__setitem__(ids, 0))})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[rebuilt], type_ignores=[])), str(path), "exec"), scope)
    return scope["NativeBoundary"]


def native_fixture():
    env = native_class()()
    env.task_cfg = config.load_config()
    env.num_envs, env.device, env.common_step_counter = 2, "cpu", 0
    env.root_states = torch.zeros(2, 2, 13)
    env.root_states[..., 6] = 1.
    env.root_states[:, 0, 2] = .55
    env.root_states[:, 1, :3] = torch.tensor([2., 0., .11])
    env.q = torch.zeros(2, 12)
    env.dq = torch.zeros_like(env.q)
    env.default_pose = torch.zeros(12)
    env.encoder_bias = torch.zeros_like(env.q)
    env.actions = torch.zeros_like(env.q)
    env.previous_actions = torch.zeros_like(env.q)
    env.gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    env.base_velocity = torch.zeros(2, 3)
    env.base_angular_velocity = torch.zeros(2, 3)
    env.foot_position = torch.tensor([[[0., .1, .02], [0., -.1, .02]]]).repeat(2, 1, 1)
    env.foot_quat = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]]).repeat(2, 1, 1)
    env.foot_ground_force = torch.zeros(2, 2, 3)
    env.foot_ground_contact = torch.zeros(2, 2, dtype=torch.bool)
    env.air_time = torch.zeros(2, 2)
    env._policy_history = torch.zeros(2, 3, config.FRAME_SIZE)
    env._critic_history = torch.zeros_like(env._policy_history)
    env._history_reset = torch.ones(2, dtype=torch.bool)
    env._obs_step = -1
    env._belief_step = -1
    env._refresh_state = lambda: None
    env._ground_height = lambda points, ids=None: torch.zeros(points.shape[:-1])
    env.belief = BallBelief(env.task_cfg["perception"], 2, .02, "cpu")
    env.belief.reset(torch.arange(2), env.root_states)
    return env


def test_native_history_dimensions_cache_and_truth_only_in_critic():
    env = native_fixture()
    torch.manual_seed(15)
    obs = env._get_observations()
    assert obs["policy"].shape == (2, 969) and obs["critic"].shape == (2, 990)
    frames = obs["policy"].reshape(2, 3, 323)
    torch.testing.assert_close(frames[:, 0], frames[:, 2])
    assert env._get_observations() is obs
    first_policy = obs["policy"].clone()
    first_critic = obs["critic"].clone()
    env.root_states[:, 1, 0] = 4.
    env._history_reset[:] = True
    torch.manual_seed(15)
    changed = env._get_observations()
    torch.testing.assert_close(changed["policy"], first_policy)
    assert not torch.equal(changed["critic"], first_critic)


def test_native_contact_pair_forces_do_not_cancel_and_timeout_is_exclusive():
    env = native_fixture()
    env.common_step_counter = 1
    env.step_dt = .02
    env._body_names = ["Trunk", "left_foot_link", "right_foot_link", "shank"]
    env.foot_filter_ids = env.foot_ids = [1, 2]
    env.leg_filter_ids = [3]
    env.contact_ids = [0, 1, 2, 3]
    env.ankle_pitch_indices = [4, 10]
    env.lower, env.upper = -torch.ones(12), torch.ones(12)
    env.episode_length_buf = torch.tensor([400, 400])
    env.max_episode_length = 400
    env.episode_return = torch.zeros(2)
    env.extras = {}
    env.receiving = task_module.ReceivingTask(env.task_cfg, 2, "cpu")
    env.receiving.reset(torch.arange(2), config.curriculum_at(env.task_cfg, 0), torch.full((2,), 3.))
    env.gravity[0, 2] = 0.  # A fall on the final step must not also be a timeout.
    normal = torch.zeros(2, 1, 4, 3)
    normal[:, 0, 1, 1] = 10.
    env.ball_contacts = types.SimpleNamespace(data=types.SimpleNamespace(
        force_matrix_w=normal, force_matrix_w_history=normal[:, None].repeat(1, 4, 1, 1, 1)))
    env.self_contacts = {}
    for name in env._body_names:
        forces = torch.zeros(2, 4, 1, 3, 3)
        if name == "Trunk":
            forces[:, :, 0, 0, 0] = 11.
            forces[:, :, 0, 1, 0] = -11.
        env.self_contacts[name] = types.SimpleNamespace(data=types.SimpleNamespace(
            force_matrix_w_history=forces, force_matrix_w=forces[:, 0]))
    net = torch.zeros(2, 4, 3)
    net[:, 1:3, 2] = 100.
    net[:, 1, 1] = -10.
    env.contacts = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=net))
    env.mass = torch.ones(2, 4)
    env.inertia = torch.eye(3).expand(2, 4, 3, 3)
    com_quat = torch.zeros(2, 4, 4)
    com_quat[..., 0] = 1.
    env.robot = types.SimpleNamespace(data=types.SimpleNamespace(
        body_com_pos_w=torch.zeros(2, 4, 3), body_com_vel_w=torch.zeros(2, 4, 6), body_link_quat_w=com_quat))
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, True]
    assert env._timeout.tolist() == [False, True]
    torch.testing.assert_close(env.extras["rew_terms"]["self_collision"], torch.full((2,), -.02))
    torch.testing.assert_close(env.foot_ground_force[..., 1], torch.zeros(2, 2))
    torch.testing.assert_close(env.foot_ground_force[..., 2], torch.full((2, 2), 100.))
    # Observation construction after reward must reuse that step's perception.
    env._history_reset.zero_()
    stamps = env.belief.ball_filter.stamp_ns.clone()
    env._get_observations()
    torch.testing.assert_close(env.belief.ball_filter.stamp_ns, stamps)


def test_native_partial_reset_records_completed_only_and_latches_curriculum():
    env = native_fixture()
    env.iteration = 20000
    env.step_dt = .02
    env._pending_finish = torch.tensor([True, False])
    env._success = torch.tensor([True, False])
    env._fall = env._timeout = torch.zeros(2, dtype=torch.bool)
    env._metrics = torch.zeros(7)
    env.episode_return = torch.tensor([2., 3.])
    env.episode_length_buf = torch.tensor([20, 35])
    env.next_push = torch.zeros(2, dtype=torch.long)
    env.spawn_speed = torch.zeros(2)
    env.spawn_offset = torch.zeros(2)
    env.foot_contact_points = torch.tensor(env.task_cfg["robot"]["asset"]["feet_edge_pos"])
    env.receiving = task_module.ReceivingTask(env.task_cfg, 2, "cpu")
    env.receiving.reset(torch.arange(2), config.curriculum_at(env.task_cfg, 0), torch.ones(2))
    env.joint_ids = list(range(12))
    written = []
    env.robot = types.SimpleNamespace(write_joint_state_to_sim=lambda q, dq, **kwargs: written.append(q.clone()))
    env._write_roots = lambda ids: None
    env._randomize_materials = lambda ids: None
    previous_other = env.root_states[1].clone()
    env._reset_idx(torch.tensor([0]))
    assert env._metrics[0] == 1. and env._metrics[1] == 1.
    assert env.receiving.target_settle.tolist() == [2., .5]
    torch.testing.assert_close(env.root_states[1], previous_other)
    assert env.episode_length_buf.tolist() == [0, 35]
    assert (env.root_states[0, 1, 7:9].norm() >= 2.6).item()
    assert not env.belief.belief_valid[0]
    torch.testing.assert_close(written[0], env.default_pose[None])
    # A manual reset of an unfinished episode does not count it as completed.
    env._reset_idx(torch.tensor([1]))
    assert env._metrics[0] == 1.


class ReusedEnv:
    def __init__(self):
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.obs = {"policy": torch.zeros(2, config.POLICY_SIZE), "critic": torch.zeros(2, config.CRITIC_SIZE)}
        self.stage = 0

    def reset(self):
        self.tick = 0
        for value in self.obs.values():
            value.zero_()
        return self.obs, {}

    def set_curriculum_iteration(self, iteration):
        self.stage = iteration

    def step(self, actions):
        assert actions.shape == (2, 12)
        self.tick += 1
        for value in self.obs.values():
            value.fill_(self.tick * .1)
        terminated = torch.tensor([False, self.tick % 2 == 0])
        truncated = torch.tensor([self.tick % 3 == 0, False])
        return self.obs, torch.tensor([.1, .2]), terminated, truncated, {}

    def drain_episode_metrics(self):
        return {"episodes": 0, "mean_return": "", "success_rate": ""}


def small_config():
    cfg = config.load_config()
    cfg["num_envs"] = 2
    cfg["runner"].update(steps_per_env=4, save_interval=1)
    cfg["ppo"].update(num_learning_epochs=2, num_mini_batches=2)
    return cfg


def test_ppo_learning_resume_optimizer_and_export(tmp_path):
    torch.manual_seed(60)
    with contextlib.redirect_stdout(io.StringIO()):
        native = ReusedEnv()
        wrapped = types.SimpleNamespace(unwrapped=native, reset=native.reset, step=native.step)
        runner = ReceivingRunner(wrapped, small_config(), tmp_path / "first")
        before = copy.deepcopy(runner.model.state_dict())
        path = runner.train(1)
        restored = ReceivingRunner(ReusedEnv(), small_config(), tmp_path / "restored")
        restored.load(path)
    assert restored.iteration == 1 and restored.total_steps == 8
    assert restored.env.stage == 1
    for prefix in ("actor.forecast.", "critic.forecast."):
        assert any(not torch.equal(value, before[name]) for name, value in runner.model.state_dict().items()
                   if name.startswith(prefix))
    for name, value in runner.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0)
    original_optimizer = runner.ppo.optimizer.state_dict()
    restored_optimizer = restored.ppo.optimizer.state_dict()
    assert original_optimizer["param_groups"] == restored_optimizer["param_groups"]
    for parameter, values in original_optimizer["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(restored_optimizer["state"][parameter][name], value, rtol=0, atol=0)
    assert restored.ppo.learning_rate == runner.ppo.learning_rate
    stored = runner.ppo.storage.observations["policy"][:, :, 0]
    torch.testing.assert_close(stored[:, 0], torch.tensor([0., .1, .2, .3]))
    export = tmp_path / "policy.pt"
    runner.export(export)
    scripted = torch.jit.load(str(export))
    runner.model.eval()
    for size in (1, 3):
        obs = torch.randn(size, config.POLICY_SIZE)
        torch.testing.assert_close(scripted(obs), runner.model.actor(runner.model.actor_obs_normalizer(obs)))
    with contextlib.redirect_stdout(io.StringIO()):
        incompatible = ReceivingRunner(ReusedEnv(), small_config(), tmp_path / "incompatible")
    changed = torch.load(path, weights_only=False)
    changed["metadata"]["contract"] = "direct_kick"
    bad = tmp_path / "bad.pt"
    torch.save(changed, bad)
    with pytest.raises(ValueError, match="contract differs"):
        incompatible.load(bad)
