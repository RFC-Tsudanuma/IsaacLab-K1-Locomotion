"""CPU task-contract checks with tensor-only physics writes and fixed belief.

The perception pipeline is deliberately excluded: these tests exercise the
observation assembly, reward/outcome, reset, and physics-substep delay logic.
"""

import math
import unittest
import xml.etree.ElementTree as ET

import torch

from test_source_parity import PORT, load_port


DirectKickingLogic = load_port("direct_kicking_logic").DirectKickingLogic


class TensorBackend(DirectKickingLogic):
    def __init__(self, count=4):
        torch.manual_seed(101)
        self.device = torch.device("cpu")
        self.num_envs = count
        self.task_cfg = load_port("config").load_config()
        self.direct_cfg = self.task_cfg["direct_kicking"]
        self.task_cfg["noise"] = {}  # Deterministic non-filter observation contract.
        robot = ET.parse(PORT / "assets/K1/K1_locomotion.urdf").getroot()
        joints = [j for j in robot.findall("joint") if j.attrib["type"] != "fixed"]
        self.dof_names = [j.attrib["name"] for j in joints]
        self.num_dofs = len(joints)
        links = robot.findall("link")
        names = [link.attrib["name"] for link in links]
        self.num_bodies = len(links)
        self.base_indice = names.index(self.task_cfg["asset"]["base_name"])
        self.feet_indices = torch.tensor([names.index(name) for name in self.task_cfg["asset"]["foot_names"]])
        self.termination_contact_indices = torch.empty(0, dtype=torch.long)
        self.penalized_contact_indices = torch.empty(0, dtype=torch.long)
        self.dof_pos_limits = torch.tensor([
            [float(j.find("limit").attrib["lower"]), float(j.find("limit").attrib["upper"])] for j in joints
        ])
        self.dof_vel_limits = torch.tensor([float(j.find("limit").attrib["velocity"]) for j in joints])
        self.torque_limits = torch.tensor([float(j.find("limit").attrib["effort"]) for j in joints])
        masses, centers = [], []
        for link in links:
            inertial = link.find("inertial")
            masses.append(float(inertial.find("mass").attrib["value"]))
            centers.append([float(v) for v in inertial.find("origin").attrib["xyz"].split()])
        self.robot_body_masses = torch.tensor(masses).repeat(count, 1)
        self.robot_body_local_com = torch.tensor(centers).repeat(count, 1, 1)
        # Source privileged first four values are the sampled uniform draws.
        self.base_mass_scaled = torch.tensor([[0.11, 0.22, 0.33, 0.44]]).repeat(count, 1)
        self.up_axis_idx = 2
        self.env_origins = torch.zeros(count, 3)
        self.env_origins[:, 0] = torch.arange(count) * 5.0
        init = self.task_cfg["init_state"]
        self.base_init_state = torch.tensor(init["pos"] + init["rot"] + init["lin_vel"] + init["ang_vel"])
        self.ball_radius = float(self.task_cfg["ball"]["radius"])
        self.ball_physics_randomization_enabled = True
        self.ball_restitution_range = tuple(self.direct_cfg["physics_randomization"]["ball_restitution_range"])
        self.sampled_ball_restitution = torch.zeros(count)
        self.env_resets = self.env_successes = self.env_falling = 0
        self.ball_velocities = []
        self.writes = []
        self._configure_ball_motion(self.task_cfg)
        self._configure_action_delay(self.task_cfg)
        self._configure_external_disturbances(self.task_cfg)
        self._init_buffers()
        self._init_stage_curriculum()
        self._prepare_reward_function()
        self.dof_pos.copy_(self.default_dof_pos)
        self.base_pos[:, 2] = 0.545
        self.body_states[:, self.feet_indices[0], :3] = torch.tensor([0.15, 0.096, 0.02])
        self.body_states[:, self.feet_indices[1], :3] = torch.tensor([0.15, -0.096, 0.02])
        self.ball_pos[:] = torch.tensor([0.3, 0.096, 0.075])
        self.body_states[:, -1] = self.root_states[:, 1]
        self._refresh_feet_state()
        self.last_feet_pos.copy_(self.feet_pos)
        self.direct_previous_feet_ankle_pos_world.copy_(self._feet_ankle_positions_world())
        self.direct_previous_base_pos_world.copy_(self.base_pos)
        self.direct_previous_base_quat_world.copy_(self.base_quat)
        self.direct_previous_ball_pos_world.copy_(self.ball_pos)
        self._update_kick_target_positions(torch.arange(count))
        self.kick_target_pos_world[:, 0] = 5.0
        self.kick_target_pos_world[:, 1] = self.ball_pos[:, 1]
        self.reset_buf.zero_()
        self.episode_length_buf.fill_(10)

    def _init_direct_perception_buffers(self):
        self._perception_step_requested = False
        self.perception_just_reset = torch.zeros(self.num_envs, dtype=torch.bool)
        self.observed_base_yaw = torch.zeros(self.num_envs)
        self.ego_yaw_rate_bias = torch.zeros(self.num_envs)
        self.ego_yaw_rate_drift = torch.zeros(self.num_envs)
        self.ego_velocity_bias = torch.zeros(self.num_envs, 2)
        self.ego_velocity_drift = torch.zeros(self.num_envs, 2)
        self.fixed_belief = torch.arange(276, dtype=torch.float32).repeat(self.num_envs, 1) / 100.0
        self.kick_detection_block_until_step = torch.zeros(self.num_envs, dtype=torch.long)
        self.direct_previous_ball_pos_world = self.ball_pos.clone()
        self.direct_previous_ball_lin_vel_world = torch.zeros(self.num_envs, 3)
        self.direct_previous_base_pos_world = self.base_pos.clone()
        self.direct_previous_base_quat_world = self.base_quat.clone()

    def _belief_observation(self, observed_linear_velocity, observed_angular_velocity):
        return self.fixed_belief

    def _terrain_heights(self, points):
        return torch.zeros(points.shape[:-1])

    def _write_root_states(self, env_ids, robot=False, ball=False):
        self.writes.append(("root", env_ids.clone(), robot, ball))
        if ball:
            self.body_states[env_ids, -1] = self.root_states[env_ids, 1]

    def _write_dof_state(self, env_ids):
        self.writes.append(("dof", env_ids.clone()))

    def _write_external_forces(self):
        self.writes.append(("forces",))

    def _write_ball_restitution(self, env_ids, restitution):
        self.writes.append(("restitution", env_ids.clone(), restitution.clone()))


class TaskLogicTest(unittest.TestCase):
    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), rtol=1e-5, atol=1e-6)

    def test_observation_layout_and_privileged_raw_uniform_draws(self):
        env = TensorBackend()
        self.assertEqual(env.num_actions, 12)
        env.projected_gravity[:] = torch.tensor([0.1, 0.2, -0.9])
        env.base_ang_vel[:] = torch.tensor([1.0, 2.0, 3.0])
        env.commands[:] = torch.tensor([0.4, 0.5, 0.6])
        env.gait_frequency[1] = 1.0
        env.gait_process[1] = 0.25
        offsets = torch.arange(12) / 10.0
        env.dof_pos[:] = env.default_dof_pos + offsets
        env.dof_vel[:] = torch.arange(12)
        env.actions[:] = torch.linspace(-1, 1, 12)
        env.base_lin_vel[:] = torch.tensor([4.0, 5.0, 6.0])
        env.pushing_forces[:, 0] = torch.tensor([7.0, 8.0, 9.0])
        env.pushing_torques[:, 0] = torch.tensor([10.0, 11.0, 12.0])
        env.ball_lin_vel[:] = torch.tensor([13.0, 14.0, 15.0])
        env.feet_pos[:, 0, :2] = torch.tensor([21.0, 22.0])
        env.feet_pos[:, 1, :2] = torch.tensor([23.0, 24.0])
        env._compute_observations()
        self.assertEqual(tuple(env.obs_buf.shape), (4, 325))
        self.assertEqual(tuple(env.privileged_obs_buf.shape), (4, 20))
        self.assert_close(env.obs_buf[0, :9], [0.1, 0.2, -0.9, 1, 2, 3, 0.4, 0.5, 0.6])
        self.assert_close(env.obs_buf[0, 9:11], [0, 0])
        self.assert_close(env.obs_buf[1, 9:11], [0, 1])
        self.assert_close(env.obs_buf[0, 11:23], offsets)
        self.assert_close(env.obs_buf[0, 23:35], torch.arange(12) * 0.1)
        self.assert_close(env.obs_buf[:, 35:47], env.actions)
        self.assert_close(env.obs_buf[:, 47:323], env.fixed_belief)
        self.assert_close(env.obs_buf[0, 323:325], [1, 0])
        self.assert_close(env.privileged_obs_buf[0], [
            0.11, 0.22, 0.33, 0.44, 4, 5, 6, 0.545,
            0.7, 0.8, 0.9, 5, 5.5, 6, 13, 14, 21, 22, 23, 24,
        ])
        # Exported teacher is a snapshot, not an alias into mutable episode state.
        env.post_kick_phase_target_buf.fill_(True)
        self.assertFalse(env.extras["post_kick_phase_target"].any())

    def test_all_34_rewards_and_unscaled_first_kick_direction(self):
        env = TensorBackend()
        self.assertEqual(len(env.reward_scales), 34)
        for name, scale in env.reward_scales.items():
            self.assertAlmostEqual(scale, env.task_cfg["rewards"]["scales"][name] * 0.02)
        env.last_feet_pos[:, 0, 0] -= 0.02
        env.root_states[:, 1, 7] = 1.0
        env.body_states[:, -1, 7] = 1.0
        env.reset_buf[1] = True
        env.episode_length_buf[3] = 0
        env.kick_detection_block_until_step.fill_(5)
        env._compute_reward()
        self.assertTrue(torch.isfinite(env.rew_buf).all())
        expected_direction = 10 / (1 + math.exp(-5))
        self.assert_close(env.extras["rew_terms"]["kick_direction"], [expected_direction, 0, expected_direction, 0])
        self.assert_close(env.extras["rew_terms"]["survival"], [0.006] * 4)
        summed = torch.stack(list(env.extras["rew_terms"].values())).sum(dim=0)
        self.assert_close(env.rew_buf, summed)
        self.assert_close(env.first_valid_kick_step, [10, 10, 10, -1])
        env._compute_reward()
        self.assert_close(env.extras["rew_terms"]["kick_direction"], [0] * 4)

    def test_pre_kick_motion_does_not_terminate_or_timeout_at_command_resample(self):
        env = TensorBackend()
        env.min_ball_vel_buf.fill_(1000)
        env.time_since_ball_is_still_buf.fill_(100)
        env.time_since_ball_is_moving_buf.fill_(100)
        env.cmd_resample_time.copy_(env.episode_length_buf)
        env._check_termination()
        self.assertFalse(env.reset_buf.any())
        self.assertFalse(env.time_out_buf.any())
        self.assert_close(env.min_ball_vel_buf, [0] * 4)
        self.assert_close(env.time_since_ball_is_still_buf, [0] * 4)
        self.assert_close(env.time_since_ball_is_moving_buf, [0] * 4)

    def test_episode_limit_post_kick_and_fall_priority(self):
        env = TensorBackend()
        env.valid_kick_buf[:] = torch.tensor([False, True, True, False])
        env.episode_length_buf[:] = torch.tensor([850, 110, 110, 851])
        env.first_valid_kick_step[:] = torch.tensor([-1, 10, 10, -1])
        env.base_pos[2, 2] = 0.40
        env._check_termination()
        self.assertEqual(env.reset_buf.tolist(), [False, True, True, True])
        self.assertEqual(env.time_out_buf.tolist(), [False, False, False, True])
        self.assertEqual(env.post_kick_terminal_buf.tolist(), [False, True, False, False])
        self.assertEqual(env.fall_buf.tolist(), [False, False, True, False])

    def test_phase_waits_for_selected_foot_landing(self):
        env = TensorBackend()
        env.kicking_foot_mask_buf[:, 0] = True
        env.previous_feet_contact_buf.fill_(True)
        env.feet_contact.fill_(True)
        env._update_post_kick_phase_target()
        self.assertFalse(env.post_kick_phase_target_buf.any())
        env.feet_contact[:, 0] = False
        env._update_post_kick_phase_target()
        self.assertFalse(env.post_kick_phase_target_buf.any())
        env.previous_feet_contact_buf.copy_(env.feet_contact)
        env.feet_contact[:, 0] = True
        env._update_post_kick_phase_target()
        self.assertTrue(env.post_kick_phase_target_buf.all())
        env.feet_contact[:, 0] = False
        env._update_post_kick_phase_target()
        self.assertTrue(env.post_kick_phase_target_buf.all())

    def test_ball_resets_only_approach_robot_for_different_positions_and_yaws(self):
        env = TensorBackend(count=128)
        ids = torch.arange(env.num_envs)
        yaw = torch.linspace(-math.pi, math.pi, env.num_envs)
        env.root_states[:, 0, 0] = torch.linspace(-10., 10., env.num_envs)
        env.root_states[:, 0, 1] = torch.linspace(5., -5., env.num_envs)
        env.root_states[:, 0, 3:7] = torch.stack(
            (torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw/2), torch.cos(yaw/2)), dim=-1
        )
        env.root_states[:, 0, 7:13].zero_()
        for _ in range(2):
            env._reset_ball_at_robot_front(ids)
            offset = env.root_states[:, 1, :2] - env.root_states[:, 0, :2]
            velocity = env.root_states[:, 1, 7:9]
            # Distance initially decreases, regardless of world position/yaw.
            self.assertTrue(((offset * velocity).sum(dim=-1) < 0).all())
            distance = offset.norm(dim=-1)
            self.assertTrue(((distance >= 1.5) & (distance <= 3.)).all())
            self.assertTrue((velocity.norm(dim=-1) <= 1.).all())

    def test_physics_substep_delay_and_selected_environment_reset(self):
        env = TensorBackend()
        self.assertEqual(env.action_delay_step_range, (1, 25))
        self.assertEqual(env.action_target_history.shape, (4, 4, 12))
        env.action_target_history.zero_()
        env.delay_steps[:] = torch.tensor([1, 10, 11, 25])
        targets = torch.ones(4, 12)
        env._update_delayed_dof_targets(targets, 0)
        self.assert_close(env.last_dof_targets[:, 0], [0, 0, 0, 0])
        env._update_delayed_dof_targets(targets, 1)
        self.assert_close(env.last_dof_targets[:, 0], [1, 0, 0, 0])
        targets.fill_(2)
        env._update_delayed_dof_targets(targets, 0)
        self.assert_close(env.last_dof_targets[:, 0], [1, 1, 0, 0])
        env._update_delayed_dof_targets(targets, 1)
        self.assert_close(env.last_dof_targets[:, 0], [2, 1, 1, 0])
        env.action_target_history[1].fill_(99)
        env.post_kick_phase_target_buf.fill_(True)
        env.common_step_counter = 1000
        saved = env.action_target_history[0].clone()
        env._reset_idx(torch.tensor([1, 3]))
        self.assert_close(env.action_target_history[0], saved)
        self.assert_close(env.action_target_history[1], env.dof_pos[1].repeat(4, 1))
        self.assertEqual(env.post_kick_phase_target_buf.tolist(), [True, False, True, False])
        self.assertEqual(env.episode_length_buf.tolist(), [10, 0, 10, 0])
        self.assertTrue(((env.delay_steps[[1, 3]] >= 1) & (env.delay_steps[[1, 3]] <= 25)).all())
        self.assertTrue((env.next_force_push_step[[1, 3]] >= 1150).all())
        self.assertTrue((env.next_velocity_push_step[[1, 3]] >= 1200).all())
        self.assertTrue((env.dof_pos[[1, 3]] >= env.dof_pos_limits[:, 0]).all())
        self.assertTrue((env.dof_pos[[1, 3]] <= env.dof_pos_limits[:, 1]).all())
        self.assertEqual(env.kick_detection_block_until_step[[1, 3]].tolist(), [5, 5])
        ball_state = env.root_states[[1, 3], 1]
        spawn_distance = (ball_state[:, :2] - env.base_pos[[1, 3], :2]).norm(dim=-1)
        self.assertTrue(((spawn_distance >= 1.5) & (spawn_distance <= 3.0)).all())
        self.assert_close(ball_state[:, 2], [0.075, 0.075])
        self.assertTrue((ball_state[:, 7:9].norm(dim=-1) <= 1.0).all())
        self.assert_close(ball_state[:, 10], -ball_state[:, 8] / env.ball_radius)
        self.assert_close(ball_state[:, 11], ball_state[:, 7] / env.ball_radius)
        self.assertTrue(((env.sampled_ball_restitution[[1, 3]] >= 0)
                         & (env.sampled_ball_restitution[[1, 3]] <= 0.7)).all())
        self.assertTrue(any(write[0] == "dof" for write in env.writes))
        self.assertTrue(any(write[0] == "restitution" for write in env.writes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
