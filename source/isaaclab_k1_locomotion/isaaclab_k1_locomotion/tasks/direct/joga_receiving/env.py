"""Native K1/PhysX receiving, following Joga's published task where applicable."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from ....direct_kick.config import ROOT
from ....direct_kick.math_xyzw import get_euler_xyz, quat_rotate, quat_rotate_inverse
from ....direct_kick.k1_foot_kinematics import k1_foot_contact_points_root
from ....direct_kick.ego_motion import forecast_constant_body_twist, relative_state_and_covariance
from ....direct_kick.direct_kicking_observation import build_horizon_tokens
from ....direct_kick.utils import apply_randomization
from ....joga_receiving.config import load_config, curriculum_at, FRAME_SIZE, HISTORY_LENGTH
from ....joga_receiving.perception import BallBelief
from ....joga_receiving.task import ReceivingTask, sample_ground_pass, torque_speed_limit, angular_momentum
from ....joga_receiving.terrain import make_tile, tile_height
from .env_cfg import JogaReceiveEnvCfg


class JogaReceiveEnv(DirectRLEnv):
    cfg: JogaReceiveEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.task_cfg = cfg.task_settings if cfg.task_settings is not None else load_config()
        cfg.sim.dt = self.task_cfg["physics_dt"]
        cfg.decimation = self.task_cfg["decimation"]
        cfg.sim.render_interval = cfg.decimation
        cfg.episode_length_s = self.task_cfg["completion"]["episode_s"]
        cfg.ball.spawn.radius = self.task_cfg["ball"]["radius"]
        cfg.ball.spawn.mass_props.mass = self.task_cfg["ball"]["mass"]
        cfg.ball.spawn.physics_material.restitution = self.task_cfg["physx"]["ball_restitution"]
        cfg.ball.spawn.collision_props.contact_offset = self.task_cfg["physx"]["contact_offset_m"]
        cfg.robot.spawn.collision_props.contact_offset = self.task_cfg["physx"]["contact_offset_m"]
        self.iteration = 0
        urdf = ET.parse(ROOT / "assets/K1/K1_locomotion.urdf").getroot()
        self._joints = [joint for joint in urdf.findall("joint") if joint.get("type") != "fixed"]
        self._body_names = ["Trunk"] + [joint.find("child").get("link") for joint in self._joints]
        super().__init__(cfg, render_mode, **kwargs)
        self._bind_assets()
        n = self.num_envs
        self.root_states = torch.zeros(n, 2, 13, device=self.device)
        self.root_states[..., 6] = 1.
        self.actions = torch.zeros(n, 12, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self._targets = torch.zeros_like(self.actions)
        self.episode_return = torch.zeros(n, device=self.device)
        self._reward = torch.zeros_like(self.episode_return)
        self._success = torch.zeros(n, device=self.device, dtype=torch.bool)
        self._fall = torch.zeros_like(self._success)
        self._timeout = torch.zeros_like(self._success)
        self._pending_finish = torch.zeros_like(self._success)
        self._history_reset = torch.ones_like(self._success)
        self._policy_history = torch.zeros(n, HISTORY_LENGTH, FRAME_SIZE, device=self.device)
        self._critic_history = torch.zeros_like(self._policy_history)
        self._obs_step = -1
        self._belief_step = -1
        self.air_time = torch.zeros(n, 2, device=self.device)
        self.foot_ground_force = torch.zeros(n, 2, 3, device=self.device)
        self.foot_ground_contact = torch.zeros(n, 2, device=self.device, dtype=torch.bool)
        self.encoder_bias = torch.empty(n, 12, device=self.device).uniform_(
            -self.task_cfg["randomization"]["encoder_bias_rad"], self.task_cfg["randomization"]["encoder_bias_rad"])
        self.next_push = torch.zeros(n, device=self.device, dtype=torch.long)
        self.belief = BallBelief(self.task_cfg["perception"], n, self.step_dt, self.device)
        self.receiving = ReceivingTask(self.task_cfg, n, self.device)
        self._metrics = torch.zeros(7, device=self.device)
        self.spawn_speed = torch.zeros(n, device=self.device)
        self.spawn_offset = torch.zeros_like(self.spawn_speed)
        self._refresh_state()

    def _mesh(self, path, vertices, faces, material):
        mesh = UsdGeom.Mesh.Define(self.sim.stage, path)
        mesh.CreatePointsAttr(vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material.Get(self.sim.stage, material), UsdShade.Tokens.weakerThanDescendants, "physics")

    def _setup_scene(self):
        physics = self.task_cfg["physx"]
        material_path = "/World/receiving_ground_material"
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics["ground_friction"], dynamic_friction=physics["ground_friction"],
            restitution=0., friction_combine_mode="min", restitution_combine_mode="max")
        material.func(material_path, material)
        extent = self.cfg.scene.env_spacing * math.ceil(math.sqrt(self.num_envs)) + 20.
        floor = np.array([[-extent, -extent, 0.], [extent, -extent, 0.],
                          [-extent, extent, 0.], [extent, extent, 0.]], dtype=np.float32)
        self._mesh("/World/ground", floor, np.array([[0, 1, 2], [2, 1, 3]]), material_path)
        vertices, faces, heights = make_tile(physics["terrain_size_m"], physics["terrain_grid_size"],
                                            self.task_cfg["randomization"]["rough_height_m"])
        self._mesh("/World/envs/env_0/RoughTerrain", vertices, faces, material_path)
        self.robot = Articulation(self.cfg.robot)
        foot_material_path = "/World/receiving_robot_material"
        foot_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=physics["foot_friction"], dynamic_friction=physics["foot_friction"],
            restitution=physics["foot_restitution"], friction_combine_mode="min", restitution_combine_mode="min")
        foot_material.func(foot_material_path, foot_material)
        UsdShade.MaterialBindingAPI.Apply(self.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")).Bind(
            UsdShade.Material.Get(self.sim.stage, foot_material_path), UsdShade.Tokens.strongerThanDescendants, "physics")
        self.ball = RigidObject(self.cfg.ball)
        self.contacts = ContactSensor(self.cfg.contacts)
        filters = [f"/World/envs/env_.*/Robot/{name}" for name in self._body_names]
        self.ball_contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Ball", filter_prim_paths_expr=filters,
            update_period=0., history_length=self.cfg.decimation))
        self.self_contacts = {}
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.sensors["contacts"] = self.contacts
        self.scene.sensors["ball_contacts"] = self.ball_contacts
        for name, path in zip(self._body_names, filters):
            sensor = ContactSensor(ContactSensorCfg(prim_path=path,
                filter_prim_paths_expr=[other for other in filters if other != path],
                update_period=0., history_length=self.cfg.decimation))
            self.self_contacts[name] = sensor
            self.scene.sensors["self_" + name] = sensor
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        rough = np.random.rand(self.num_envs) < self.task_cfg["randomization"]["rough_probability"]
        self.rough_envs = torch.tensor(rough, device=self.device)
        self.terrain_heights = torch.tensor(heights, device=self.device)
        com_range = self.task_cfg["randomization"]["com_offset_m"]
        inertia = (2. / 3.) * self.task_cfg["ball"]["mass"] * self.task_cfg["ball"]["radius"] ** 2
        for index in range(self.num_envs):
            root = f"/World/envs/env_{index}"
            if not rough[index]:
                prim = self.sim.stage.GetPrimAtPath(root + "/RoughTerrain")
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                UsdGeom.Imageable(prim).MakeInvisible()
            sim_utils.make_uninstanceable(root + "/Robot")
            base = UsdPhysics.MassAPI.Apply(self.sim.stage.GetPrimAtPath(root + "/Robot/Trunk"))
            center = base.GetCenterOfMassAttr().Get()
            if center is None or not all(np.isfinite(float(value)) for value in center):
                raise RuntimeError("K1 torso COM is unavailable after URDF import")
            offset = np.random.uniform(-np.asarray(com_range), np.asarray(com_range))
            base.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(center[j]) + float(offset[j]) for j in range(3)]))
            ball_mass = UsdPhysics.MassAPI.Apply(self.sim.stage.GetPrimAtPath(root + "/Ball"))
            ball_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
        light = sim_utils.DomeLightCfg(intensity=2000.)
        light.func("/World/Light", light)

    def _bind_assets(self):
        names = [joint.get("name") for joint in self._joints]
        if len(names) != 12 or self.robot.num_joints != 12:
            raise RuntimeError("Receiving adaptation requires the existing K1 12-leg-joint asset")
        self.joint_ids = [self.robot.joint_names.index(name) for name in names]
        self.body_ids = [self.robot.body_names.index(name) for name in self._body_names]
        foot_names = self.task_cfg["robot"]["asset"]["foot_names"]
        self.foot_ids = [self.robot.body_names.index(name) for name in foot_names]
        self.foot_filter_ids = [self._body_names.index(name) for name in foot_names]
        self.leg_filter_ids = [i for i, name in enumerate(self._body_names) if name != "Trunk" and name not in foot_names]
        self.contact_ids = [self.contacts.body_names.index(name) for name in self.robot.body_names]
        self.ankle_pitch_indices = [i for i, name in enumerate(names) if "Ankle_Pitch" in name]
        source = self.task_cfg["robot"]
        def values(section):
            return torch.tensor([next(value for pattern, value in source["control"][section].items()
                                      if pattern in name) for name in names], device=self.device)
        self.stiffness, self.damping = values("stiffness"), values("damping")
        self.stall_torque, self.no_load_speed = values("effort_limit"), values("velocity_limit")
        angles = source["init_state"]["default_joint_angles"]
        self.default_pose = torch.tensor([next((value for pattern, value in angles.items()
                                               if pattern != "default" and pattern in name), angles["default"])
                                          for name in names], device=self.device)
        self.lower = torch.tensor([float(joint.find("limit").get("lower")) for joint in self._joints], device=self.device)
        self.upper = torch.tensor([float(joint.find("limit").get("upper")) for joint in self._joints], device=self.device)
        self.foot_contact_points = torch.tensor(source["asset"]["feet_edge_pos"], device=self.device)
        self.mass = self.robot.root_physx_view.get_masses().to(self.device)
        self.inertia = self.robot.root_physx_view.get_inertias().to(self.device).reshape(self.num_envs, -1, 3, 3)
        self._foot_shape_ranges = []
        counts = [self.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                  for path in self.robot.root_physx_view.link_paths[0]]
        offsets = np.cumsum([0] + counts)
        for body in self.foot_ids:
            self._foot_shape_ranges.append(slice(offsets[body], offsets[body + 1]))
        materials = self.robot.root_physx_view.get_material_properties().clone()
        if offsets[-1] != materials.shape[1]:
            raise RuntimeError("K1 material shape mapping differs from the body view")
        materials[..., 2] = self.task_cfg["physx"]["foot_restitution"]
        self.robot.root_physx_view.set_material_properties(materials, torch.arange(self.num_envs, dtype=torch.int32))

    def _ground_height(self, points, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        height = tile_height(points, self.terrain_heights, self.task_cfg["physx"]["terrain_size_m"])
        rough = self.rough_envs[ids]
        while rough.ndim < height.ndim:
            rough = rough.unsqueeze(-1)
        return height * rough

    def _write_roots(self, ids):
        for index, asset in enumerate((self.robot, self.ball)):
            state = self.root_states[ids, index].clone()
            state[:, :3] += self.scene.env_origins[ids]
            state[:, 3:7] = state[:, [6, 3, 4, 5]]
            asset.write_root_state_to_sim(state, env_ids=ids)

    def _randomize_materials(self, ids):
        cpu_ids = ids.cpu().to(torch.int32)
        robot = self.robot.root_physx_view.get_material_properties().clone()
        friction = torch.empty(len(ids)).uniform_(*self.task_cfg["randomization"]["foot_friction"])
        for span in self._foot_shape_ranges:
            robot[cpu_ids.long(), span, 0] = friction[:, None]
            robot[cpu_ids.long(), span, 1] = friction[:, None]
        self.robot.root_physx_view.set_material_properties(robot, cpu_ids)
        ball = self.ball.root_physx_view.get_material_properties().clone()
        friction = torch.empty(len(ids)).uniform_(*self.task_cfg["randomization"]["ball_friction"])
        ball[cpu_ids.long(), :, 0] = friction[:, None]
        ball[cpu_ids.long(), :, 1] = friction[:, None]
        self.ball.root_physx_view.set_material_properties(ball, cpu_ids)

    def set_curriculum_iteration(self, iteration):
        self.iteration = int(iteration)

    def _reset_idx(self, env_ids):
        finished = env_ids[self._pending_finish[env_ids]]
        if len(finished):
            self._metrics += torch.stack((self._pending_finish[finished].float().sum(),
                self._success[finished].float().sum(), self._fall[finished].float().sum(),
                self._timeout[finished].float().sum(), self.episode_return[finished].sum(),
                self.episode_length_buf[finished].float().sum(),
                self.root_states[finished, 1, 7:10].norm(dim=-1).sum()))
        DirectRLEnv._reset_idx(self, env_ids)
        count = len(env_ids)
        if not count:
            return
        stage = curriculum_at(self.task_cfg, self.iteration)
        root = self.root_states[env_ids].clone().zero_()
        yaw = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        root[:, :, 6] = 1.
        root[:, 0, :2] = torch.empty(count, 2, device=self.device).uniform_(-.5, .5)
        root[:, 0, 5], root[:, 0, 6] = (yaw / 2.).sin(), (yaw / 2.).cos()
        q = self.default_pose.expand(count, -1).clone()
        foot_points = k1_foot_contact_points_root(q, torch.arange(12, device=self.device).reshape(2, 6),
                                                self.foot_contact_points)
        rotated = quat_rotate(root[:, 0, None, None, 3:7].expand(-1, 2, len(self.foot_contact_points), -1).reshape(-1, 4),
                              foot_points.reshape(-1, 3)).reshape_as(foot_points)
        world_points = rotated[..., :2] + root[:, 0, None, None, :2]
        terrain = self._ground_height(world_points, env_ids)
        root[:, 0, 2] = (terrain - rotated[..., 2]).amax(dim=(1, 2)) + .003
        local, velocity, speed, offset = sample_ground_pass(count, stage, self.device)
        rotation = torch.stack((yaw.cos(), -yaw.sin(), yaw.sin(), yaw.cos()), dim=-1).reshape(-1, 2, 2)
        root[:, 1, :2] = root[:, 0, :2] + (rotation @ local[..., None]).squeeze(-1)
        root[:, 1, 2] = self._ground_height(root[:, 1, :2], env_ids) + self.task_cfg["ball"]["radius"]
        root[:, 1, 7:9] = (rotation @ velocity[..., None]).squeeze(-1)
        root[:, 1, 10] = -root[:, 1, 8] / self.task_cfg["ball"]["radius"]
        root[:, 1, 11] = root[:, 1, 7] / self.task_cfg["ball"]["radius"]
        self.root_states[env_ids] = root
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=self.joint_ids, env_ids=env_ids)
        self._write_roots(env_ids)
        self._randomize_materials(env_ids)
        self.spawn_speed[env_ids], self.spawn_offset[env_ids] = speed, offset
        self.receiving.reset(env_ids, stage, speed)
        self.belief.reset(env_ids, self.root_states)
        self.actions[env_ids] = self.previous_actions[env_ids] = 0.
        self.episode_return[env_ids] = 0.
        self.air_time[env_ids] = 0.
        self.foot_ground_force[env_ids] = 0.
        self.foot_ground_contact[env_ids] = False
        self._success[env_ids] = self._fall[env_ids] = self._timeout[env_ids] = False
        self._pending_finish[env_ids] = False
        self._history_reset[env_ids] = True
        low, high = self.task_cfg["randomization"]["push_interval_s"]
        self.next_push[env_ids] = self.common_step_counter + torch.ceil(
            torch.empty(count, device=self.device).uniform_(low, high) / self.step_dt).long()

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        self._targets = self.default_pose + self.task_cfg["robot"]["control"]["action_scale"] * actions
        ids = (self.common_step_counter >= self.next_push).nonzero(as_tuple=False).flatten()
        if len(ids):
            random = self.task_cfg["randomization"]
            self.root_states[ids, 0, 7:10] += ((2. * torch.rand(len(ids), 3, device=self.device) - 1.)
                                              * actions.new_tensor(random["push_linear_mps"]))
            self.root_states[ids, 0, 10:13] += ((2. * torch.rand(len(ids), 3, device=self.device) - 1.)
                                               * actions.new_tensor(random["push_angular_rps"]))
            state = self.root_states[ids, 0].clone()
            state[:, :3] += self.scene.env_origins[ids]
            state[:, 3:7] = state[:, [6, 3, 4, 5]]
            self.robot.write_root_state_to_sim(state, env_ids=ids)
            self.next_push[ids] = self.common_step_counter + torch.ceil(
                torch.empty(len(ids), device=self.device).uniform_(*random["push_interval_s"]) / self.step_dt).long()

    def _apply_action(self):
        _, torque = torque_speed_limit(self._targets, self.robot.data.joint_pos[:, self.joint_ids],
                                      self.robot.data.joint_vel[:, self.joint_ids], self.stiffness, self.damping,
                                      self.stall_torque, self.no_load_speed)
        self.robot.set_joint_effort_target(torque, joint_ids=self.joint_ids)

    def _refresh_state(self):
        for index, asset in enumerate((self.robot, self.ball)):
            state = asset.data.root_state_w
            self.root_states[:, index] = state
            self.root_states[:, index, :3] -= self.scene.env_origins
            self.root_states[:, index, 3:7] = state[:, [4, 5, 6, 3]]
        self.q = self.robot.data.joint_pos[:, self.joint_ids]
        self.dq = self.robot.data.joint_vel[:, self.joint_ids]
        base = self.root_states[:, 0]
        self.base_velocity = quat_rotate_inverse(base[:, 3:7], base[:, 7:10])
        self.base_angular_velocity = quat_rotate_inverse(base[:, 3:7], base[:, 10:13])
        down = torch.zeros(self.num_envs, 3, device=self.device)
        down[:, 2] = -1.
        self.gravity = quat_rotate_inverse(base[:, 3:7], down)
        self.foot_position = self.robot.data.body_link_pos_w[:, self.foot_ids] - self.scene.env_origins[:, None]
        self.foot_quat = self.robot.data.body_link_quat_w[:, self.foot_ids][..., [1, 2, 3, 0]]

    def _get_dones(self):
        self._refresh_state()
        self._belief_features, self._observed_angular = self.belief.observe(
            self.root_states, self.common_step_counter, advance=True)
        self._belief_step = self.common_step_counter
        ball_force = self.ball_contacts.data.force_matrix_w
        history = self.ball_contacts.data.force_matrix_w_history
        if ball_force.shape[2] != len(self._body_names):
            raise RuntimeError("Filtered contact ordering does not match K1 body filters")
        pair_max = history[:, :, 0].norm(dim=-1).amax(dim=1)
        self_forces = []
        self_current = []
        for name in self._body_names:
            sensor = self.self_contacts[name].data
            self_forces.append(sensor.force_matrix_w_history[:, :, 0].norm(dim=-1).amax(dim=(1, 2)))
            self_current.append(sensor.force_matrix_w.sum(dim=2)[:, 0])
        self_current = torch.stack(self_current, dim=1)
        foot_self = self_current[:, self.foot_filter_ids]
        # Ball sensor reports force ON the ball. Add it to the foot's net force
        # to remove the equal/opposite ball contribution from ground sensing.
        foot_net = self.contacts.data.net_forces_w[:, [self.contact_ids[i] for i in self.foot_ids]]
        self.foot_ground_force = foot_net + ball_force[:, 0, self.foot_filter_ids] - foot_self
        self.foot_ground_contact = self.foot_ground_force.norm(dim=-1) > self.task_cfg["completion"]["ground_contact_threshold_n"]
        self.air_time = torch.where(self.foot_ground_contact, 0., self.air_time + self.step_dt)
        base = self.root_states[:, 0]
        ball = self.root_states[:, 1]
        _, _, yaw = get_euler_xyz(base[:, 3:7])
        yaw_quat = torch.zeros_like(base[:, 3:7])
        yaw_quat[:, 2], yaw_quat[:, 3] = (yaw / 2.).sin(), (yaw / 2.).cos()
        ball_local = quat_rotate_inverse(yaw_quat, ball[:, :3] - base[:, :3])
        foot_local = quat_rotate_inverse(yaw_quat[:, None].expand(-1, 2, -1).reshape(-1, 4),
                                         (self.foot_position - base[:, None, :3]).reshape(-1, 3)).reshape(-1, 2, 3)
        momentum = angular_momentum(self.mass, self.robot.data.body_com_pos_w,
                                    self.robot.data.body_com_vel_w,
                                    self.robot.data.body_link_quat_w[..., [1, 2, 3, 0]], self.inertia)
        reward_state = {
            "ball_local": ball_local, "ball_position": ball[:, :3], "ball_velocity": ball[:, 7:10],
            "ball_ground_height": ball[:, 2] - self._ground_height(ball[:, :2]),
            "foot_position": self.foot_position, "foot_quat": self.foot_quat, "foot_local": foot_local,
            "foot_ball_force": pair_max[:, self.foot_filter_ids], "self_contact_force": torch.stack(self_forces, 1),
            "leg_ball_force": pair_max[:, self.leg_filter_ids],
            "visible": self.belief.visible, "base_velocity": self.base_velocity,
            "base_angular_velocity": self.base_angular_velocity, "gravity": self.gravity,
            "base_height": base[:, 2] - self._ground_height(base[:, :2]),
            "joint_position": self.q, "default_joint_position": self.default_pose,
            "joint_lower": self.lower, "joint_upper": self.upper, "ankle_pitch_indices": self.ankle_pitch_indices,
            "action": self.actions, "previous_action": self.previous_actions, "angular_momentum": momentum,
        }
        self._reward, terms, self._success, self._fall = self.receiving.evaluate(reward_state)
        self.episode_return += self._reward
        self._timeout = ((self.episode_length_buf >= self.max_episode_length) & ~self._success & ~self._fall)
        self._pending_finish = self._success | self._fall | self._timeout
        self.extras["rew_terms"] = terms
        self.extras["time_outs"] = self._timeout.clone()
        return self._success | self._fall, self.extras["time_outs"].clone()

    def _get_rewards(self):
        return self._reward

    def _frame(self, belief, angular, noisy):
        base = self.root_states[:, 0]
        _, _, yaw = get_euler_xyz(base[:, 3:7])
        noise = self.task_cfg["perception"]["noise"]
        gravity = apply_randomization(self.gravity, noise["gravity"]) if noisy else self.gravity
        position = self.q - self.default_pose
        velocity = self.dq
        if noisy:
            position = apply_randomization(position, noise["dof_pos"]) + self.encoder_bias
            velocity = apply_randomization(velocity, noise["dof_vel"])
        # 47 proprioceptive values: g3, omega3, q12, dq12, action12,
        # commanded world ball velocity3 (=0), heading cos/sin2.
        proprio = torch.cat((gravity, angular, position, velocity, self.actions,
                             torch.zeros(self.num_envs, 3, device=self.device),
                             torch.stack((yaw.cos(), yaw.sin()), dim=-1)), dim=-1)
        return torch.cat((proprio, belief), dim=-1)

    def _truth_belief(self):
        horizons = torch.tensor(self.belief.prediction_horizons, device=self.device)
        base, ball = self.root_states[:, 0], self.root_states[:, 1]
        _, _, yaw = get_euler_xyz(base[:, 3:7])
        yaw_velocity = (self.belief._world_to_local_rotation(yaw) @ base[:, 7:9, None]).squeeze(-1)
        offsets = horizons[None].expand(self.num_envs, -1)
        base_position, base_yaw = forecast_constant_body_twist(base[:, :2], yaw, yaw_velocity,
                                                              self.base_angular_velocity[:, 2], offsets)
        mean = torch.cat((ball[:, None, :2] + offsets[..., None] * ball[:, None, 7:9],
                          ball[:, None, 7:9].expand(-1, len(horizons), -1)), dim=-1)
        covariance = torch.zeros(self.num_envs, len(horizons), 4, 4, device=self.device)
        local, covariance = relative_state_and_covariance(mean, covariance, base_position, base_yaw,
            yaw_velocity[:, None].expand(-1, len(horizons), -1),
            self.base_angular_velocity[:, None, 2].expand(-1, len(horizons)),
            torch.zeros(self.num_envs, len(horizons), 6, 6, device=self.device))
        valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        tokens = build_horizon_tokens(local, covariance, self.belief.prediction_horizons, valid,
                                       torch.zeros(4, 4, device=self.device))
        status = torch.tensor([0., 1., 1.], device=self.device).expand(self.num_envs, -1)
        return torch.cat((tokens.flatten(1), status), dim=-1)

    def _get_observations(self):
        if self._obs_step == self.common_step_counter and not self._history_reset.any():
            return self._observations
        self._refresh_state()
        advance = self._obs_step != self.common_step_counter and self.common_step_counter > 0
        if self._belief_step == self.common_step_counter and not self._history_reset.any():
            belief, angular = self._belief_features, self._observed_angular
        else:
            belief, angular = self.belief.observe(self.root_states, self.common_step_counter,
                                                  advance and self._belief_step != self.common_step_counter)
            self._belief_step = self.common_step_counter
            self._belief_features, self._observed_angular = belief, angular
        frame = self._frame(belief, angular, noisy=True)
        truth = self._frame(self._truth_belief(), self.base_angular_velocity, noisy=False)
        if advance:
            self._policy_history = self._policy_history.roll(-1, dims=1)
            self._critic_history = self._critic_history.roll(-1, dims=1)
        self._policy_history[:, -1] = frame
        self._critic_history[:, -1] = truth
        ids = self._history_reset.nonzero(as_tuple=False).flatten()
        self._policy_history[ids] = frame[ids, None]
        self._critic_history[ids] = truth[ids, None]
        self._history_reset.zero_()
        self._obs_step = self.common_step_counter
        base, ball = self.root_states[:, 0], self.root_states[:, 1]
        foot_height = self.foot_position[..., 2] - self._ground_height(self.foot_position[..., :2])
        privileged = torch.cat((self.base_velocity, foot_height, self.foot_ground_contact.float(), self.air_time,
            self.foot_ground_force.flatten(1), quat_rotate_inverse(base[:, 3:7], ball[:, :3] - base[:, :3]),
            quat_rotate_inverse(base[:, 3:7], ball[:, 7:10])), dim=-1)
        self._observations = {"policy": self._policy_history.flatten(1),
                              "critic": torch.cat((self._critic_history.flatten(1), privileged), dim=-1)}
        return self._observations

    def drain_episode_metrics(self):
        episodes, success, falls, timeouts, returns, lengths, residual_speed = self._metrics.tolist()
        self._metrics.zero_()
        return {"episodes": int(episodes), "success_rate": success / episodes if episodes else "",
                "fall_rate": falls / episodes if episodes else "", "timeout_rate": timeouts / episodes if episodes else "",
                "mean_return": returns / episodes if episodes else "",
                "mean_episode_steps": lengths / episodes if episodes else "",
                "mean_final_ball_speed": residual_speed / episodes if episodes else ""}
