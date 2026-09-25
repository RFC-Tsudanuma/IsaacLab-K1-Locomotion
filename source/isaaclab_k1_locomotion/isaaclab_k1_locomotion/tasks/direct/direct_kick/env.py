"""Native IsaacLab backend for the pinned DirectKick task math."""
import xml.etree.ElementTree as ET

import numpy as np
import torch
from pxr import Gf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from ....direct_kick.config import ROOT, load_config
from ....direct_kick.direct_kicking_logic import DirectKickingLogic
from ....direct_kick.math_xyzw import quat_rotate_inverse
from ....direct_kick.utils import apply_randomization
from .env_cfg import DirectKickEnvCfg


class DirectKickEnv(DirectKickingLogic, DirectRLEnv):
    cfg: DirectKickEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.task_cfg = load_config()
        self.task_cfg['env']['num_envs'] = cfg.scene.num_envs
        self.task_cfg['basic']['enable_csv_logging'] = cfg.enable_csv_logging
        self.task_cfg['basic']['log_dir'] = cfg.log_dir
        self.direct_cfg = self.task_cfg['direct_kicking']
        self._configure_ball_motion(self.task_cfg)
        self._configure_action_delay(self.task_cfg)
        self._configure_external_disturbances(self.task_cfg)
        self._perception_step_requested = False
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        self._bind_task_assets()
        self._init_buffers()
        self._init_stage_curriculum()
        self._prepare_reward_function()
        self._init_csv_logging()
        self.env_resets = self.env_successes = self.env_falling = 0
        self._refresh_state()

    def _setup_scene(self):
        # DirectRLEnv seeds NumPy/Torch before scene creation.
        self._randomize_ground_material(self.task_cfg)
        ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=self.sampled_ground_static_friction,
            dynamic_friction=self.sampled_ground_dynamic_friction,
            restitution=self.task_cfg['terrain']['restitution'],
        ))
        ground.func('/World/ground', ground)
        self.robot = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)
        self.contacts = ContactSensor(self.cfg.contacts)
        self.scene.articulations['robot'] = self.robot
        self.scene.rigid_objects['ball'] = self.ball
        self.scene.sensors['contacts'] = self.contacts
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        sim_utils.DomeLightCfg(intensity=2000.).func('/World/Light', sim_utils.DomeLightCfg(intensity=2000.))
        self._author_mass_properties()

    def _author_mass_properties(self):
        """Recompute inertia from collision geometry and each sampled mass/COM.

        Zero inertia/principalAxes request PhysX auto-computation, including the
        parallel-axis contribution of the specified COM. This runs before reset.
        """
        self._base_mass_noise = np.zeros((self.num_envs, 4), dtype=np.float32)
        stage = self.sim.stage
        random = self.task_cfg['randomization']
        for env_id in range(self.num_envs):
            root = stage.GetPrimAtPath(f'/World/envs/env_{env_id}/Robot')
            sim_utils.make_uninstanceable(str(root.GetPath()))
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                api = UsdPhysics.MassAPI.Apply(prim)
                base = prim.GetName() == self.task_cfg['asset']['base_name']
                prefix = 'base' if base else 'other'
                com = api.GetCenterOfMassAttr().Get()
                mass = api.GetMassAttr().Get()
                if com is None or not all(np.isfinite(float(v)) for v in com) or mass is None or mass <= 0:
                    raise RuntimeError(f'Imported mass/COM missing for {prim.GetPath()}')
                values = []
                for axis in range(3):
                    value, noise = apply_randomization(float(com[axis]), random[prefix + '_com'], return_noise=True)
                    values.append(value)
                    if base:
                        self._base_mass_noise[env_id, axis] = noise
                value, noise = apply_randomization(float(mass), random[prefix + '_mass'], return_noise=True)
                if base:
                    self._base_mass_noise[env_id, 3] = noise
                api.CreateMassAttr().Set(float(value))
                api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*values))
                api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.))
                api.CreatePrincipalAxesAttr().Set(Gf.Quatf(0.))

    def _bind_task_assets(self):
        urdf = ET.parse(ROOT / 'assets/K1/K1_locomotion.urdf').getroot()
        joints = [j for j in urdf.findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')]
        self.dof_names = [j.get('name') for j in joints]
        self.joint_ids = [self.robot.joint_names.index(name) for name in self.dof_names]
        if len(self.joint_ids) != 12 or self.robot.num_joints != 12:
            raise RuntimeError('DirectKick requires exactly the 12 source leg joints')
        self.num_dofs = len(self.joint_ids)
        # Body tensors remain in the backend order; every semantic index is name-bound.
        self.robot_body_names = self.robot.body_names
        self.num_bodies = self.robot.num_bodies
        self.base_indice = self.robot.body_names.index(self.task_cfg['asset']['base_name'])
        if self.base_indice != 0:
            raise RuntimeError('Source privileged external-force fields require root body index zero')
        self.feet_indices = torch.tensor([self.robot.body_names.index(n) for n in self.task_cfg['asset']['foot_names']], device=self.device)
        def contact_indices(key):
            return torch.tensor([i for i, name in enumerate(self.robot.body_names)
                                 if any(part in name for part in self.task_cfg['rewards'][key])],
                                dtype=torch.long, device=self.device)
        self.penalized_contact_indices = contact_indices('penalize_contacts_on')
        self.termination_contact_indices = contact_indices('terminate_contacts_on')
        self.contact_ids = [self.contacts.body_names.index(name) for name in self.robot.body_names]
        self.dof_pos_limits = torch.tensor([[float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))] for j in joints], device=self.device)
        control = self.task_cfg['control']
        def joint_values(key):
            return torch.tensor([next(value for pattern, value in control[key].items() if pattern in name)
                                 for name in self.dof_names], device=self.device)
        self.torque_limits = joint_values('effort_limit')
        self.dof_vel_limits = joint_values('velocity_limit')
        self.dof_stiffness = apply_randomization(joint_values('stiffness').repeat(self.num_envs, 1), self.task_cfg['randomization']['dof_stiffness'])
        self.dof_damping = apply_randomization(joint_values('damping').repeat(self.num_envs, 1), self.task_cfg['randomization']['dof_damping'])
        self.dof_friction = apply_randomization(torch.zeros_like(self.dof_stiffness), self.task_cfg['randomization']['dof_friction'])
        self.base_mass_scaled = torch.tensor(self._base_mass_noise, device=self.device)
        del self._base_mass_noise
        self.robot_body_masses = self.robot.root_physx_view.get_masses().to(self.device)
        self.robot_body_local_com = self.robot.root_physx_view.get_coms()[..., :3].to(self.device)
        init = self.task_cfg['init_state']
        self.base_init_state = torch.tensor(init['pos'] + init['rot'] + init['lin_vel'] + init['ang_vel'], device=self.device)
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.up_axis_idx = 2
        self.ball_radius = self.task_cfg['ball']['radius']
        self._randomize_materials()

    def _randomize_materials(self):
        ids = torch.arange(self.num_envs, dtype=torch.int32)
        materials = self.robot.root_physx_view.get_material_properties().clone()
        shape_counts = [self.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                        for path in self.robot.root_physx_view.link_paths[0]]
        if sum(shape_counts) != materials.shape[1]:
            raise RuntimeError('Robot shape mapping does not match PhysX material tensor')
        offset = np.cumsum([0] + shape_counts)
        # Independent continuous draws for every foot collision shape, as in Gym.
        for body in self.feet_indices.cpu().tolist():
            span = slice(offset[body], offset[body + 1])
            shape = materials[:, span, 0].shape
            friction = apply_randomization(torch.zeros(shape), self.task_cfg['randomization']['friction'])
            restitution = apply_randomization(torch.zeros(shape), self.task_cfg['randomization']['restitution'])
            materials[:, span, 0] = materials[:, span, 1] = friction
            materials[:, span, 2] = restitution
        self.robot.root_physx_view.set_material_properties(materials, ids)
        physics = self.direct_cfg['physics_randomization']
        self.ball_physics_randomization_enabled = bool(physics['enabled'])
        self.ball_restitution_range = tuple(physics['ball_restitution_range'])
        scales = np.random.uniform(*physics['ball_friction_scale_range'], size=self.num_envs) if physics['enabled'] else np.ones(self.num_envs)
        self.sampled_ball_friction = torch.tensor(scales * self.task_cfg['ball']['friction'], device=self.device, dtype=torch.float32)
        self.sampled_ball_restitution = torch.full((self.num_envs,), self.task_cfg['ball']['restitution'], device=self.device)
        material = self.ball.root_physx_view.get_material_properties().clone()
        material[..., 0] = material[..., 1] = self.sampled_ball_friction.cpu()[:, None]
        material[..., 2] = self.task_cfg['ball']['restitution']
        self.ball.root_physx_view.set_material_properties(material, ids)

    def _write_ball_restitution(self, ids, values):
        material = self.ball.root_physx_view.get_material_properties().clone()
        material[ids.cpu(), :, 2] = values.cpu()[:, None]
        self.ball.root_physx_view.set_material_properties(material, ids.cpu())

    def _write_dof_state(self, ids):
        self.robot.write_joint_state_to_sim(self.dof_pos[ids], self.dof_vel[ids], joint_ids=self.joint_ids, env_ids=ids)

    def _write_root_states(self, ids, robot=False, ball=False):
        for enabled, index, asset in ((robot, 0, self.robot), (ball, 1, self.ball)):
            if enabled:
                state = self.root_states[ids, index].clone()
                state[:, :3] += self.scene.env_origins[ids]
                state[:, 3:7] = state[:, [6, 3, 4, 5]]
                asset.write_root_state_to_sim(state, env_ids=ids)

    def _write_external_forces(self):
        # Gym applies the load at the COM for the next physics step only.
        # A permanent Lab wrench would multiply the source impulse by decimation.
        forces = self.pushing_forces[:, :self.num_bodies].contiguous()
        # Lab composes at the link origin. Transport the COM wrench explicitly;
        # the installed composer's positions path overwrites the supplied torque.
        torques = self.pushing_torques[:, :self.num_bodies] + torch.cross(
            self.robot_body_local_com, forces, dim=-1
        )
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques.contiguous(),
            is_global=False,
        )

    def _terrain_heights(self, points):
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)

    def _refresh_state(self):
        self.dof_pos[:] = self.robot.data.joint_pos[:, self.joint_ids]
        self.dof_vel[:] = self.robot.data.joint_vel[:, self.joint_ids]
        origins = self.scene.env_origins
        for index, asset in enumerate((self.robot, self.ball)):
            state = asset.data.root_state_w
            self.root_states[:, index] = state
            self.root_states[:, index, :3] -= origins
            self.root_states[:, index, 3:7] = state[:, [4, 5, 6, 3]]
        # Match the source helpers' COM-position/link-orientation convention.
        # Legacy Gym tensor frame equivalence still needs simulator validation.
        self.body_states[:, :self.num_bodies, :3] = self.robot.data.body_com_pos_w - origins[:, None]
        self.body_states[:, :self.num_bodies, 3:7] = self.robot.data.body_link_quat_w[..., [1, 2, 3, 0]]
        self.body_states[:, :self.num_bodies, 7:13] = self.robot.data.body_com_vel_w
        self.body_states[:, -1] = self.root_states[:, 1]
        self.contact_forces[:, :self.num_bodies] = self.contacts.data.net_forces_w[:, self.contact_ids]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self._refresh_feet_state()

    def _pre_physics_step(self, actions):
        self.direct_previous_feet_ankle_pos_world[:] = self._feet_ankle_positions_world()
        self.direct_previous_ball_pos_world[:] = self.root_states[:, 1, :3]
        self.direct_previous_ball_lin_vel_world[:] = self.root_states[:, 1, 7:10]
        self.direct_previous_base_pos_world[:] = self.root_states[:, 0, :3]
        self.direct_previous_base_quat_world[:] = self.root_states[:, 0, 3:7]
        self.previous_feet_contact_buf[:] = self.feet_contact
        self._direct_kick_detection_processed = False
        self.actions[:] = actions.clamp(-self.task_cfg['normalization']['clip_actions'], self.task_cfg['normalization']['clip_actions'])
        self._dof_targets = self.default_dof_pos + self.task_cfg['control']['action_scale'] * self.actions
        self._substep = 0
        self.torques.zero_()

    def _apply_action(self):
        self.dof_pos[:] = self.robot.data.joint_pos[:, self.joint_ids]
        self.dof_vel[:] = self.robot.data.joint_vel[:, self.joint_ids]
        self._update_delayed_dof_targets(self._dof_targets, self._substep)
        torque = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
        torque -= torch.minimum(self.dof_friction, torque.abs()) * torque.sign()
        torque = torque.clamp(-self.torque_limits, self.torque_limits)
        self.torques += torque
        self.robot.set_joint_effort_target(torque, joint_ids=self.joint_ids)
        self._substep += 1

    def _get_dones(self):
        self.torques /= self.cfg.decimation
        self._refresh_state()
        weight = self.task_cfg['normalization']['filter_weight']
        self.filtered_lin_vel[:] = weight * self.base_lin_vel + (1 - weight) * self.filtered_lin_vel
        self.filtered_ang_vel[:] = weight * self.base_ang_vel + (1 - weight) * self.filtered_ang_vel
        self.min_ball_vel_buf = torch.where(self.ball_lin_vel[:, 0] > 0.1, self.min_ball_vel_buf + 1., 0.)
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.)
        active = self.root_states[:, 1, 7:10].norm(dim=-1) > self.task_cfg['rewards']['ball_stationary_speed_threshold']
        self.time_since_ball_is_still_buf = torch.where(active, 0., self.time_since_ball_is_still_buf + self.dt)
        self.time_since_ball_is_moving_buf = torch.where(~active, 0., self.time_since_ball_is_moving_buf + self.dt)
        self._kick_robots()
        self._push_robots()
        self._check_termination()
        return self.reset_buf & ~self.time_out_buf, self.time_out_buf.clone()

    def _get_rewards(self):
        self._compute_reward()
        self._update_valid_kick(self.direct_previous_ball_pos_world, self.direct_previous_ball_lin_vel_world)
        self._log_rewards_to_csv()
        self.last_ball_lin_vel_world[:] = self.body_states[:, -1, 7:10]
        self.extras['time_outs'] = self.time_out_buf.clone()
        return self.rew_buf

    def _reset_idx(self, env_ids):
        self._record_completed_episodes(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        DirectKickingLogic._reset_idx(self, env_ids)
        self.reset_ball_buf[env_ids] = False
        # scene.reset clears its wrench buffer, while source task owns the values.
        self._write_external_forces()

    def _get_observations(self):
        self._compute_observations()
        return {'policy': self.obs_buf, 'critic': self.privileged_obs_buf}

    def step(self, actions):
        self._perception_step_requested = True
        try:
            result = DirectRLEnv.step(self, actions)
            self.last_actions[:] = self.actions
            self.last_dof_vel[:] = self.dof_vel
            self.last_root_vel[:] = self.root_states[:, 0, 7:13]
            self.last_feet_pos[:] = self.feet_pos
            return result
        finally:
            self._perception_step_requested = False
