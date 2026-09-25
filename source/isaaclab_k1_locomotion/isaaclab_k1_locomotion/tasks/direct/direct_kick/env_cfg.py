"""IsaacLab physical settings for the pinned DirectKick task."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from ....direct_kick.config import ROOT, load_config


@configclass
class DirectKickEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 17.0
    action_space = 12
    observation_space = load_config()['env']['num_observations']
    state_space = 20
    seed = 42
    sim = sim_utils.SimulationCfg(
        dt=0.002, render_interval=10, gravity=(0., 0., -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1, min_position_iteration_count=8, max_position_iteration_count=8,
            min_velocity_iteration_count=4, max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2, gpu_max_rigid_contact_count=8388608,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1., dynamic_friction=1., restitution=0.,
        ),
    )
    scene = InteractiveSceneCfg(num_envs=8192, env_spacing=10., replicate_physics=False)
    robot = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ROOT / 'assets/K1/K1_locomotion.urdf'),
            fix_base=False, merge_fixed_joints=True, self_collision=True,
            replace_cylinders_with_capsules=False, link_density=0.001,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type='none', gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0., angular_damping=0.,
                max_linear_velocity=1000., max_angular_velocity=1000., max_depenetration_velocity=1.,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.545), joint_pos={'.*': 0.}, joint_vel={'.*': 0.}),
        # The original task clamps its own PD effort. Its modified dof_props
        # (armature/velocity limits) were never applied to the simulator.
        actuators={'legs': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0., damping=0., armature=0.)},
    )
    ball = RigidObjectCfg(
        prim_path='/World/envs/env_.*/Ball',
        spawn=sim_utils.SphereCfg(
            radius=0.075,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0., angular_damping=0., max_linear_velocity=1000.,
                max_angular_velocity=1000., max_depenetration_velocity=1.,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0., 0.075)),
    )
    contacts = ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*', update_period=0., history_length=1)
    # The trainer supplies its run directory; the environment keeps the source CSV schema.
    log_dir: str = 'logs/direct_kick'
    enable_csv_logging: bool = True
