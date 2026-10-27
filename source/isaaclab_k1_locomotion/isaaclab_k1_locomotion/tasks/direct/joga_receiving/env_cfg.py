"""K1 / PhysX adaptation of Joga receiving (paper physics step = 5 ms)."""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ....joga_receiving.config import POLICY_SIZE, CRITIC_SIZE
from ..direct_kick.env_cfg import DirectKickEnvCfg


@configclass
class JogaReceiveEnvCfg(DirectKickEnvCfg):
    decimation = 4
    episode_length_s = 8.
    action_space = 12
    observation_space = POLICY_SIZE
    state_space = CRITIC_SIZE
    robot = DirectKickEnvCfg().robot
    robot.spawn.collision_props.contact_offset = .005
    scene = InteractiveSceneCfg(num_envs=16384, env_spacing=12., replicate_physics=False)
    sim = sim_utils.SimulationCfg(
        dt=.005, render_interval=4, gravity=(0., 0., -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1, min_position_iteration_count=8, max_position_iteration_count=8,
            min_velocity_iteration_count=4, max_velocity_iteration_count=4,
            bounce_threshold_velocity=.2, gpu_max_rigid_contact_count=8388608,
        ),
    )
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=.11, activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=.43),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0., angular_damping=0.,
                                                       max_depenetration_velocity=1.),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=.75, dynamic_friction=.75, restitution=.6324555320336759,
                friction_combine_mode="min", restitution_combine_mode="average"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95, .95, .95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2., 0., .11)),
    )
    contacts = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*",
                                update_period=0., history_length=4)
    # Receiving owns its logging; the inherited kicking CSV writer is unused.
    enable_csv_logging = False
    # A saved receiving config may be supplied by train/play before construction.
    task_settings: dict | None = None
