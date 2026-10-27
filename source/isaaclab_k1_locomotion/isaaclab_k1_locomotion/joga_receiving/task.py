"""Tensor-only receiving equations and event state from Joga Table II.

    Quaternions are xyzw. Unspecified event/reset details are explicit config
    choices, not claimed to be the original unpublished implementation.
"""
import math
import torch

from ..direct_kick.math_xyzw import quat_rotate, quat_rotate_inverse


def sample_ground_pass(count, stage, device):
    """Launch toward a sampled crossing point (x=0,y=offset) in robot yaw frame."""
    distance = torch.empty(count, device=device).uniform_(*stage["distance"])
    bearing = torch.empty(count, device=device).uniform_(
        -math.radians(stage["bearing_deg"]), math.radians(stage["bearing_deg"]))
    offset = torch.empty(count, device=device).uniform_(-stage["offset"], stage["offset"])
    speed = torch.empty(count, device=device).uniform_(*stage["speed"])
    position = torch.stack((distance * bearing.cos(), distance * bearing.sin()), dim=-1)
    crossing = torch.stack((torch.zeros_like(offset), offset), dim=-1)
    direction = crossing - position
    velocity = speed[:, None] * direction / direction.norm(dim=-1, keepdim=True)
    return position, velocity, speed, offset


def torque_speed_limit(target, position, velocity, stiffness, damping, stall_torque, no_load_speed):
    """Equation (9): clip desired angles so the PD torque lies in its envelope."""
    available = stall_torque * (1. - (velocity / no_load_speed).abs()).clamp_min(0.)
    center = position + damping * velocity / stiffness
    limited_target = torch.maximum(torch.minimum(target, center + available / stiffness),
                                   center - available / stiffness)
    return limited_target, stiffness * (limited_target - position) - damping * velocity


def angular_momentum(mass, com_position, com_velocity, link_quat, inertia):
    """World momentum about the total COM; inertia is in each link's actor frame."""
    center = (mass[..., None] * com_position).sum(1) / mass.sum(1, keepdim=True)
    center_velocity = (mass[..., None] * com_velocity[..., :3]).sum(1) / mass.sum(1, keepdim=True)
    orbital = torch.cross(com_position - center[:, None],
                          mass[..., None] * (com_velocity[..., :3] - center_velocity[:, None]), dim=-1)
    omega_local = quat_rotate_inverse(link_quat.reshape(-1, 4),
                                      com_velocity[..., 3:].reshape(-1, 3)).reshape_as(com_position)
    spin_local = (inertia @ omega_local[..., None]).squeeze(-1)
    spin = quat_rotate(link_quat.reshape(-1, 4), spin_local.reshape(-1, 3)).reshape_as(com_position)
    return (orbital + spin).sum(1)


class ReceivingTask:
    def __init__(self, cfg, count, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dt = cfg["physics_dt"] * cfg["decimation"]
        self.touch_count = torch.zeros(count, device=device, dtype=torch.long)
        self.was_contact = torch.zeros(count, device=device, dtype=torch.bool)
        self.side_seen = torch.zeros_like(self.was_contact)
        self.settle_time = torch.zeros(count, device=device)
        self.blind_steps = torch.zeros_like(self.touch_count)
        self.previous_speed = torch.zeros_like(self.settle_time)
        self.target_settle = torch.zeros_like(self.settle_time)
        self.side_threshold = torch.zeros_like(self.settle_time)

    def reset(self, ids, stage, initial_speed):
        self.touch_count[ids] = 0
        self.was_contact[ids] = False
        self.side_seen[ids] = False
        self.settle_time[ids] = 0.
        self.blind_steps[ids] = 0
        self.previous_speed[ids] = initial_speed
        self.target_settle[ids] = stage["settle_s"]
        self.side_threshold[ids] = stage["side_threshold"]

    def evaluate(self, state):
        """Advance event state once per policy step; return weighted rewards/dones."""
        c = self.cfg["completion"]
        ball = state["ball_local"]
        velocity = state["ball_velocity"]
        speed = velocity.norm(dim=-1)
        speed_xy = velocity[:, :2].norm(dim=-1)
        contact_feet = state["foot_ball_force"] > c["foot_contact_threshold_n"]
        contact = contact_feet.any(-1)
        new_touch = contact & ~self.was_contact
        first_touch = new_touch & (self.touch_count == 0)
        self.touch_count += new_touch.long()
        self.was_contact.copy_(contact)
        displacement = state["ball_position"][:, None] - state["foot_position"]
        foot_to_ball = quat_rotate_inverse(state["foot_quat"].reshape(-1, 4),
                                           displacement.reshape(-1, 3)).reshape_as(displacement)
        unit = foot_to_ball / foot_to_ball.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
        side = unit[..., 1].abs()
        self.side_seen |= (contact_feet & (side >= self.side_threshold[:, None])).any(-1)
        zone_distance = (ball[:, :2] - ball.new_tensor([.2, 0.])).norm(dim=-1)
        on_ground = state["ball_ground_height"] <= self.cfg["ball"]["radius"] + c["ground_tolerance_m"]
        zone = (zone_distance <= .2) & (ball[:, 0] > 0.) & on_ground
        trapped = self.side_seen & zone & (speed < c["settle_speed_mps"])
        self.settle_time = torch.where(trapped, self.settle_time + self.dt, 0.)
        self.blind_steps = torch.where(state["visible"], 0, self.blind_steps + 1)
        blind = self.blind_steps >= 2
        distances = displacement.norm(dim=-1)
        distance, nearest = distances.min(-1)
        foot_lateral = state["foot_local"][..., 1].gather(1, nearest[:, None]).squeeze(1)
        same_side = torch.where(foot_lateral * ball[:, 1] >= 0., 1., -1.)
        engaged = (ball[:, 0] > 0.) & (ball[:, :2].norm(dim=-1) < 3.)
        controlled = (distance < .3) & (speed < .5)
        base_speed = state["base_velocity"][:, :2].norm(dim=-1)
        deviation = state["joint_position"] - state["default_joint_position"]
        touching_side = (side * contact_feet).amax(-1)
        touch_weight = torch.pow(2., 1. - self.touch_count.float())
        foot_above = ((displacement[..., :2].norm(dim=-1) < c["foot_above_ball_xy_m"])
                      & (state["foot_position"][..., 2] > state["ball_position"][:, None, 2]))
        x_axis = torch.zeros_like(displacement)
        x_axis[..., 0] = 1.
        foot_forward = quat_rotate(state["foot_quat"].reshape(-1, 4),
                                    x_axis.reshape(-1, 3)).reshape_as(displacement)
        sole_direction = (foot_forward[..., 2].clamp_min(0.) * contact_feet).amax(-1)
        ankle_deviation = deviation[:, state["ankle_pitch_indices"]].abs()
        terms = {
            "ball_in_zone": torch.exp(-zone_distance.square()),
            "face_ball": torch.exp(-2. * (1. - ball[:, 0] / ball[:, :2].norm(dim=-1).clamp_min(1.e-8))),
            "ball_trapped": self.side_seen * zone * torch.exp(-4. * speed.square()),
            "settle_progress": (self.settle_time / self.target_settle).clamp(max=1.),
            "side_first_touch": first_touch * touching_side,
            "velocity_kill": contact * touch_weight * (self.previous_speed - speed_xy).clamp_min(0.),
            "near_foot": contact * same_side * (ball[:, 1].abs() - .08).clamp_min(0.),
            "approach_loose_ball": engaged * ~controlled * torch.exp(-2. * distance),
            "stand_still": (~engaged) * torch.exp(-4. * base_speed.square())
                           + blind * torch.exp(-2. * base_speed - state["base_angular_velocity"][:, 2].abs()),
            "upright": torch.exp(-state["gravity"][:, :2].square().sum(-1) / .2),
            "pose": torch.exp(-(deviation / c["pose_sigma_rad"]).square().mean(-1)),
            "joint_limits": ((state["joint_lower"] - state["joint_position"]).clamp_min(0.)
                              + (state["joint_position"] - state["joint_upper"]).clamp_min(0.)).sum(-1),
            "self_collision": (state["self_contact_force"] > 10.).sum(-1).float(),
            "leg_ball": (state["leg_ball_force"] > 5.).sum(-1).float(),
            "action_rate": (state["action"] - state["previous_action"]).square().sum(-1),
            "torso_angular_velocity": state["base_angular_velocity"][:, :2].square().sum(-1),
            "angular_momentum": state["angular_momentum"].square().sum(-1),
            "pelvis_height": (c["pelvis_height_m"] - state["base_height"]).clamp_min(0.),
            "foot_on_ball": foot_above.sum(-1).float(),
            "sole_contact": contact * (2. * sole_direction + 4. * (ankle_deviation - .1).clamp_min(0.).sum(-1)),
            "move_when_blind": blind * (base_speed.square() + .25 * state["base_angular_velocity"][:, 2].square()
                                         + .5 * deviation.square().sum(-1)),
        }
        self.previous_speed.copy_(speed_xy)
        weighted = {name: value * self.cfg["reward_weights"][name] for name, value in terms.items()}
        if c["reward_scale_by_dt"]:
            weighted = {name: value * self.dt for name, value in weighted.items()}
        reward = torch.stack(list(weighted.values())).sum(0)
        success = self.settle_time >= self.target_settle - 1.e-6
        fall = ((state["gravity"][:, 2] > c["fall_gravity_z"])
                | (state["base_height"] < c["fall_height_m"]))
        return reward, weighted, success & ~fall, fall
