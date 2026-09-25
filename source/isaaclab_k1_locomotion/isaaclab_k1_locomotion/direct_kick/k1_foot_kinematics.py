# Ported from rl_humanoid_htwk 3af2acc97f1081b4cbfd9556efd39408dea92cc6: utils/k1_foot_kinematics.py
"""Batched K1 leg forward kinematics used for contact-aware resets."""

import torch


def _rotation_x(angle):
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation = torch.zeros(*angle.shape, 3, 3, device=angle.device, dtype=angle.dtype)
    rotation[..., 0, 0] = 1.0
    rotation[..., 1, 1] = cosine
    rotation[..., 1, 2] = -sine
    rotation[..., 2, 1] = sine
    rotation[..., 2, 2] = cosine
    return rotation


def _rotation_y(angle):
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation = torch.zeros(*angle.shape, 3, 3, device=angle.device, dtype=angle.dtype)
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 2] = sine
    rotation[..., 1, 1] = 1.0
    rotation[..., 2, 0] = -sine
    rotation[..., 2, 2] = cosine
    return rotation


def _rotation_z(angle):
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotation = torch.zeros(*angle.shape, 3, 3, device=angle.device, dtype=angle.dtype)
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 1] = -sine
    rotation[..., 1, 0] = sine
    rotation[..., 1, 1] = cosine
    rotation[..., 2, 2] = 1.0
    return rotation


def _rotate(rotation, vector):
    return torch.matmul(rotation, vector.unsqueeze(-1)).squeeze(-1)


def k1_foot_contact_points_root(
    dof_pos,
    leg_joint_indices,
    foot_contact_points,
):
    """Return both feet's contact points in the Trunk frame.

    ``leg_joint_indices`` is ordered as left/right x
    [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll].
    The fixed translations match ``resources/K1/K1_locomotion.urdf``.
    """
    if dof_pos.ndim != 2:
        raise ValueError("dof_pos must have shape [num_envs, num_dofs]")
    if leg_joint_indices.shape != (2, 6):
        raise ValueError("leg_joint_indices must have shape [2, 6]")
    if foot_contact_points.ndim != 2 or foot_contact_points.shape[1] != 3:
        raise ValueError("foot_contact_points must have shape [num_points, 3]")

    angles = dof_pos[:, leg_joint_indices]
    num_envs = dof_pos.shape[0]
    dtype = dof_pos.dtype
    device = dof_pos.device

    lateral_sign = torch.tensor(
        [1.0, -1.0],
        device=device,
        dtype=dtype,
    )
    hip_pitch_origin = torch.zeros(2, 3, device=device, dtype=dtype)
    hip_pitch_origin[:, 1] = 0.096 * lateral_sign
    hip_pitch_origin[:, 2] = -0.077
    hip_roll_origin = torch.tensor(
        [0.0, 0.0, -0.026],
        device=device,
        dtype=dtype,
    ).expand(2, -1)
    hip_yaw_origin = torch.tensor(
        [0.012, 0.0, -0.0485],
        device=device,
        dtype=dtype,
    ).expand(2, -1)
    knee_origin = torch.tensor(
        [-0.014, 0.0, -0.117],
        device=device,
        dtype=dtype,
    ).expand(2, -1)
    ankle_origin = torch.zeros(2, 3, device=device, dtype=dtype)
    ankle_origin[:, 0] = 0.00019706
    ankle_origin[:, 1] = 0.0002 * lateral_sign
    ankle_origin[:, 2] = -0.24519

    position = hip_pitch_origin.unsqueeze(0).expand(num_envs, -1, -1).clone()
    rotation = _rotation_y(angles[..., 0])
    position += _rotate(rotation, hip_roll_origin)
    rotation = torch.matmul(rotation, _rotation_x(angles[..., 1]))
    position += _rotate(rotation, hip_yaw_origin)
    rotation = torch.matmul(rotation, _rotation_z(angles[..., 2]))
    position += _rotate(rotation, knee_origin)
    rotation = torch.matmul(rotation, _rotation_y(angles[..., 3]))
    position += _rotate(rotation, ankle_origin)
    rotation = torch.matmul(rotation, _rotation_y(angles[..., 4]))
    rotation = torch.matmul(rotation, _rotation_x(angles[..., 5]))

    points = foot_contact_points.to(device=device, dtype=dtype)
    rotated_points = torch.matmul(
        rotation.unsqueeze(2),
        points.view(1, 1, -1, 3, 1),
    ).squeeze(-1)
    return position.unsqueeze(2) + rotated_points
