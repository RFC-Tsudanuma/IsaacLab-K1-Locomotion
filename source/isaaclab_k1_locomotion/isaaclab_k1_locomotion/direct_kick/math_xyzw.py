"""Quaternion boundary helpers using the source task's xyzw convention."""
import torch


def quat_rotate(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    return v * (2 * w.square() - 1) + 2 * w * torch.cross(xyz, v, dim=-1) + 2 * xyz * (xyz * v).sum(-1, keepdim=True)


def quat_rotate_inverse(q, v):
    conjugate = torch.cat((-q[..., :3], q[..., 3:4]), -1)
    return quat_rotate(conjugate, v)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy), -1)


def get_euler_xyz(q):
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), w*w-x*x-y*y+z*z)
    sinp = 2*(w*y-z*x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * torch.pi / 2, torch.asin(sinp))
    yaw = torch.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z)
    return roll % (2*torch.pi), pitch % (2*torch.pi), yaw % (2*torch.pi)


def get_axis_params(value, axis):
    values = [0., 0., 0.]
    values[axis] = value
    return values


def to_torch(value, device, dtype=torch.float, requires_grad=False):
    return torch.tensor(value, device=device, dtype=dtype, requires_grad=requires_grad)


def torch_rand_float(lower, upper, shape, device):
    return (upper-lower)*torch.rand(*shape, device=device)+lower
