# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ロボットの左右対称性 (sagittal plane mirror) を表現する関数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

_LEFT_JOINT_IDX = [0, 1, 2, 3, 4, 5]
_RIGHT_JOINT_IDX = [6, 7, 8, 9, 10, 11]
_JOINT_MIRROR_SIGN = [
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
]

_ANG_VEL_SLICE = slice(0, 3)
_PROJ_GRAVITY_SLICE = slice(3, 6)
_VEL_CMD_SLICE = slice(6, 9)
_JOINT_POS_SLICE = slice(9, 21)
_JOINT_VEL_SLICE = slice(21, 33)
_LAST_ACTION_SLICE = slice(33, 45)
_GAIT_PHASE_SLICE = slice(45, 49)
_POLICY_OBS_DIM = 49
_GAIT_PHASE_SWAP = [47, 48, 45, 46]


def _mirror_joints(joint_data: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(joint_data)
    out[..., _LEFT_JOINT_IDX] = joint_data[..., _RIGHT_JOINT_IDX]
    out[..., _RIGHT_JOINT_IDX] = joint_data[..., _LEFT_JOINT_IDX]
    sign = torch.tensor(_JOINT_MIRROR_SIGN, device=joint_data.device, dtype=joint_data.dtype)
    return out * sign


def _mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != _POLICY_OBS_DIM:
        raise ValueError(
            f"symmetry: unexpected policy obs dim {obs.shape[-1]} (expected {_POLICY_OBS_DIM})."
        )

    out = obs.clone()
    device, dtype = out.device, out.dtype
    out[:, _ANG_VEL_SLICE] *= torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype)
    out[:, _PROJ_GRAVITY_SLICE] *= torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)
    out[:, _VEL_CMD_SLICE] *= torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)
    out[:, _JOINT_POS_SLICE] = _mirror_joints(obs[:, _JOINT_POS_SLICE])
    out[:, _JOINT_VEL_SLICE] = _mirror_joints(obs[:, _JOINT_VEL_SLICE])
    out[:, _LAST_ACTION_SLICE] = _mirror_joints(obs[:, _LAST_ACTION_SLICE])
    out[:, _GAIT_PHASE_SLICE] = obs[:, _GAIT_PHASE_SWAP]
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joints(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
