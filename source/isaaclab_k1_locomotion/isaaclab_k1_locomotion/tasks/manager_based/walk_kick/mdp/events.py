from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def perturb_ball_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    distance_range: tuple[float, float] = (0.01, 0.05),
) -> None:
    """ボールをランダムな方向に distance_range [m] だけずらす。"""
    ball = env.scene[ball_cfg.name]
    n = len(env_ids)

    state = ball.data.root_state_w[env_ids].clone()

    dist = torch.empty(n, device=env.device).uniform_(*distance_range)
    angle = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * torch.pi)
    state[:, 0] += dist * torch.cos(angle)
    state[:, 1] += dist * torch.sin(angle)

    ball.write_root_state_to_sim(state, env_ids=env_ids)
