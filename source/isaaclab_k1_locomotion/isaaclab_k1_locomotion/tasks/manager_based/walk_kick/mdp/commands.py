# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DiscreteVelocityCommand(UniformVelocityCommand):
    """lin_vel_x, lin_vel_y, ang_vel_z をすべて離散的にサンプリングする。"""

    cfg: "DiscreteVelocityCommandCfg"

    def _sample_discrete(self, n: int, high_val: float, low_max: float) -> torch.Tensor:
        use_high = torch.rand(n, device=self.device) < self.cfg.high_prob
        low_vel = torch.rand(n, device=self.device) * low_max
        high_vel = torch.full((n,), high_val, device=self.device)
        vel = torch.where(use_high, high_vel, low_vel)
        sign = torch.sign(torch.randn(n, device=self.device))
        return vel * sign

    def _resample(self, env_ids: torch.Tensor):
        super()._resample(env_ids)
        n = len(env_ids)
        self.command[env_ids, 0] = self._sample_discrete(n, self.cfg.high_vel, self.cfg.low_vel_max)
        self.command[env_ids, 1] = self._sample_discrete(n, self.cfg.high_vel, self.cfg.low_vel_max)
        self.command[env_ids, 2] = self._sample_discrete(n, self.cfg.high_ang_vel, self.cfg.low_ang_vel_max)


@configclass
class DiscreteVelocityCommandCfg(UniformVelocityCommandCfg):
    """離散速度コマンド（0~low_vel_max と high_vel のみ）の設定クラス。"""

    class_type: type = DiscreteVelocityCommand

    high_vel: float = 1.0
    low_vel_max: float = 0.2
    high_ang_vel: float = 1.0
    low_ang_vel_max: float = 0.2
    high_prob: float = 0.5