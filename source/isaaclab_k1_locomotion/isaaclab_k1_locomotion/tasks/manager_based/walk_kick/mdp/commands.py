# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

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


class KickDirectionCommand(UniformVelocityCommand):
    """蹴り方向コマンド。

    エピソードごとにランダムな角度 θ をサンプリングし、
    command = [sin θ, cos θ, 0] を返す。
    kick_ball_velocity 報酬は command[:, :2] を方向ベクトルとして使用する。
    """

    cfg: "KickDirectionCommandCfg"

    def _resample(self, env_ids: torch.Tensor):
        n = len(env_ids)
        low, high = self.cfg.ranges.heading
        theta = torch.empty(n, device=self.device).uniform_(low, high)
        self.command[env_ids, 0] = torch.sin(theta)
        self.command[env_ids, 1] = torch.cos(theta)
        self.command[env_ids, 2] = 0.0


@configclass
class KickDirectionCommandCfg(UniformVelocityCommandCfg):
    """蹴り方向コマンドの設定クラス。

    ranges.heading でサンプリング範囲を指定する。
    ranges.lin_vel_x/y, ang_vel_z は使用しない（0 に設定すること）。
    """

    class_type: type = KickDirectionCommand


class BallFollowVelocityCommand(UniformVelocityCommand):
    """ボール追従速度コマンド。

    毎ステップ、速度コマンド (vx, vy, wz) をロボットフレームでの
    ボール相対位置 (x, y) と wz=0 に更新する。
    velocity tracking 報酬と組み合わせることでボール追従を実現する。
    """

    cfg: "BallFollowVelocityCommandCfg"

    def _resample_command(self, _env_ids):
        # 毎ステップ _update_command で上書きするためリサンプルは不要
        pass

    def _update_command(self):
        ball = self._env.scene["soccer_ball"]
        robot = self._env.scene["robot"]

        rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
        rel_pos_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)

        max_vel = self.cfg.max_vel
        self.vel_command_b[:, 0] = torch.clamp(rel_pos_b[:, 0], -max_vel, max_vel)
        self.vel_command_b[:, 1] = torch.clamp(rel_pos_b[:, 1], -max_vel, max_vel)
        self.vel_command_b[:, 2] = 0.0


@configclass
class BallFollowVelocityCommandCfg(UniformVelocityCommandCfg):
    """ボール追従速度コマンドの設定クラス。"""

    class_type: type = BallFollowVelocityCommand

    max_vel: float = 1.0
    """速度コマンドの上限 [m/s]。ボール相対位置をこの値でクランプする。"""