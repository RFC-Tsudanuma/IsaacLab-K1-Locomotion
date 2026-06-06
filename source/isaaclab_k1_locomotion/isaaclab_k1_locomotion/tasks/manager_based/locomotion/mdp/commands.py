# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DiscreteVelocityCommand(UniformVelocityCommand):
    """lin_vel_x / lin_vel_y / ang_vel_z を一様離散格子からサンプリングする速度コマンド。

    各軸について ``cfg.ranges`` の ``(low, high)`` と ``*_resolution`` から
    格子点 ``{low, low+r, low+2r, ..., high}`` を生成し、その中から一様に選ぶ。
    resolution が ``None`` または非正の場合は連続一様サンプリングにフォールバックする。
    """

    cfg: "DiscreteVelocityCommandCfg"

    def _sample_axis(self, n: int, vel_range: tuple[float, float], resolution: float | None) -> torch.Tensor:
        low, high = float(vel_range[0]), float(vel_range[1])
        if resolution is None or resolution <= 0.0:
            return torch.empty(n, device=self.device).uniform_(low, high)
        if high <= low:
            return torch.full((n,), low, device=self.device)
        num_bins = int(round((high - low) / resolution)) + 1
        if num_bins <= 1:
            return torch.full((n,), low, device=self.device)
        idx = torch.randint(0, num_bins, (n,), device=self.device)
        values = low + idx.to(torch.float32) * resolution
        return values.clamp_(low, high)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        r = torch.empty(n, device=self.device)
        self.vel_command_b[env_ids, 0] = self._sample_axis(n, self.cfg.ranges.lin_vel_x, self.cfg.lin_vel_x_resolution)
        self.vel_command_b[env_ids, 1] = self._sample_axis(n, self.cfg.ranges.lin_vel_y, self.cfg.lin_vel_y_resolution)
        self.vel_command_b[env_ids, 2] = self._sample_axis(n, self.cfg.ranges.ang_vel_z, self.cfg.ang_vel_z_resolution)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs


@configclass
class DiscreteVelocityCommandCfg(UniformVelocityCommandCfg):
    """離散速度コマンド（軸ごとの resolution で格子化）の設定クラス。"""

    class_type: type = DiscreteVelocityCommand

    lin_vel_x_resolution: float | None = None
    lin_vel_y_resolution: float | None = None
    ang_vel_z_resolution: float | None = None


class KickDirectionCommand(CommandTerm):
    """ワールド座標系で定義されたキック方向 (xy 単位ベクトル) を返すコマンド。

    各環境に対して `cfg.angle_range` から角度 θ を一様サンプリングし、
    `(cos θ, sin θ)` をワールド座標系のキック方向として保持する。
    """

    cfg: "KickDirectionCommandCfg"

    def __init__(self, cfg: "KickDirectionCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # ロボットの参照 (メトリック計算用)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # ワールド座標系の単位ベクトル (num_envs, 2)
        self.kick_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.kick_dir_w[:, 0] = 1.0  # 初期は +x
        # ヒストリ角度 (メトリック用)
        self.kick_angle_w = torch.zeros(self.num_envs, device=self.device)
        # メトリック
        self.metrics["angle_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 2) のワールド座標 xy 単位ベクトル。"""
        return self.kick_dir_w

    def _update_metrics(self):
        # ロボットの yaw を角度として取り出す: heading_w は world frame の x 軸からの yaw 角
        heading_w = self.robot.data.heading_w
        # 角度差をラップ
        diff = self.kick_angle_w - heading_w
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        self.metrics["angle_error"] = torch.abs(diff)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        low, high = self.cfg.angle_range
        angles = torch.empty(n, device=self.device).uniform_(float(low), float(high))
        self.kick_angle_w[env_ids] = angles
        self.kick_dir_w[env_ids, 0] = torch.cos(angles)
        self.kick_dir_w[env_ids, 1] = torch.sin(angles)

    def _update_command(self):
        # ワールド座標系定義なので、毎ステップの再計算は不要。
        pass


@configclass
class KickDirectionCommandCfg(CommandTermCfg):
    """`KickDirectionCommand` の設定。"""

    class_type: type = KickDirectionCommand

    asset_name: str = MISSING
    """メトリック計算に使うロボット asset の名前。"""

    angle_range: tuple[float, float] = (-math.pi, math.pi)
    """サンプリングされる角度 θ (rad) のレンジ。ワールド座標 x 軸からの方位角。"""
