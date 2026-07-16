# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers
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
    """蹴り方向 + 目標ボール速度コマンド。

    エピソードごとにランダムな角度 θ と目標ボール速度 v をサンプリングし、
    command = [sin θ, cos θ, v] を返す。
    kick_state は command[:, :2] を蹴り方向ベクトル (cos θ, sin θ) として、
    target_kick_velocity 観測と kick_velocity_scaled 報酬は command[:, 2] を目標速度として使用する。
    """

    cfg: "KickDirectionCommandCfg"

    def _resample_command(self, env_ids: torch.Tensor):
        n = len(env_ids)
        low, high = self.cfg.ranges.heading

        # ロボットの現在ヨー角を取得し、そこからの相対オフセットとしてサンプリング
        robot_quat = self.robot.data.root_quat_w[env_ids]
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        offset = torch.empty(n, device=self.device).uniform_(low, high)
        theta = robot_yaw + offset

        speed_low, speed_high = self.cfg.target_speed_range

        self.command[env_ids, 0] = torch.sin(theta)
        self.command[env_ids, 1] = torch.cos(theta)
        self.command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(speed_low, speed_high)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "kick_dir_visualizer"):
                self.kick_dir_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
            self.kick_dir_visualizer.set_visibility(True)
        else:
            if hasattr(self, "kick_dir_visualizer"):
                self.kick_dir_visualizer.set_visibility(False)

    def _debug_vis_callback(self, _event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # command = [sin θ, cos θ, 0] → world frame angle θ
        kick_x = self.command[:, 1]  # cos θ
        kick_y = self.command[:, 0]  # sin θ
        theta = torch.atan2(kick_y, kick_x)  # = θ (world frame)

        zeros = torch.zeros_like(theta)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, theta)

        default_scale = self.kick_dir_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)

        self.kick_dir_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)


@configclass
class KickDirectionCommandCfg(UniformVelocityCommandCfg):
    """蹴り方向コマンドの設定クラス。

    ranges.heading で蹴り方向のサンプリング範囲を指定する。
    ranges.lin_vel_x/y, ang_vel_z は使用しない（0 に設定すること）。
    """

    class_type: type = KickDirectionCommand

    target_speed_range: tuple[float, float] = (1.0, 4.0)
    """目標ボール速度 [m/s] のサンプリング範囲。command[:, 2] に格納される。"""


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

        ball_pos_w = ball.data.root_pos_w[:, :3]
        robot_pos_w = robot.data.root_pos_w[:, :3]

        kick_cmd = None
        if self.cfg.kick_direction_command_name:
            kick_cmd = self._env.command_manager.get_command(self.cfg.kick_direction_command_name)

        # ボール - ロボット のワールドフレームベクトルをロボットフレームに変換
        to_ball_3d = torch.zeros_like(ball_pos_w)
        to_ball_3d[:, :2] = ball_pos_w[:, :2] - robot_pos_w[:, :2]
        to_ball_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), to_ball_3d)

        max_vel = self.cfg.max_vel
        self.vel_command_b[:, 0] = torch.clamp(to_ball_b[:, 0], -max_vel, max_vel)
        self.vel_command_b[:, 1] = torch.clamp(to_ball_b[:, 1], -max_vel, max_vel)

        # wz: ロボットのヨー角と kick_direction の角度誤差
        if kick_cmd is not None:
            kick_theta = torch.atan2(kick_cmd[:, 0], kick_cmd[:, 1])

            quat = robot.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            ang_error = kick_theta - robot_yaw
            ang_error = torch.atan2(torch.sin(ang_error), torch.cos(ang_error))
            self.vel_command_b[:, 2] = torch.clamp(ang_error, -self.cfg.max_ang_vel, self.cfg.max_ang_vel)
        else:
            self.vel_command_b[:, 2] = 0.0


@configclass
class BallFollowVelocityCommandCfg(UniformVelocityCommandCfg):
    """ボール追従速度コマンドの設定クラス。"""

    class_type: type = BallFollowVelocityCommand

    max_vel: float = 1.0
    """速度コマンドの上限 [m/s]。"""

    max_ang_vel: float = 1.0
    """角速度コマンドの上限 [rad/s]。"""

    kick_direction_command_name: str | None = None
    """角速度コマンドの参照先となる kick_direction コマンド名。None なら wz=0。"""

