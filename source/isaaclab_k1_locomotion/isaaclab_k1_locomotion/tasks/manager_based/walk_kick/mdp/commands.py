# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat

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

    def _resample_command(self, env_ids: torch.Tensor):
        n = len(env_ids)
        low, high = self.cfg.ranges.heading
        theta = torch.empty(n, device=self.device).uniform_(low, high)
        self.command[env_ids, 0] = torch.sin(theta)
        self.command[env_ids, 1] = torch.cos(theta)
        self.command[env_ids, 2] = 0.0

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

    def _is_random_phase(self) -> bool:
        if self.cfg.ball_follow_start_iteration <= 0:
            return False
        step = self._env.common_step_counter // max(self.cfg.steps_per_iteration, 1)
        return step < self.cfg.ball_follow_start_iteration

    def _resample_command(self, env_ids: torch.Tensor):
        if self._is_random_phase():
            # Phase 1: 親クラスのランダムサンプリングを使用
            super()._resample_command(env_ids)
        # Phase 2+: _update_command が毎ステップ上書きするため不要

    def _update_command(self):
        ball = self._env.scene["soccer_ball"]
        robot = self._env.scene["robot"]

        rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
        rel_pos_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)

        dist = rel_pos_b[:, :2].norm(dim=-1)
        scale = torch.clamp(dist / self.cfg.kick_approach_radius, 0.0, 1.0)

        kick_x = rel_pos_b[:, 0] - self.cfg.kick_offset_x * scale
        kick_y = rel_pos_b[:, 1] - torch.sign(rel_pos_b[:, 1]) * self.cfg.kick_lateral_offset * scale

        # 可視化用に常に更新（Phase 1でも表示できるよう）
        self._kick_pos_b = torch.stack([kick_x, kick_y, torch.zeros_like(kick_x)], dim=-1)

        if self._is_random_phase():
            return  # Phase 1: 速度コマンドはリサンプル時のランダム値をそのまま使用

        max_vel = self.cfg.max_vel
        self.vel_command_b[:, 0] = torch.clamp(kick_x, -max_vel, max_vel)
        self.vel_command_b[:, 1] = torch.clamp(kick_y, -max_vel, max_vel)

        # wz: ロボットの現在ヨー角と kick_direction の角度誤差（相対）
        if self.cfg.kick_direction_command_name:
            kick_cmd = self._env.command_manager.get_command(self.cfg.kick_direction_command_name)
            kick_theta = torch.atan2(kick_cmd[:, 0], kick_cmd[:, 1])

            quat = robot.data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            ang_error = kick_theta - robot_yaw
            ang_error = torch.atan2(torch.sin(ang_error), torch.cos(ang_error))
            self.vel_command_b[:, 2] = torch.clamp(ang_error, -self.cfg.max_ang_vel, self.cfg.max_ang_vel)
        else:
            self.vel_command_b[:, 2] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "kick_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/KickPosition",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.5, 0.0),
                            ),
                        )
                    },
                )
                self.kick_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.kick_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "kick_pos_visualizer"):
                self.kick_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, _event):
        if not self.robot.is_initialized:
            return
        if not hasattr(self, "_kick_pos_b"):
            return

        robot = self._env.scene["robot"]
        robot_pos_w = robot.data.root_pos_w[:, :3].clone()
        robot_yaw_q = yaw_quat(robot.data.root_quat_w)

        kick_pos_w = robot_pos_w + quat_rotate(robot_yaw_q, self._kick_pos_b)
        kick_pos_w[:, 2] = robot_pos_w[:, 2]

        identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        identity_quat[:, 0] = 1.0

        self.kick_pos_visualizer.visualize(kick_pos_w, identity_quat)


@configclass
class BallFollowVelocityCommandCfg(UniformVelocityCommandCfg):
    """ボール追従速度コマンドの設定クラス。"""

    class_type: type = BallFollowVelocityCommand

    max_vel: float = 1.0
    """速度コマンドの上限 [m/s]。ボール相対位置をこの値でクランプする。"""

    ball_follow_start_iteration: int = 1000
    """このiteration数からボール追従に切り替える。0以下で常にボール追従。"""

    steps_per_iteration: int = 24
    """1 iteration あたりのステップ数（PPO config の num_steps_per_env）。"""

    max_ang_vel: float = 1.0
    """角速度コマンドの上限 [rad/s]。角度誤差をこの値でクランプする。"""

    kick_offset_x: float = 0.1
    """キック位置のx方向オフセット [m]。ボールより手前に止まる距離。"""

    kick_lateral_offset: float = 0.15
    """キック位置の横方向オフセット [m]。近い方の足でキックするようにずらす量。"""

    kick_approach_radius: float = 0.4
    """このボールまでの距離以下でオフセットが線形に縮小し始める [m]。0でボール中心に収束。"""

    kick_direction_command_name: str | None = None
    """角速度コマンドの参照先となる kick_direction コマンド名。None なら wz=0。"""