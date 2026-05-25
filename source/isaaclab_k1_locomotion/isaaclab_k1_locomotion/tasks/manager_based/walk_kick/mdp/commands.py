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

    def _resample_command(self, _env_ids):
        # 毎ステップ _update_command で上書きするためリサンプルは不要
        pass

    def _update_command(self):
        ball = self._env.scene["soccer_ball"]
        robot = self._env.scene["robot"]

        ball_pos_w = ball.data.root_pos_w[:, :3]
        robot_pos_w = robot.data.root_pos_w[:, :3]

        # キック方向ベクトル (world frame)
        kick_dir_w = torch.zeros_like(ball_pos_w)
        kick_cmd = None
        if self.cfg.kick_direction_command_name:
            kick_cmd = self._env.command_manager.get_command(self.cfg.kick_direction_command_name)
            kick_dir_w[:, 0] = kick_cmd[:, 1]  # cos θ (world x)
            kick_dir_w[:, 1] = kick_cmd[:, 0]  # sin θ (world y)

        # robot → ball ベクトル
        to_ball = ball_pos_w[:, :2] - robot_pos_w[:, :2]
        dist = to_ball.norm(dim=-1).clamp(min=1e-6)
        to_ball_dir = to_ball / dist.unsqueeze(-1)

        # キック方向との角度一致度 [0, 1]
        alignment = (to_ball_dir * kick_dir_w[:, :2]).sum(dim=-1).clamp(min=0.0)

        # ボールまでの距離に応じて動的オフセットを変化させる
        # dist >= decay_start_dist → approach_offset（後方ターゲット）
        # dist = 0               → -overshoot_offset（向こう側ターゲット）
        t = (dist / self.cfg.decay_start_dist).clamp(0.0, 1.0)
        tp = t.pow(self.cfg.decay_exponent)
        dynamic_offset = tp * self.cfg.approach_offset + (1.0 - tp) * (-self.cfg.overshoot_offset)

        # 動的ターゲット
        dynamic_target = ball_pos_w[:, :2] - dynamic_offset.unsqueeze(-1) * kick_dir_w[:, :2]
        att_raw = dynamic_target - robot_pos_w[:, :2]
        att_dist = att_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        att_dir = att_raw / att_dist
        att_speed = att_dist.squeeze(-1).clamp(min=self.cfg.min_att_speed, max=self.cfg.max_vel)
        att_vec = att_dir * att_speed.unsqueeze(-1)

        # 斥力: アプローチ方向にだけ穴をあける
        hole_factor = 1.0 - alignment.pow(self.cfg.hole_sharpness)  # (N,)

        in_range = dist < self.cfg.repulsion_radius
        rep_mag = hole_factor * torch.where(
            in_range,
            self.cfg.repulsion_gain * (1.0 / dist - 1.0 / self.cfg.repulsion_radius) / dist.pow(2),
            torch.zeros_like(dist),
        )
        rep_vec = -(to_ball_dir) * rep_mag.unsqueeze(-1)

        F_total = att_vec + rep_vec
        F_total_3d = torch.zeros_like(ball_pos_w)
        F_total_3d[:, :2] = F_total
        F_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), F_total_3d)

        max_vel = self.cfg.max_vel
        self.vel_command_b[:, 0] = torch.clamp(F_b[:, 0], -max_vel, max_vel)
        self.vel_command_b[:, 1] = torch.clamp(F_b[:, 1], -max_vel, max_vel)

        # wz: ロボットの現在ヨー角と kick_direction の角度誤差（相対）
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
    """速度コマンドの上限 [m/s]。ボール相対位置をこの値でクランプする。"""

    min_att_speed: float = 0.6
    """引力の最小速度 [m/s]。ターゲットに近づいても最低限この速度で引き付ける。"""

    max_ang_vel: float = 1.0
    """角速度コマンドの上限 [rad/s]。角度誤差をこの値でクランプする。"""

    kick_direction_command_name: str | None = None
    """角速度コマンドの参照先となる kick_direction コマンド名。None なら wz=0。"""

    approach_offset: float = 0.5
    """ボールから蹴る向きの反対側へのオフセット距離 [m]。ロボットはこの位置を目標とする。"""

    repulsion_radius: float = 0.3
    """ボールからの斥力が発生する距離 [m]。ボール半径(0.11m)より大きく設定する。"""

    repulsion_gain: float = 0.5
    """斥力の強さ。大きいほど強くボールを避ける。"""

    hole_sharpness: float = 2.0
    """アプローチ方向の穴の鋭さ。大きいほど穴が狭くなる。"""

    decay_start_dist: float = 0.5 * math.sqrt(2)
    """ボールまでの距離がこれ以下になったら動的オフセット変化を開始する [m]。"""

    overshoot_offset: float = 0.0
    """ボール到達時のオーバーシュート距離 [m]。ターゲットがボールの向こう側へ移る。"""

    decay_exponent: float = 2.0
    """動的オフセット縮小の冪乗。1=線形、2以上で近距離側が急峻になる。"""