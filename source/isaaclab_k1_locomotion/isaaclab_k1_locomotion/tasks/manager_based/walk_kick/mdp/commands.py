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

from ...locomotion.mdp.commands import DiscreteVelocityCommand, DiscreteVelocityCommandCfg
from .kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


class BallFollowVelocityCommand(DiscreteVelocityCommand):
    """目標終端 G へ向かう速度コマンド。walk phase では通常の歩行コマンドに切り替わる。

    ``cfg.follow_ball`` が True (kick phase) のとき、毎ステップ速度コマンド (vx, vy, wz) を
    「G へのロボット相対位置」と「kick_direction への角度誤差」で上書きする。

    G はボールそのものではなく :mod:`.kick_state` が計算する理想キック立ち位置側の点
    (後方レイ R 上をボール側へ滑る点で、P_kick で下限クランプされる)。ボール中心へ直行
    させると walk_speed 報酬 (G へ引く) と kick_pose_overshoot 罰 (後方レイ R の左右跨ぎ)
    の両方と衝突するため、指令も G を向ける。latch 後は kick_state 側で G が P_kick に
    固定されるので、飛翔したボールを追いかけることもない。

    False (walk phase) のときは何も上書きせず、親の :class:`DiscreteVelocityCommand`
    そのもの、つまり K1FlatEnvCfg で使っている通常の歩行コマンド（離散格子からの
    ランダムサンプリング）として振る舞う。lin_vel_command / command_resampling_time_range
    カリキュラムもこのときだけ意味を持つ。
    """

    cfg: "BallFollowVelocityCommandCfg"

    def _resample_command(self, env_ids):
        if not self.cfg.follow_ball:
            # walk phase: 通常の歩行コマンドを離散格子からサンプリングする
            super()._resample_command(env_ids)
        # kick phase: 毎ステップ _update_command で上書きするためリサンプルは不要

    def _update_command(self):
        if not self.cfg.follow_ball:
            # walk phase: 親 (UniformVelocityCommand) の standing-env ゼロ化などに任せる
            super()._update_command()
            return

        robot = self._env.scene["robot"]

        # NOTE: kick_state はステップ単位でキャッシュされる。CommandManager は
        #       TerminationManager / RewardManager より後に走るので、ここで得られるのは
        #       同じステップに確定済みの状態 (再計算されない)。
        state = kick_state(
            self._env,
            r_stance=self.cfg.r_stance,
            alpha=self.cfg.alpha,
            v_thresh=self.cfg.v_thresh,
            command_name=self.cfg.kick_direction_command_name or "kick_direction",
        )

        robot_pos_w = robot.data.root_pos_w[:, :2]

        # G - ロボット のワールドフレームベクトルをロボットフレームに変換
        to_G_w = torch.zeros(self.num_envs, 3, device=self.device)
        to_G_w[:, :2] = state["G"] - robot_pos_w
        to_G_b = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), to_G_w)

        max_vel = self.cfg.max_vel
        self.vel_command_b[:, 0] = torch.clamp(to_G_b[:, 0], -max_vel, max_vel)
        self.vel_command_b[:, 1] = torch.clamp(to_G_b[:, 1], -max_vel, max_vel)

        # wz: ロボットのヨー角と kick_direction の角度誤差
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


@configclass
class BallFollowVelocityCommandCfg(DiscreteVelocityCommandCfg):
    """ボール追従速度コマンドの設定クラス。

    ``follow_ball=False`` にすると DiscreteVelocityCommandCfg（通常の歩行コマンド）として
    振る舞うので、ranges / *_resolution も引き継いで設定しておくこと。
    """

    class_type: type = BallFollowVelocityCommand

    follow_ball: bool = True
    """True (kick phase) で G 追従、False (walk phase) で通常の歩行コマンド。"""

    max_vel: float = 1.0
    """速度コマンドの上限 [m/s]。follow_ball=True のときのみ使用。"""

    max_ang_vel: float = 1.0
    """角速度コマンドの上限 [rad/s]。follow_ball=True のときのみ使用。"""

    kick_direction_command_name: str | None = None
    """角速度コマンドの参照先となる kick_direction コマンド名。None なら wz=0。"""

    # -- kick_state (G の計算) に渡すパラメータ。キック報酬側と同じ値にすること。
    r_stance: float = 0.25
    """P_kick 半径 [m]。follow_ball=True のときのみ使用。"""

    alpha: float = 0.5
    """G の追従係数。follow_ball=True のときのみ使用。"""

    v_thresh: float = 0.8
    """値 latch のトリガー速度 [m/s]。follow_ball=True のときのみ使用。"""

