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

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
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


class ExtremeVelocityCommand(UniformVelocityCommand):
    """一様サンプリングに「レンジ端の組み合わせ」サンプリングを混ぜる速度コマンド。

    最適制御などの上位プランナはコマンド範囲の上限付近を多用するが、一様サンプリング
    では「vx・vy が同時に上限付近」「その状態で大きな旋回」のような角 (corner) の
    組み合わせはほとんど出現せず、その領域で転倒しやすいポリシーになる。

    このコマンドは確率 ``cfg.extreme_prob`` で resample を「extreme モード」にする:
      - lin_vel_x / lin_vel_y の各成分を、符号ランダムで ``|v| ∈ [extreme_frac*max, max]``
        から引く (現在の ``cfg.ranges`` を参照するのでカリキュラムの範囲拡張に追従)
      - heading target を現在の機体 yaw から ``extreme_heading_err_range`` [rad] だけ
        離した方位に置き、heading 制御則 (stiffness×err を ang_vel_z 範囲で clip) が
        飽和 yaw コマンドを出す状態を作る
      - extreme に選ばれた env は standing 抽選から除外する

    ``extreme_prob=0.0`` (既定) では完全に ``UniformVelocityCommand`` と同一の挙動。
    カリキュラム (``extreme_command_curriculum``) が学習後半で prob を上げる。
    """

    cfg: "ExtremeVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        prob = float(self.cfg.extreme_prob)
        if prob <= 0.0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pick = torch.rand(ids.numel(), device=self.device) < prob
        if not pick.any():
            return
        ids = ids[pick]
        m = ids.numel()
        frac = float(self.cfg.extreme_frac)

        # lin_vel_x / lin_vel_y: 符号をランダムに選び、その側の端 [frac*|bound|, |bound|] から引く
        for col, rng in ((0, self.cfg.ranges.lin_vel_x), (1, self.cfg.ranges.lin_vel_y)):
            lo, hi = float(rng[0]), float(rng[1])
            pick_pos = torch.rand(m, device=self.device) < 0.5
            bound = torch.where(
                pick_pos,
                torch.full((m,), hi, device=self.device),
                torch.full((m,), lo, device=self.device),
            )
            u = torch.rand(m, device=self.device)
            self.vel_command_b[ids, col] = bound * (frac + (1.0 - frac) * u)

        # heading: 現在 yaw から大きく外した目標を与え、飽和 yaw コマンドを作る。
        # (heading_command=True の場合、実際の yaw コマンドは毎ステップ
        #  clip(stiffness * heading_err, ang_vel_z range) で再計算されるため、
        #  ang_vel_z を直接書いてもすぐ上書きされる。heading 側を動かすのが正)
        if self.cfg.heading_command:
            err_lo, err_hi = self.cfg.extreme_heading_err_range
            err_mag = torch.empty(m, device=self.device).uniform_(float(err_lo), float(err_hi))
            err_sign = torch.where(
                torch.rand(m, device=self.device) < 0.5,
                torch.ones(m, device=self.device),
                -torch.ones(m, device=self.device),
            )
            target = self.robot.data.heading_w[ids] + err_sign * err_mag
            self.heading_target[ids] = math_utils.wrap_to_pi(target)
            self.is_heading_env[ids] = True

        # extreme env は立ち止まり抽選から除外 (コマンドが 0 に上書きされるのを防ぐ)
        self.is_standing_env[ids] = False


@configclass
class ExtremeVelocityCommandCfg(UniformVelocityCommandCfg):
    """`ExtremeVelocityCommand` の設定クラス。"""

    class_type: type = ExtremeVelocityCommand

    extreme_prob: float = 0.0
    """resample を extreme モードにする確率 [0,1]。0 で通常の一様サンプリングと同一。"""

    extreme_frac: float = 0.7
    """extreme モードで引く線速度成分の大きさの下限 (レンジ端に対する割合)。"""

    extreme_heading_err_range: tuple[float, float] = (2.0, math.pi)
    """extreme モードで heading target を現在 yaw から離す角度 [rad] の範囲。
    stiffness 0.5・ang_vel_z 上限 1.0 の場合、誤差 2.0 rad 以上で yaw コマンドが飽和する。"""


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
        # ボールの参照 (矢印の可視化起点用)
        self.ball = env.scene[cfg.ball_name]
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

    """
    可視化 (debug_vis=True のとき、ロボット頭上にキック方向の矢印を表示)。
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # マーカーの表示/非表示を切り替える。
        if debug_vis:
            # 初回のみマーカーを生成。
            if not hasattr(self, "kick_dir_visualizer"):
                self.kick_dir_visualizer = VisualizationMarkers(self.cfg.goal_dir_visualizer_cfg)
            self.kick_dir_visualizer.set_visibility(True)
        else:
            if hasattr(self, "kick_dir_visualizer"):
                self.kick_dir_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # ボールが未初期化なら何もしない (de-init 時に data へアクセスできないため)。
        if not self.ball.is_initialized:
            return
        # 矢印の起点: ボールの少し上。
        arrow_pos_w = self.ball.data.root_pos_w.clone()
        arrow_pos_w[:, 2] += 0.3
        # ワールド座標 xy 単位ベクトルを矢印の scale / quaternion に変換。
        arrow_scale, arrow_quat = self._resolve_xy_dir_to_arrow(self.kick_dir_w)
        self.kick_dir_visualizer.visualize(arrow_pos_w, arrow_quat, arrow_scale)

    def _resolve_xy_dir_to_arrow(self, xy_dir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ワールド座標系の xy 方向ベクトルを矢印の (scale, quaternion) に変換する。"""
        # マーカーのデフォルト scale。
        default_scale = self.kick_dir_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_dir.shape[0], 1)
        # 方位角 (ワールド x 軸から) を計算。kick_dir_w は単位ベクトルなので向きのみ使う。
        heading_angle = torch.atan2(xy_dir[:, 1], xy_dir[:, 0])
        zeros = torch.zeros_like(heading_angle)
        # kick_dir_w はワールド座標系なので base quaternion を掛ける必要はない。
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat


@configclass
class KickDirectionCommandCfg(CommandTermCfg):
    """`KickDirectionCommand` の設定。"""

    class_type: type = KickDirectionCommand

    asset_name: str = MISSING
    """メトリック計算に使うロボット asset の名前。"""

    ball_name: str = "soccer_ball"
    """矢印の可視化起点に使うボール asset の名前。"""

    angle_range: tuple[float, float] = (-math.pi, math.pi)
    """サンプリングされる角度 θ (rad) のレンジ。ワールド座標 x 軸からの方位角。"""

    goal_dir_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/kick_direction"
    )
    """キック方向を示す矢印マーカーの設定 (デフォルトは緑の矢印)。"""

    goal_dir_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
