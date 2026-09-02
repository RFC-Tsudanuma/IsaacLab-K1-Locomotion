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
        prev = self.vel_command_b[env_ids].clone()
        self.vel_command_b[env_ids, 0] = self._sample_axis(n, self.cfg.ranges.lin_vel_x, self.cfg.lin_vel_x_resolution)
        self.vel_command_b[env_ids, 1] = self._sample_axis(n, self.cfg.ranges.lin_vel_y, self.cfg.lin_vel_y_resolution)
        self.vel_command_b[env_ids, 2] = self._sample_axis(n, self.cfg.ranges.ang_vel_z, self.cfg.ang_vel_z_resolution)
        self._apply_pure_axis(env_ids, n)
        _reversed = self._apply_reversal(env_ids, n, prev)
        self._apply_stop(env_ids, n, prev, _reversed)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _apply_pure_axis(self, env_ids: Sequence[int], n: int) -> None:
        """``pure_axis_prob`` の確率で **1 軸だけ残して他をゼロ**にする。

        ★ 2026-08-20 追加。既定 0.0 なので、指定しない限り従来の挙動と完全に同じ。

        なぜ要るか (実測):
            3 軸を独立に一様サンプルすると、「ほぼ純粋な後退」(vx ≤ -0.5 かつ |vy| ≤ 0.05)
            は resolution 0.05 のとき **全指令の約 1.5%** しか出ない (純粋前進も同じ)。
            ところが **実機はコントローラで純粋方向の指令を出す**。学習分布と
            デプロイ分布が食い違っており、実機で「後退だけ人が支えないと転倒する」
            症状が出た (2026-08-20 のデプロイ)。

            なお vx の符号は完全に対称にサンプルされているので、「後退の分布が
            足りない」わけではない。薄いのは **符号ではなく「純粋方向」** である。

        ☠ standing env (指令ゼロ) とは別物。あちらは全軸ゼロ、こちらは 1 軸だけ生かす。
        """
        p = float(getattr(self.cfg, "pure_axis_prob", 0.0) or 0.0)
        if p <= 0.0:
            return
        pure = torch.rand(n, device=self.device) < p
        if not bool(pure.any()):
            return
        # 残す軸を 0/1/2 から一様に選ぶ (vx / vy / wz)
        keep = torch.randint(0, 3, (n,), device=self.device)
        axis_idx = torch.arange(3, device=self.device).unsqueeze(0)          # (1, 3)
        mask = (axis_idx == keep.unsqueeze(1)) | ~pure.unsqueeze(1)          # (n, 3)
        self.vel_command_b[env_ids] = self.vel_command_b[env_ids] * mask.float()

    def _apply_stop(self, env_ids: Sequence[int], n: int, prev: torch.Tensor,
                    exclude: torch.Tensor) -> None:
        """``stop_prob`` の確率で新しい指令を **ゼロ** に差し替える (歩行→停止の遷移)。

        Args:
            prev: 直前の指令 (n, 3)。再サンプル前に控えたもの。
            exclude: 既に反転が当たった env のマスク (n,)。二重適用を避ける。
        """
        p = float(getattr(self.cfg, "stop_prob", 0.0) or 0.0)
        if p <= 0.0:
            return
        min_speed = float(getattr(self.cfg, "stop_min_speed", 0.3))
        prev_speed = torch.norm(prev[:, :2], dim=1)
        do = (torch.rand(n, device=self.device) < p) & (prev_speed > min_speed) & ~exclude
        if not bool(do.any()):
            return
        ids = torch.as_tensor(env_ids, device=self.device)[do]
        self.vel_command_b[ids] = 0.0
        # ☠ standing 抽選で上書きされないよう、この env は standing 扱いにする。
        #   (is_standing_env は _update_command でゼロ埋めに使われる)
        self.is_standing_env[ids] = True

    def _apply_reversal(self, env_ids: Sequence[int], n: int, prev: torch.Tensor) -> torch.Tensor:
        """``reversal_prob`` の確率で新しい指令を **直前の指令の符号反転** に差し替える。

        ★ 2026-08-21 追加。既定 0.0 なので、指定しない限り従来の挙動と完全に同じ。

        なぜ要るか:
            ゴールキーパーで実際に効くのは「静止 → 全開」より **左右への振り直し**
            (+1.3 → -1.3、速度差 2.6 m/s) の方。ところが 3 軸を独立にサンプルすると、
            この最悪ケースは偶然にしか出ない (符号が反転し、かつ大きさも残る組み合わせ)。
            実機の GK は上位方策やコントローラから左右に振られ続けるので、
            **デプロイ分布に合わせて最悪ケースの密度を上げる**。

        ☆ 追加の報酬は要らない。反転で転べば ``termination_penalty`` (-200) と
          以降の報酬喪失で十分強く罰される。**まず分布だけ変えて、足りなければ
          姿勢の項を足す** 順序にすること (報酬を先に足すと何が効いたか分からなくなる)。

        Args:
            prev: 直前の指令 (n, 3)。再サンプル前に控えたもの。
            reversal_min_speed: 直前の指令がこれ未満の env は対象外 [m/s]。
                ほぼ停止していた env を「反転」させても最悪ケースにならないため。
        """
        p = float(getattr(self.cfg, "reversal_prob", 0.0) or 0.0)
        if p <= 0.0:
            return torch.zeros(n, dtype=torch.bool, device=self.device)
        min_speed = float(getattr(self.cfg, "reversal_min_speed", 0.3))
        prev_speed = torch.norm(prev[:, :2], dim=1)
        do = (torch.rand(n, device=self.device) < p) & (prev_speed > min_speed)
        if not bool(do.any()):
            return do
        ids = torch.as_tensor(env_ids, device=self.device)[do]
        self.vel_command_b[ids] = -prev[do]
        return do


@configclass
class DiscreteVelocityCommandCfg(UniformVelocityCommandCfg):
    """離散速度コマンド（軸ごとの resolution で格子化）の設定クラス。"""

    class_type: type = DiscreteVelocityCommand

    stop_prob: float = 0.0
    """直前が全開でも **強制的にゼロ指令** にする確率 [0,1]。0 で従来どおり。

    ★ 2026-08-26 追加。実機で「歩行から停止する瞬間に振動する」と報告されたため。

    ☠ ``rel_standing_envs`` (既定 0.10) は **直前の指令と無関係な抽選** なので、
      「全開 → 停止」という最悪の遷移が出るのは偶然でしかない。``reversal_prob`` を
      入れたときと同じ論理 (GK で実際に効く遷移の密度を上げる) をそのまま適用する。
      GK は飛んだあと必ず止まって構えるので、停止の遷移は反転と同じくらい頻出する。

    ☠ ``reversal_prob`` と排他にすること (両方当たったら反転を優先)。同時に適用すると
      「反転してから即ゼロ」になって、どちらの遷移も学習できない。
    """

    stop_min_speed: float = 0.3
    """直前の指令がこれ未満の env は ``stop_prob`` の対象外 [m/s]。
    ほぼ止まっていた env をさらに止めても「歩行→停止」の遷移にならないため。"""

    lin_vel_x_resolution: float | None = None
    lin_vel_y_resolution: float | None = None
    ang_vel_z_resolution: float | None = None

    # 1 軸だけ残して他をゼロにする確率 (:meth:`DiscreteVelocityCommand._apply_pure_axis`)。
    # 既定 0.0 = 無効なので、既存タスクの挙動は変わらない。
    pure_axis_prob: float = 0.0

    # 直前の指令の符号反転に差し替える確率 (:meth:`DiscreteVelocityCommand._apply_reversal`)。
    # 既定 0.0 = 無効。
    reversal_prob: float = 0.0
    # 反転の対象にする直前指令の最低速度 [m/s]。これ未満はほぼ停止なので対象外。
    reversal_min_speed: float = 0.3


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
