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

    def __init__(self, cfg: "ExtremeVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # 着地時歩幅メトリクス (2026-08-25): 足が接地した瞬間の「着地足 − 支持足」の
        # コマンド方向距離を記録し、エピソード内の着地平均を Metrics/.../stride_touchdown
        # として出す。歩幅誘導報酬 (feet_stride_length) が実際に歩幅を伸ばしているかの
        # 検証用で、報酬値ではなく実測歩幅そのものを見る。
        self.metrics["stride_touchdown"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stride_touchdown_fast"] = torch.zeros(self.num_envs, device=self.device)
        self._stride_sum = torch.zeros(self.num_envs, device=self.device)
        self._stride_cnt = torch.zeros(self.num_envs, device=self.device)
        self._stride_fast_sum = torch.zeros(self.num_envs, device=self.device)
        self._stride_fast_cnt = torch.zeros(self.num_envs, device=self.device)
        self._stride_ids: tuple[list[int], list[int]] | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        # 着地サンプルが 1 つも無い env (停止・低速・早期転倒) を平均に含めると値が
        # 希釈されるので、サンプルのある env だけで平均を取り直す
        masked = {}
        for name, s, c in (
            ("stride_touchdown", self._stride_sum, self._stride_cnt),
            ("stride_touchdown_fast", self._stride_fast_sum, self._stride_fast_cnt),
        ):
            cnt = c[env_ids]
            has = cnt > 0
            if has.any():
                masked[name] = (s[env_ids][has] / cnt[has]).mean().item()
        extras = super().reset(env_ids)
        # サンプル無しの reset ではキーごと落とす (NaN を返すと logger 側の平均が NaN に汚染される)
        for name in ("stride_touchdown", "stride_touchdown_fast"):
            extras.pop(name, None)
        extras.update(masked)
        for buf in (self._stride_sum, self._stride_cnt, self._stride_fast_sum, self._stride_fast_cnt):
            buf[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        super()._update_metrics()
        sensor = self._env.scene.sensors.get("contact_forces")
        if sensor is None:
            return
        if self._stride_ids is None:
            sensor_ids, names = sensor.find_bodies(".*_foot_link")
            robot_ids = [self.robot.find_bodies(n)[0][0] for n in names]
            self._stride_ids = (sensor_ids, robot_ids)
        sensor_ids, robot_ids = self._stride_ids
        # 遊脚 0.1 s 以上の後の接地のみ「着地」とみなす (接地中のバウンド再接触を除外)
        first = sensor.compute_first_contact(self._env.step_dt)[:, sensor_ids]  # [N, 2]
        first = first & (sensor.data.last_air_time[:, sensor_ids] > 0.1)

        speed = torch.norm(self.vel_command_b[:, :2], dim=1)
        cmd_dir = self.vel_command_b[:, :2] / speed.clamp(min=1e-6).unsqueeze(1)
        base_pos_w = self.robot.data.root_pos_w[:, :3]
        quat_yaw = math_utils.yaw_quat(self.robot.data.root_quat_w)
        rel = [
            math_utils.quat_apply_inverse(quat_yaw, self.robot.data.body_pos_w[:, i, :3] - base_pos_w)[:, :2]
            for i in robot_ids
        ]
        gap01 = ((rel[0] - rel[1]) * cmd_dir).sum(dim=1)
        # 着地した足を先頭に取った「着地足 − 支持足」距離。両足同時着地は片側のみ数える。
        stride = torch.where(first[:, 0], gap01, -gap01)
        hit = (first[:, 0] | first[:, 1]) & (speed > 0.1)
        self._stride_sum += torch.where(hit, stride, torch.zeros_like(stride))
        self._stride_cnt += hit.float()
        self.metrics["stride_touchdown"] = self._stride_sum / self._stride_cnt.clamp(min=1.0)
        # 前後高速コマンド (|vx|>1.0, |vy|<0.3) 時のみの前後方向着地歩幅: 歩幅誘導の効果は
        # ここに出る。横歩きは「踏み出し→引き寄せ」で引き寄せ側の歩幅が負になるため除外。
        vx, vy = self.vel_command_b[:, 0], self.vel_command_b[:, 1]
        gap_x = (rel[0][:, 0] - rel[1][:, 0]) * torch.sign(vx)
        stride_x = torch.where(first[:, 0], gap_x, -gap_x)
        hit_fast = hit & (vx.abs() > 1.0) & (vy.abs() < 0.3)
        self._stride_fast_sum += torch.where(hit_fast, stride_x, torch.zeros_like(stride))
        self._stride_fast_cnt += hit_fast.float()
        self.metrics["stride_touchdown_fast"] = self._stride_fast_sum / self._stride_fast_cnt.clamp(min=1.0)

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


class LateralVelocityCommand(ExtremeVelocityCommand):
    """横 (y) 方向の速度コマンドを重点的にサンプリングする速度コマンド (ゴールキーパー用)。

    ``ExtremeVelocityCommand`` の一様 + extreme corner サンプリングに加えて、
    確率 ``cfg.lateral_prob`` で resample を「lateral モード」にする:

      - lin_vel_y を符号ランダムで ``|vy| ∈ [lateral_frac*max, max]`` から引く
        (現在の ``cfg.ranges`` を参照するのでカリキュラムの y 範囲拡張に追従)
      - lin_vel_x は範囲を ``lateral_x_scale`` 倍に縮めた範囲から引き直す
        (横ステップ主体の状況を作る。0 にはしないので斜め移動も残る)
      - lateral に選ばれた env は standing 抽選から除外する

    extreme と lateral の両方に選ばれた env は lateral が後勝ちする (vy は上端域、
    vx は縮小域)。``lateral_prob=0.0`` (既定) では ``ExtremeVelocityCommand`` と同一挙動。
    """

    cfg: "LateralVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        prob = float(self.cfg.lateral_prob)
        if prob <= 0.0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pick = torch.rand(ids.numel(), device=self.device) < prob
        if not pick.any():
            return
        ids = ids[pick]
        m = ids.numel()

        # lin_vel_y: 符号ランダムで上端域 [frac*|bound|, |bound|] から引く
        frac = float(self.cfg.lateral_frac)
        lo, hi = float(self.cfg.ranges.lin_vel_y[0]), float(self.cfg.ranges.lin_vel_y[1])
        pick_pos = torch.rand(m, device=self.device) < 0.5
        bound = torch.where(
            pick_pos,
            torch.full((m,), hi, device=self.device),
            torch.full((m,), lo, device=self.device),
        )
        u = torch.rand(m, device=self.device)
        self.vel_command_b[ids, 1] = bound * (frac + (1.0 - frac) * u)

        # lin_vel_x: 縮小レンジから引き直す
        x_scale = float(self.cfg.lateral_x_scale)
        x_lo, x_hi = float(self.cfg.ranges.lin_vel_x[0]), float(self.cfg.ranges.lin_vel_x[1])
        self.vel_command_b[ids, 0] = torch.empty(m, device=self.device).uniform_(
            x_lo * x_scale, x_hi * x_scale
        )

        # lateral env は立ち止まり抽選から除外 (コマンドが 0 に上書きされるのを防ぐ)
        self.is_standing_env[ids] = False


@configclass
class LateralVelocityCommandCfg(ExtremeVelocityCommandCfg):
    """`LateralVelocityCommand` の設定クラス。"""

    class_type: type = LateralVelocityCommand

    lateral_prob: float = 0.0
    """resample を lateral (横重視) モードにする確率 [0,1]。0 で ExtremeVelocityCommand と同一。"""

    lateral_frac: float = 0.6
    """lateral モードで引く lin_vel_y の大きさの下限 (レンジ端に対する割合)。"""

    lateral_x_scale: float = 0.4
    """lateral モードで lin_vel_x の範囲に掛ける縮小倍率。"""


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


class TargetHeadingCommand(UniformVelocityCommand):
    """目標ヘディングまでの「残り角」をコマンドとして返す (その場高速回転タスク用)。

    歩行タスクのコマンドは ``(vx, vy, ωz)`` の速度指令だが、本コマンドは::

        command = [0, 0, wrap_to_pi(ψ_target - ψ_current) / π]     shape (num_envs, 3)

    を返す。次元を 3 のままにしているのは意図的で、観測 1 ステップ 49 次元という
    レイアウトが ``history_layout.py`` / ``mdp/symmetry.py`` /
    ``agents/history_policy_exporter.py`` / C++ デプロイ側に共有ハードコードされている
    ため、枠の「中身」だけを差し替えて既存資産をそのまま使う。

    この符号化が持つ性質:

    * **左右対称性がそのまま通る**: ``mdp/symmetry.py`` の ``vel_cmd`` 変換は符号
      ``(+1, -1, -1)``。左右反転で Δψ → -Δψ なので slot2 (符号 -1) と完全一致し、
      ゼロ 2 枠は符号不変。symmetry / history_layout の変更は不要。
    * **目標到達で ``||cmd||`` が 0 に落ちる**: 既存の「停止指令」ゲートが自然に発火する。
      ``phase_obs`` は ``||cmd[:, :3]|| < cmd_threshold`` で位相をゼロ化するので
      |Δψ| < 9° (閾値 0.05 の場合) で位相クロックが切れ、``_stand_still_boost`` が
      action 平滑ペナルティを強めて「目標で静かに止まる」ことを促す。
    * ``||cmd[:, :2]|| = 0`` なので ``compute_cmd_phase_freq`` は常に ``low_freq`` を返す。

    継承元の heading 制御則 (``ωz = clip(stiffness * Δψ)``) は**使わない**。
    P 則で ωz 参照値を作ると「加速 → 減速 → 目標で停止」の減速プロファイルを
    ポリシー側が設計できないため、残り角そのものを渡して任せる。
    """

    cfg: "TargetHeadingCommandCfg"

    def __init__(self, cfg: "TargetHeadingCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # 速度追従メトリクスは意味を失うので取り除き、回転タスク用に差し替える。
        self.metrics.pop("error_vel_xy", None)
        self.metrics.pop("error_vel_yaw", None)
        self.metrics["heading_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["settle_time_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_in_target"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

        # 「その場」判定の基準位置 (コマンド発行時の xy)。base_xy_drift_l2 が参照する。
        self.origin_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

        # --- 計測バッファ ---
        # コマンド単位 (再サンプリングごとにリセット)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._settled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # エピソード単位 (reset でのみクリア)
        self._err_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_cnt = torch.zeros(self.num_envs, device=self.device)
        self._in_target_cnt = torch.zeros(self.num_envs, device=self.device)
        self._settle_sum = torch.zeros(self.num_envs, device=self.device)
        self._settle_cnt = torch.zeros(self.num_envs, device=self.device)
        self._issued_cnt = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TargetHeadingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTurn angle range [rad]: {self.cfg.turn_angle_range}\n"
        msg += f"\tSuccess threshold [rad]: {self.cfg.success_threshold}"
        return msg

    """
    Properties
    """

    @property
    def heading_error(self) -> torch.Tensor:
        """``wrap_to_pi(ψ_target - ψ_current)`` [rad], shape ``(num_envs,)``。

        毎回 ``robot.data.heading_w`` を live 参照して計算し直すので、常に最新値になる。
        報酬側がこれを使うのは重要で、``ManagerBasedRLEnv.step`` は
        reward → reset → ``command_manager.compute`` の順に走るため、報酬計算時点の
        ``command_manager.get_command(...)`` は 1 ステップ古い値を返すため。
        """
        return math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)

    """
    Implementation specific functions.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        # 一度も目標に到達しなかった env を平均に含めると settle_time が希釈されるので、
        # 到達サンプルのある env だけで平均を取り直す (ExtremeVelocityCommand と同じ方針)。
        masked: dict[str, float] = {}
        cnt = self._settle_cnt[env_ids]
        has = cnt > 0
        if has.any():
            masked["settle_time_s"] = (self._settle_sum[env_ids][has] / cnt[has]).mean().item()

        # エピソード累積バッファはここでクリアする。この後 super().reset() 内の
        # _resample が新しいコマンドを立てて _issued_cnt を 1 にするので、順序が重要。
        for buf in (
            self._err_sum,
            self._step_cnt,
            self._in_target_cnt,
            self._settle_sum,
            self._settle_cnt,
            self._issued_cnt,
        ):
            buf[env_ids] = 0.0

        extras = super().reset(env_ids)
        # 到達サンプル無しの reset ではキーごと落とす (0 での希釈を防ぐ)
        extras.pop("settle_time_s", None)
        extras.update(masked)
        return extras

    def _update_metrics(self):
        # super() は呼ばない (速度誤差メトリクスを積まないため)。
        err = self.heading_error.abs()
        self._elapsed += self._env.step_dt
        self._err_sum += err
        self._step_cnt += 1.0

        in_target = err < float(self.cfg.success_threshold)
        self._in_target_cnt += in_target.float()
        # 「初めて目標圏内に入った時刻」= 到達時間。以降の出入りでは更新しない。
        newly_settled = in_target & (~self._settled)
        self._settled |= in_target
        self._settle_sum += torch.where(newly_settled, self._elapsed, torch.zeros_like(self._elapsed))
        self._settle_cnt += newly_settled.float()

        denom = self._step_cnt.clamp(min=1.0)
        self.metrics["heading_error_deg"] = self._err_sum / denom * (180.0 / math.pi)
        self.metrics["time_in_target"] = self._in_target_cnt / denom
        self.metrics["settle_time_s"] = self._settle_sum / self._settle_cnt.clamp(min=1.0)
        self.metrics["success_rate"] = self._settle_cnt / self._issued_cnt.clamp(min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        # 継承元は cfg.ranges から (vx, vy, ωz) を引くが、本コマンドでは全て無視する。
        n = len(env_ids)
        if n == 0:
            return
        lo, hi = self.cfg.turn_angle_range
        # 回転量は「大きさ」を引いてから符号をランダムに付ける。こうすると
        # turn_angle_range の下限で「ほぼ 0 の目標」が出にくくなり、カリキュラムで
        # 上限だけを動かせば難易度が素直に上がる。
        magnitude = torch.empty(n, device=self.device).uniform_(float(lo), float(hi))
        sign = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.ones(n, device=self.device),
            -torch.ones(n, device=self.device),
        )
        self.heading_target[env_ids] = math_utils.wrap_to_pi(
            self.robot.data.heading_w[env_ids] + sign * magnitude
        )
        # 「その場」の基準位置を、このコマンドの発行時点で取り直す。
        self.origin_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2]

        # 継承元のフラグは参照しないが、play.py 等が触っても破綻しないよう整合させておく。
        self.is_heading_env[env_ids] = True
        self.is_standing_env[env_ids] = False

        # コマンド単位の計測をリセットし、発行回数を数える。
        self._elapsed[env_ids] = 0.0
        self._settled[env_ids] = False
        self._issued_cnt[env_ids] += 1.0

    def _update_command(self):
        # 継承元の P 則 (clip(stiffness * Δψ)) と standing ゼロ化は使わない。
        self.vel_command_b[:, 0] = 0.0
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = self.heading_error / math.pi

    """
    可視化: 継承元は command[:, :2] (= 常に 0) で矢印を描くため長さ 0 になる。
    代わりに「目標ヘディング (緑)」と「現在ヘディング (青)」の向きを描く。
    """

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        goal_scale, goal_quat = self._resolve_heading_to_arrow(
            self.heading_target, self.goal_vel_visualizer
        )
        cur_scale, cur_quat = self._resolve_heading_to_arrow(
            self.robot.data.heading_w, self.current_vel_visualizer
        )
        self.goal_vel_visualizer.visualize(base_pos_w, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(base_pos_w, cur_quat, cur_scale)

    def _resolve_heading_to_arrow(
        self, heading_w: torch.Tensor, visualizer: VisualizationMarkers
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ワールド yaw 角 [rad] を矢印の (scale, quaternion) に変換する。"""
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(heading_w.shape[0], 1)
        zeros = torch.zeros_like(heading_w)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_w)
        return arrow_scale, arrow_quat


@configclass
class TargetHeadingCommandCfg(UniformVelocityCommandCfg):
    """`TargetHeadingCommand` の設定クラス。

    ``ranges`` の ``lin_vel_x`` / ``lin_vel_y`` / ``ang_vel_z`` は
    ``_resample_command`` を完全に置き換えたため**未使用**。``configclass`` の
    MISSING チェックを通すためだけにダミー値を入れておくこと。
    ``heading_command`` は False 固定 (継承元の P 則を無効化する意思表示であり、
    かつ ``ranges.heading=None`` でも ``__init__`` のバリデーションを通すため)。
    """

    class_type: type = TargetHeadingCommand

    heading_command: bool = False

    turn_angle_range: tuple[float, float] = (0.3, math.pi)
    """1 コマンドあたりの回転量 |Δψ_0| [rad] のサンプル範囲。符号は別途ランダムに付く。
    ``turn_angle_curriculum`` がこの上限を段階的に拡げる。"""

    success_threshold: float = 0.087
    """「到達した」とみなす残り角 [rad] (既定 5°)。メトリクス集計にのみ使い、報酬には使わない。"""
