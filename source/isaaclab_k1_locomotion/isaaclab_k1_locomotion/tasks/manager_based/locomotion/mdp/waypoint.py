# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""経由点追従タスク (Isaac-Waypoint-K1) の MDP 部品 (2026-10-29)。

凍結した歩行 expert (frozen_policies/best_walk) に (vx, vy, ωz) を与える上位方策を学習する。
上位方策の出力は環境の **action 項** ``walk_command`` (3 次元) として環境に渡り、
コマンド項 ``base_velocity`` (:class:`ExternalVelocityCommand`) が毎 step それをコピーする。
これにより歩行 expert が見る観測 (``velocity_commands`` 履歴 / 位相周波数) と報酬は
歩行学習環境のコードを一切変えずにそのまま使える。

経路は :class:`WaypointCommand` が各エピソード開始時にロボットの初期姿勢を基準に相対で生成する:
経由点 0〜``max_waypoints`` 個 + 最終目標姿勢 (xy + yaw)。``pass_radius`` (= モード) の
半径内に入ると次の経由点へ進む。

* **fast モード** (``loose_radius``, 既定 0.6 m): 経由点の近くを通れば良いので、角を曲線で
  ショートカットして最終姿勢へ最速で向かう。
* **strict モード** (``strict_radius``, 既定 0.15 m): 経由点の上をきちんと通る必要がある。

ゴール報酬は「全経由点を通過済み」の env にだけ出るので、どちらのモードでも経由点を
飛ばして直接ゴールへ行っても得をしない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

COMMAND_DIM = 3


# ---------------------------------------------------------------------------
# action 項: 上位方策の歩行コマンド
# ---------------------------------------------------------------------------
class WalkCommandAction(ActionTerm):
    """上位方策が出す歩行コマンド (vx, vy, ωz) を保持するだけの action 項 (物理には何もしない)。

    ``cfg.dim`` を変えれば任意次元の「値を保持するだけの action 項」として使える
    (block_env_cfg の関節残差 ``joint_residual`` 等)。
    """

    cfg: "WalkCommandActionCfg"

    def __init__(self, cfg: "WalkCommandActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, cfg.dim, device=self.device)
        self._prev = torch.zeros_like(self._raw)

    @property
    def action_dim(self) -> int:
        return self.cfg.dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def prev_actions(self) -> torch.Tensor:
        """1 step 前のコマンド (コマンド変化率ペナルティ用)。"""
        return self._prev

    def process_actions(self, actions: torch.Tensor):
        self._prev[:] = self._raw
        self._raw[:] = actions

    def apply_actions(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._prev[env_ids] = 0.0
        return {}


@configclass
class WalkCommandActionCfg(ActionTermCfg):
    class_type: type = WalkCommandAction

    dim: int = COMMAND_DIM
    """保持する値の次元 (既定は歩行コマンドの 3)。"""


# ---------------------------------------------------------------------------
# コマンド項: 外部 (action 項) から与えられる歩行コマンド
# ---------------------------------------------------------------------------
class ExternalVelocityCommand(CommandTerm):
    """``base_velocity`` の中身を action 項 ``walk_command`` からコピーするコマンド項。

    歩行 expert の観測 (``generated_commands``) と位相周波数 (``compute_cmd_phase_freq``) は
    ``command_manager.get_command("base_velocity")`` を読むので、ここに上位方策の出力を
    流し込めば歩行側のコードは無改造で済む。
    """

    cfg: "ExternalVelocityCommandCfg"

    def __init__(self, cfg: "ExternalVelocityCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, COMMAND_DIM, device=self.device)
        self.metrics["cmd_speed"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        self.metrics["cmd_speed"] = torch.norm(self.vel_command_b[:, :2], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        # リセット直後は停止指令 (action 項も reset で 0 になる)
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        self.vel_command_b[:] = self._env.action_manager.get_term(self.cfg.action_name).processed_actions


@configclass
class ExternalVelocityCommandCfg(CommandTermCfg):
    class_type: type = ExternalVelocityCommand

    asset_name: str = MISSING
    action_name: str = "walk_command"
    """コマンドの供給元となる action 項の名前。"""

    # エピソード中に再サンプルされると 0 に戻るので、実質 reset 時のみにする
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)


# ---------------------------------------------------------------------------
# コマンド項: 経由点 + 最終目標姿勢
# ---------------------------------------------------------------------------
WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/waypoints",
    markers={
        "waypoint": sim_utils.SphereCfg(
            radius=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0))
        ),
        "passed": sim_utils.SphereCfg(
            radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
        ),
        "goal": GREEN_ARROW_X_MARKER_CFG.markers["arrow"].replace(scale=(0.5, 0.5, 0.5)),
    },
)


class WaypointCommand(CommandTerm):
    """経由点列 + 最終目標姿勢を生成し、通過状況を追跡するコマンド項。

    内部表現 (world xy): ``points_w[:, k]`` は k 番目の経由点、index ``num_waypoints`` が
    ゴール、それ以降はゴールのコピー (パディング)。``target_idx`` は次に向かう点の index。

    観測 (``command``, base yaw 座標系): ``[次の K 点の xy (2K), ゴール xy (2),
    ゴール yaw 誤差の cos/sin (2), pass_radius (1)]``。
    """

    cfg: "WaypointCommandCfg"

    def __init__(self, cfg: "WaypointCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        n, k = self.num_envs, cfg.max_waypoints
        self.points_w = torch.zeros(n, k + 1, 2, device=self.device)
        self.num_waypoints = torch.zeros(n, dtype=torch.long, device=self.device)
        self.goal_yaw_w = torch.zeros(n, device=self.device)
        self.pass_radius = torch.full((n,), cfg.loose_radius, device=self.device)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        # suffix_len[k] = Σ_{j>=k} |p_{j+1} - p_j| (点 k からゴールまでの折れ線長)
        self.suffix_len = torch.zeros(n, k + 1, device=self.device)
        self.prev_path_length = torch.zeros(n, device=self.device)
        self.reached_goal = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.time_to_goal = torch.zeros(n, device=self.device)
        self.cmd_buf = torch.zeros(n, 2 * k + 5, device=self.device)
        self._env_idx = torch.arange(n, device=self.device)
        self._window = torch.arange(k, device=self.device)
        for name in ("goal_dist", "goal_yaw_err", "waypoints_passed_frac", "reached_goal", "time_to_goal"):
            self.metrics[name] = torch.zeros(n, device=self.device)

    # -- 状態 ----------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return self.cmd_buf

    def _pos(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, :2]

    def goal_w(self) -> torch.Tensor:
        return self.points_w[self._env_idx, self.num_waypoints]

    def goal_dist(self) -> torch.Tensor:
        return torch.norm(self._pos() - self.goal_w(), dim=1)

    def goal_yaw_err(self) -> torch.Tensor:
        return math_utils.wrap_to_pi(self.goal_yaw_w - self.robot.data.heading_w)

    def all_passed(self) -> torch.Tensor:
        return self.target_idx >= self.num_waypoints

    def path_length(self) -> torch.Tensor:
        """現在位置 → 次の点 → … → ゴール の残り折れ線長。"""
        target = self.points_w[self._env_idx, self.target_idx]
        return torch.norm(self._pos() - target, dim=1) + self.suffix_len[self._env_idx, self.target_idx]

    # -- CommandTerm hooks ------------------------------------------------------
    def _update_metrics(self):
        self.metrics["goal_dist"] = self.goal_dist()
        self.metrics["goal_yaw_err"] = self.goal_yaw_err().abs()
        self.metrics["waypoints_passed_frac"] = self.target_idx.float() / self.num_waypoints.clamp(min=1).float()
        self.metrics["reached_goal"] = self.reached_goal.float()
        self.metrics["time_to_goal"] = torch.where(
            self.reached_goal, self.time_to_goal, torch.full_like(self.time_to_goal, self._env.max_episode_length_s)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        k = cfg.max_waypoints
        pos = self._pos()[env_ids]
        heading = self.robot.data.heading_w[env_ids]

        lo, hi = cfg.num_waypoints_range
        num_wp = torch.randint(lo, hi + 1, (n,), device=self.device)
        # 折れ線をランダムウォークで生成: 最初の方向は全方位、以降は前セグメントからの旋回
        direction = heading + torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        cur = pos.clone()
        pts = []
        for i in range(k + 1):
            if i > 0:
                direction = direction + torch.empty(n, device=self.device).uniform_(-cfg.max_turn, cfg.max_turn)
            seg = torch.empty(n, device=self.device).uniform_(*cfg.segment_length_range)
            cur = cur + seg.unsqueeze(1) * torch.stack([torch.cos(direction), torch.sin(direction)], dim=1)
            pts.append(cur)
        pts = torch.stack(pts, dim=1)  # (n, k+1, 2)
        # index num_wp がゴール。それ以降はゴールのコピーで埋める
        goal = pts[torch.arange(n, device=self.device), num_wp]
        pad = torch.arange(k + 1, device=self.device).unsqueeze(0) > num_wp.unsqueeze(1)
        pts = torch.where(pad.unsqueeze(-1), goal.unsqueeze(1), pts)
        seg_len = torch.norm(pts[:, 1:] - pts[:, :-1], dim=-1)  # (n, k)
        suffix = torch.flip(torch.cumsum(torch.flip(seg_len, [1]), dim=1), [1])
        suffix = torch.cat([suffix, torch.zeros(n, 1, device=self.device)], dim=1)

        self.points_w[env_ids] = pts
        self.num_waypoints[env_ids] = num_wp
        self.suffix_len[env_ids] = suffix
        self.goal_yaw_w[env_ids] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        strict = torch.rand(n, device=self.device) < cfg.strict_prob
        self.pass_radius[env_ids] = torch.where(
            strict, torch.full_like(strict, cfg.strict_radius, dtype=torch.float), torch.full_like(strict, cfg.loose_radius, dtype=torch.float)
        )
        self.target_idx[env_ids] = 0
        self.reached_goal[env_ids] = False
        self.time_to_goal[env_ids] = 0.0
        self.prev_path_length[env_ids] = self.path_length()[env_ids]

    def _update_command(self):
        pos = self._pos()
        # 経由点通過判定 (ゴールの index からは進めない)
        target = self.points_w[self._env_idx, self.target_idx]
        near = torch.norm(pos - target, dim=1) < self.pass_radius
        self.target_idx += (near & (self.target_idx < self.num_waypoints)).long()

        # ゴール到達 (全経由点通過済み + 位置 + 向き)
        at_goal = (
            self.all_passed()
            & (self.goal_dist() < self.cfg.goal_pos_tolerance)
            & (self.goal_yaw_err().abs() < self.cfg.goal_yaw_tolerance)
        )
        newly = at_goal & ~self.reached_goal
        self.time_to_goal[newly] = self._env.episode_length_buf[newly].float() * self._env.step_dt
        self.reached_goal |= at_goal

        # 観測: 次の K 点 + ゴール を base yaw 座標系に変換
        k = self.cfg.max_waypoints
        idx = (self.target_idx.unsqueeze(1) + self._window.unsqueeze(0)).clamp(max=k)  # (n, k)
        sel = torch.gather(self.points_w, 1, idx.unsqueeze(-1).expand(-1, -1, 2))  # (n, k, 2)
        rel = torch.cat([sel, self.goal_w().unsqueeze(1)], dim=1) - pos.unsqueeze(1)  # (n, k+1, 2)
        heading = self.robot.data.heading_w
        c, s = torch.cos(heading).unsqueeze(1), torch.sin(heading).unsqueeze(1)
        rel_b = torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], dim=-1)
        yaw_err = self.goal_yaw_err()
        self.cmd_buf[:] = torch.cat(
            [
                rel_b.flatten(1),
                torch.cos(yaw_err).unsqueeze(1),
                torch.sin(yaw_err).unsqueeze(1),
                self.pass_radius.unsqueeze(1),
            ],
            dim=1,
        )

    # -- 可視化 ----------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_visualizer"):
                self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self._visualizer.set_visibility(True)
        elif hasattr(self, "_visualizer"):
            self._visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        n, k1 = self.points_w.shape[:2]
        pos = torch.cat([self.points_w, torch.full((n, k1, 1), 0.1, device=self.device)], dim=-1).reshape(-1, 3)
        idx = torch.arange(k1, device=self.device).unsqueeze(0).expand(n, -1)
        marker = torch.where(idx < self.target_idx.unsqueeze(1), 1, 0)  # passed / waypoint
        marker = torch.where(idx >= self.num_waypoints.unsqueeze(1), 2, marker)  # goal (+ padding)
        zeros = torch.zeros(n, k1, device=self.device)
        yaw = self.goal_yaw_w.unsqueeze(1).expand(n, k1)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw).reshape(-1, 4)
        self._visualizer.visualize(pos, quat, marker_indices=marker.reshape(-1))


@configclass
class WaypointCommandCfg(CommandTermCfg):
    class_type: type = WaypointCommand

    asset_name: str = MISSING
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """エピソード中は再サンプルしない (reset 時のみ)。"""

    max_waypoints: int = 4
    """経由点の最大数 (観測の経由点スロット数でもある)。"""
    num_waypoints_range: tuple[int, int] = (0, 4)
    """経由点数の範囲 (両端含む)。0 ならゴールへ直行。"""
    segment_length_range: tuple[float, float] = (0.6, 2.0)
    """隣接する点の間の距離 [m]。"""
    max_turn: float = 2.0
    """セグメント間の方向変化の最大値 [rad]。"""

    strict_prob: float = 0.5
    """strict モード (経由点の上を通る) になる確率。残りは fast (ショートカット可) モード。"""
    strict_radius: float = 0.15
    """strict モードの経由点通過半径 [m]。"""
    loose_radius: float = 0.6
    """fast モードの経由点通過半径 [m]。"""

    goal_pos_tolerance: float = 0.15
    """ゴール到達判定の位置許容 [m] (メトリクス用)。"""
    goal_yaw_tolerance: float = 0.2
    """ゴール到達判定の向き許容 [rad] (メトリクス用)。"""

    visualizer_cfg: VisualizationMarkersCfg = WAYPOINT_MARKER_CFG


# ---------------------------------------------------------------------------
# 報酬
# ---------------------------------------------------------------------------
def path_progress(env: "ManagerBasedRLEnv", command_name: str = "waypoints") -> torch.Tensor:
    """残り経路長の減少量 [m/step] (ポテンシャル形式: 総和 = 初期経路長 − 最終残り長)。"""
    term: WaypointCommand = env.command_manager.get_term(command_name)
    length = term.path_length()
    delta = term.prev_path_length - length
    term.prev_path_length = length
    return delta


def goal_position_exp(env: "ManagerBasedRLEnv", std: float, command_name: str = "waypoints") -> torch.Tensor:
    """全経由点通過後、ゴール位置に近いほど高い [0, 1]。"""
    term: WaypointCommand = env.command_manager.get_term(command_name)
    return term.all_passed().float() * torch.exp(-torch.square(term.goal_dist() / std))


def goal_heading_exp(
    env: "ManagerBasedRLEnv", std_yaw: float, std_pos: float, command_name: str = "waypoints"
) -> torch.Tensor:
    """全経由点通過後、ゴール近傍でゴール向きに合っているほど高い [0, 1]。"""
    term: WaypointCommand = env.command_manager.get_term(command_name)
    gate = term.all_passed().float() * torch.exp(-torch.square(term.goal_dist() / std_pos))
    return gate * torch.exp(-torch.square(term.goal_yaw_err() / std_yaw))


def walk_command_rate_l2(env: "ManagerBasedRLEnv", action_name: str = "walk_command") -> torch.Tensor:
    """上位方策の歩行コマンドの 1 step 変化量 ||c_t − c_{t−1}||²。"""
    term: WalkCommandAction = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(term.raw_actions - term.prev_actions), dim=1)
