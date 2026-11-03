# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール阻止 (キーパー) タスク (Isaac-Block-K1) の MDP 部品 (2026-11-01)。

凍結した歩行 expert (frozen_policies/best_walk) を事前分布として、「歩行コマンド (3) +
関節残差 (12)」を出す方策を学習する。ボールは :class:`BallLaunchCommand` が各エピソード開始時に
ロボットへ向けて射出し、ロボットの初期姿勢で決まる **ゴールライン** (初期位置を通り初期進行方向に
垂直な線) を基準に

* **conceded** (失点): ボールがラインの後ろ (``behind_x``) かつゴール幅 (``goal_half_width``) 内に入った
* **resolved** (阻止): 失点せずに、ボールが (a) ゴール幅の外に逸れた、(b) 前方へ弾き返された、
  (c) 止まった、のいずれかになった後 ``resolve_hold_s`` 秒 (既定 2.5 s) 転倒も失点もせずに立ち続けた

を判定する。両方とも終了条件 (time_out=False) で、報酬は ``is_terminated_term`` で与える。
阻止の瞬間に終わらせないのは、体勢を崩しながら触ればよい方策にしないため (2026-11-02 ユーザー指示):
触った後に転べば転倒の −200 だけで阻止の +100 は貰えない。

ボール観測は 2 系統:

* 真値 (:func:`ball_state_rel`): base yaw 座標系の相対位置 xy + 速度 xy (critic 側 encoder 用)
* ノイズ付き (:class:`DelayedBallObs`): カメラ 30 fps の離散更新 + 検出遅延 (既定 160〜240 ms, env ごと) +
  検出欠け + 位置 ±10 cm / 速度 ±1.0 m/s の一様ノイズ (actor 側 encoder 用)

いずれも ObservationGroupCfg の history_length (``BALL_HISTORY``) で履歴化し、方策側の
causal transformer (agents/block_policy.py) に (T, 4) で渡す。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.utils import configclass

from .observations import ball_pos_rel, ball_vel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

BALL_DIM = 4
"""ボール観測 1 フレームの次元 (相対位置 xy + 速度 xy)。"""

BALL_HISTORY = 20
"""encoder に渡すボール観測の履歴長 (step 数, 50 Hz で 0.4 s)。"""

JOINT_DIM = 12
"""関節残差の次元 (歩行 expert の action 次元)。"""


# ---------------------------------------------------------------------------
# 観測
# ---------------------------------------------------------------------------
def ball_state_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """ボールの真値: base yaw 座標系の相対位置 xy + 速度 xy (N, 4)。"""
    return torch.cat([ball_pos_rel(env), ball_vel(env)], dim=-1)


class DelayedBallObs(ManagerTermBase):
    """カメラ検出を模したボール観測: 遅延 + 30 fps 離散更新 + 検出欠け + ノイズ。

    真値 (:func:`ball_state_rel`) をリングバッファに毎 step 積み、env ごとに抽選した遅延 step 分
    前の値を「カメラが見た値」とする。カメラ周期 (1/fps) ごとに新フレームが届き、その時だけ
    ノイズを乗せた値で出力を更新する (届かない step は前回値を保持 = 実機の stale フレーム)。
    ノイズは cfg の noise ではなくここで乗せる (保持フレームに毎 step 別ノイズが乗るのを防ぐ)。
    """

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        n, dev = env.num_envs, env.device
        self._len = int(math.ceil(p["delay_range"][1] / env.step_dt)) + 1
        self._buf = torch.zeros(n, self._len, BALL_DIM, device=dev)
        self._ptr = 0
        self._delay = torch.zeros(n, dtype=torch.long, device=dev)
        self._acc = torch.zeros(n, device=dev)
        self._out = torch.zeros(n, BALL_DIM, device=dev)
        self._fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self._env_idx = torch.arange(n, device=dev)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        lo, hi = self.cfg.params["delay_range"]
        delay = torch.empty_like(self._acc[env_ids]).uniform_(lo, hi) / self._env.step_dt
        self._delay[env_ids] = delay.round().long().clamp(0, self._len - 1)
        self._acc[env_ids] = 0.0
        self._fresh[env_ids] = True

    def _noise(self, x: torch.Tensor, pos_noise: float, vel_noise: float, vel_noise_rel: float) -> torch.Tensor:
        """ガウスノイズ。速度の σ は速さに比例して増える (カルマンフィルタ推定の誤差を模す)。"""
        speed = torch.norm(x[:, 2:4], dim=1, keepdim=True)
        std = torch.cat([torch.full_like(speed, pos_noise).expand(-1, 2), (vel_noise + vel_noise_rel * speed).expand(-1, 2)], dim=1)
        return x + torch.randn_like(x) * std

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        delay_range: tuple[float, float] = (0.16, 0.24),
        camera_fps: float = 30.0,
        pos_noise: float = 0.05,
        vel_noise: float = 1.0,
        vel_noise_rel: float = 0.3,
        drop_prob: float = 0.1,
    ) -> torch.Tensor:
        true = ball_state_rel(env)
        fresh = self._fresh
        if fresh.any():
            # リセット直後: 履歴を現在値で埋める (「少し前から見えていた」扱い)
            self._buf[fresh] = true[fresh].unsqueeze(1)
            self._out[fresh] = self._noise(true[fresh], pos_noise, vel_noise, vel_noise_rel)
            self._fresh[:] = False
        self._buf[:, self._ptr] = true
        delayed = self._buf[self._env_idx, (self._ptr - self._delay) % self._len]
        self._ptr = (self._ptr + 1) % self._len

        # カメラ周期で新フレーム到着 (検出欠けは到着扱いにしない → 前回値を保持)
        self._acc += env.step_dt * camera_fps
        new_frame = self._acc >= 1.0
        self._acc -= new_frame.float()
        new_frame &= torch.rand_like(self._acc) >= drop_prob
        self._out = torch.where(new_frame.unsqueeze(1), self._noise(delayed, pos_noise, vel_noise, vel_noise_rel), self._out)
        return self._out.clone()


# ---------------------------------------------------------------------------
# コマンド項: ボール射出 + ゴールライン + 阻止/失点判定
# ---------------------------------------------------------------------------
class BallLaunchCommand(CommandTerm):
    """エピソード開始時にボールをロボットへ向けて射出し、ゴールライン基準で結果を判定する。

    観測 (``command``, base yaw 座標系): ``[ライン原点の相対位置 xy (2), ライン向きの誤差 cos/sin (2)]``。
    ロボットが「自分の守るべき線がどこか」を知るための 4 次元。
    """

    cfg: "BallLaunchCommandCfg"

    def __init__(self, cfg: "BallLaunchCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self._feet_idx = self.robot.find_bodies(cfg.feet_body_names)[0]
        n = self.num_envs
        self.line_pos_w = torch.zeros(n, 2, device=self.device)
        self.line_heading = torch.zeros(n, device=self.device)
        self.cmd_buf = torch.zeros(n, 4, device=self.device)
        self.launch_lateral = torch.zeros(n, device=self.device)  # 狙った左右位置 (評価用)
        # 最初に阻止条件を満たした時刻 [s] (未達なら −1) と、その時の内訳 (0 なし / 1 幅外 / 2 前へ弾く / 3 止めた)
        self.resolved_at = torch.full((n,), -1.0, device=self.device)
        self.resolve_kind = torch.zeros(n, dtype=torch.long, device=self.device)
        for name in ("conceded", "resolved", "out_wide", "away", "stopped", "lost_after_touch", "launch_speed"):
            self.metrics[name] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.cmd_buf

    # -- ライン座標系 (x: 初期進行方向 = 前, y: 左) ------------------------------------
    def _to_line(self, vec_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c, s = torch.cos(self.line_heading), torch.sin(self.line_heading)
        return c * vec_w[:, 0] + s * vec_w[:, 1], -s * vec_w[:, 0] + c * vec_w[:, 1]

    def ball_line_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ボールのライン座標系での (x, y, vx, vy)。"""
        x, y = self._to_line(self.ball.data.root_pos_w[:, :2] - self.line_pos_w)
        vx, vy = self._to_line(self.ball.data.root_lin_vel_w[:, :2])
        return x, y, vx, vy

    def body_line_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """胴体のライン座標系での (x, y)。x > 0 はラインより前に出ている。"""
        return self._to_line(self.robot.data.root_pos_w[:, :2] - self.line_pos_w)

    def feet_line_y(self) -> torch.Tensor:
        """両足のライン座標系 y (N, 2)。"""
        feet = self.robot.data.body_link_pos_w[:, self._feet_idx, :2] - self.line_pos_w.unsqueeze(1)
        c, s = torch.cos(self.line_heading).unsqueeze(1), torch.sin(self.line_heading).unsqueeze(1)
        return -s * feet[..., 0] + c * feet[..., 1]

    def conceded(self) -> torch.Tensor:
        x, y, _, _ = self.ball_line_state()
        return (x < -self.cfg.behind_x) & (y.abs() < self.cfg.goal_half_width)

    def resolve_parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """阻止の内訳 (幅外に逸れた, 前へ弾いた, 止めた)。"""
        x, y, vx, _ = self.ball_line_state()
        speed = torch.norm(self.ball.data.root_lin_vel_w[:, :2], dim=1)
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt
        out_wide = (y.abs() >= self.cfg.goal_half_width) & (x < 0.5)
        away = (vx > self.cfg.away_speed) & (x > 0.0)
        stopped = (speed < self.cfg.stop_speed) & (x > -self.cfg.behind_x) & (elapsed > 0.5)
        return out_wide, away, stopped

    def resolved_instant(self) -> torch.Tensor:
        """いま阻止条件を満たしているか (瞬間値)。"""
        out_wide, away, stopped = self.resolve_parts()
        return ~self.conceded() & (out_wide | away | stopped)

    def elapsed(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    def resolved(self) -> torch.Tensor:
        """阻止 (終了条件): 最初の阻止から resolve_hold_s 秒、失点せずに経過した。"""
        held = (self.resolved_at >= 0.0) & (self.elapsed() - self.resolved_at >= self.cfg.resolve_hold_s)
        return held & ~self.conceded()

    # -- CommandTerm hooks ----------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # 終了した step の状態で結果メトリクスを確定させてからログ・再射出する
        # (基底の reset は compute() 時点 = 1 step 前のメトリクスをログしてしまう)
        self._update_metrics()
        return super().reset(env_ids)

    def _update_metrics(self):
        conceded = self.conceded()
        resolved = self.resolved()
        self.metrics["conceded"] = conceded.float()
        self.metrics["resolved"] = resolved.float()
        for kind, name in enumerate(("out_wide", "away", "stopped"), start=1):
            self.metrics[name] = (resolved & (self.resolve_kind == kind)).float()
        # 一度は阻止したが、保持時間を待たずに終わった (転倒 / 時間切れ)
        self.metrics["lost_after_touch"] = ((self.resolved_at >= 0.0) & ~resolved & ~conceded).float()

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        dev = self.device
        pos = self.robot.data.root_pos_w[env_ids, :2]
        heading = self.robot.data.heading_w[env_ids]
        self.line_pos_w[env_ids] = pos
        self.line_heading[env_ids] = heading

        speed = torch.empty(n, device=dev).uniform_(*cfg.speed_range)
        tta = torch.empty(n, device=dev).uniform_(*cfg.time_to_arrival_range)
        dist = (speed * tta).clamp(*cfg.distance_range)
        angle = heading + torch.empty(n, device=dev).uniform_(-cfg.approach_angle, cfg.approach_angle)
        spawn = pos + dist.unsqueeze(1) * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
        # 狙い: ロボット初期位置から左右 ±target_lateral の点
        lateral = torch.empty(n, device=dev).uniform_(-cfg.target_lateral, cfg.target_lateral)
        target = pos + lateral.unsqueeze(1) * torch.stack([-torch.sin(heading), torch.cos(heading)], dim=1)
        direction = target - spawn
        direction = direction / torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
        vel_xy = speed.unsqueeze(1) * direction
        bounce = torch.rand(n, device=dev) < cfg.bounce_prob
        vz = torch.where(bounce, torch.empty(n, device=dev).uniform_(0.0, cfg.bounce_vz), torch.zeros(n, device=dev))

        z = self._env.scene.env_origins[env_ids, 2] + cfg.ball_radius
        root_pose = torch.zeros(n, 7, device=dev)
        root_pose[:, :2] = spawn
        root_pose[:, 2] = z
        root_pose[:, 3] = 1.0  # 単位クォータニオン (w, x, y, z)
        root_vel = torch.zeros(n, 6, device=dev)
        root_vel[:, :2] = vel_xy
        root_vel[:, 2] = vz
        # 転がり整合の角速度 ω = (ẑ × v) / r (滑らずに転がり始める)
        root_vel[:, 3] = -vel_xy[:, 1] / cfg.ball_radius
        root_vel[:, 4] = vel_xy[:, 0] / cfg.ball_radius
        self.ball.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.metrics["launch_speed"][env_ids] = speed
        self.launch_lateral[env_ids] = lateral
        self.resolved_at[env_ids] = -1.0
        self.resolve_kind[env_ids] = 0

    def _update_command(self):
        # 阻止のラッチ (毎 step、終了判定の後に呼ばれる → 次 step の終了判定が見る)
        out_wide, away, stopped = self.resolve_parts()
        first = (self.resolved_at < 0.0) & ~self.conceded() & (out_wide | away | stopped)
        self.resolved_at[first] = self.elapsed()[first]
        kind = torch.where(out_wide, 1, torch.where(away, 2, 3))
        self.resolve_kind[first] = kind[first]

        rel = self.line_pos_w - self.robot.data.root_pos_w[:, :2]
        h = self.robot.data.heading_w
        c, s = torch.cos(h), torch.sin(h)
        err = self.line_heading - h
        self.cmd_buf[:, 0] = c * rel[:, 0] + s * rel[:, 1]
        self.cmd_buf[:, 1] = -s * rel[:, 0] + c * rel[:, 1]
        self.cmd_buf[:, 2] = torch.cos(err)
        self.cmd_buf[:, 3] = torch.sin(err)


@configclass
class BallLaunchCommandCfg(CommandTermCfg):
    class_type: type = BallLaunchCommand

    asset_name: str = MISSING
    ball_name: str = "soccer_ball"
    feet_body_names: str = ".*_foot_link"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """エピソード中は再射出しない (reset 時のみ)。"""

    speed_range: tuple[float, float] = (0.4, 6.0)
    """射出速度 [m/s]。"""
    time_to_arrival_range: tuple[float, float] = (0.8, 3.0)
    """到達までの時間 [s]。射出距離 = 速度 × これ (distance_range にクランプ)。"""
    distance_range: tuple[float, float] = (1.0, 5.0)
    """射出距離 [m] のクランプ。"""
    approach_angle: float = math.pi / 3
    """射出位置の方位 (ロボット初期進行方向からの角度) の範囲 ±[rad]。"""
    target_lateral: float = 1.0
    """狙い点のロボット初期位置からの左右オフセット ±[m]。"""
    bounce_prob: float = 0.3
    """上向き初速を与える (バウンドする) 確率。"""
    bounce_vz: float = 2.0
    """上向き初速の上限 [m/s] (2.0 で最高到達 ≈ 0.2 m)。"""
    ball_radius: float = 0.11

    behind_x: float = 0.3
    """ライン後方 [m]: これより後ろに入ったら失点判定。"""
    goal_half_width: float = 1.6
    """ゴール半幅 [m]: ライン後方でもこの外は失点にしない (横に逸れた = 阻止)。"""
    away_speed: float = 0.3
    """ライン前方でこの前向き速度以上なら「前に弾いた」= 阻止。"""
    stop_speed: float = 0.1
    """この速度未満 (かつ失点していない) なら「止めた」= 阻止。"""
    resolve_hold_s: float = 2.5
    """最初に阻止条件を満たしてから、転倒・失点せずにこの秒数立ち続けたら阻止 (終了)。"""


# ---------------------------------------------------------------------------
# 終了条件
# ---------------------------------------------------------------------------
def ball_conceded(env: "ManagerBasedRLEnv", command_name: str = "ball_launch") -> torch.Tensor:
    return env.command_manager.get_term(command_name).conceded()


def ball_resolved(env: "ManagerBasedRLEnv", command_name: str = "ball_launch") -> torch.Tensor:
    return env.command_manager.get_term(command_name).resolved()


# ---------------------------------------------------------------------------
# 報酬
# ---------------------------------------------------------------------------
def _intercept(term: BallLaunchCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """(接近中か, ボールの予測ライン到達点 y)。"""
    x, y, vx, vy = term.ball_line_state()
    approaching = (vx < -0.1) & (x > 0.0)
    return approaching, y + vy * (x / (-vx).clamp(min=0.1))


def foot_on_intercept(env: "ManagerBasedRLEnv", std: float = 0.15, command_name: str = "ball_launch") -> torch.Tensor:
    """ボールの予測ライン到達点 (y) に近い足があるほど高い [0, 1] (ボール接近中のみ)。"""
    term: BallLaunchCommand = env.command_manager.get_term(command_name)
    approaching, y_int = _intercept(term)
    d = (term.feet_line_y() - y_int.unsqueeze(1)).abs().min(dim=1).values
    return approaching.float() * torch.exp(-torch.square(d / std))


def body_on_intercept(env: "ManagerBasedRLEnv", std: float = 0.3, command_name: str = "ball_launch") -> torch.Tensor:
    """胴体の横位置 (ライン系 y) がボールの予測到達点に近いほど高い [0, 1] (接近中のみ)。横へ動く動機。"""
    term: BallLaunchCommand = env.command_manager.get_term(command_name)
    approaching, y_int = _intercept(term)
    _, body_y = term.body_line_xy()
    return approaching.float() * torch.exp(-torch.square((body_y - y_int) / std))


def body_forward_l2(env: "ManagerBasedRLEnv", margin: float = 0.2, command_name: str = "ball_launch") -> torch.Tensor:
    """胴体がゴールラインより margin [m] 以上前に出た分の二乗 (ボールへ突っ込む動きを抑える)。"""
    body_x, _ = env.command_manager.get_term(command_name).body_line_xy()
    return torch.square((body_x - margin).clamp(min=0.0))


def joint_residual_l2(env: "ManagerBasedRLEnv", action_name: str = "joint_residual") -> torch.Tensor:
    """関節残差 (歩行 expert からのずれ) の L2。事前分布 (expert) に留める引力。"""
    return torch.sum(torch.square(env.action_manager.get_term(action_name).raw_actions), dim=1)
