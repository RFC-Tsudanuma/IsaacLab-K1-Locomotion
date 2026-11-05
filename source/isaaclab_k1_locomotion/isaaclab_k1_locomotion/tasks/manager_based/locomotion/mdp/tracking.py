# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""経路追従 + 姿勢到達 / 見ながら到達タスク (Isaac-Tracking-K1) の MDP 部品 (2026-11-05)。

凍結した歩行 expert・回転 (pivot) expert・ゲート網 (frozen_policies/best_gating) の上で、
上位方策が 5 次元 ``[vx, vy, ωz, Δψ, mode]`` を出す。設計は rl_traking.md を参照。

* action 項 ``walk_command`` (`WalkCommandAction`, dim=5) が上位の出力を保持する。
* :class:`ExternalTransitionCommand` (``base_velocity``) がそれを読み、mode (0=walk / 1=turn) の
  切替・pivot 目標角 Δψ のラッチ・最小滞在時間・切替時の履歴コマンド枠ゼロ埋めを行う。
  歩行 / 回転 expert と ゲートが見るコマンドの意味は `TransitionCommand` と同じ
  (walk: (vx, vy, ωz) / turn: [0, 0, 残り角/π])。
* :class:`TrackingGoalCommand` (``tracking``) が経由点 + ゴール + 見つめる対象を生成する。
  task_mode 0 = ゴール姿勢 (xy + yaw) に到達、1 = 見つめる対象を首可動域 (±90°) 内に保ちながら
  ゴール位置に到達 (向きは自由)。

上位の出力は env に 1 step 遅れて反映される (方策が出した値を次の step のコマンドとして
expert が見る)。C++ 側も「tick t の上位出力を tick t+1 のコマンドに使う」こと。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .commands import TransitionCommand, TransitionCommandCfg
from .waypoint import WalkCommandAction, WaypointCommand, WaypointCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

TRACK_ACTION_DIM = 5
"""上位の出力: [vx, vy, ωz, Δψ 残差 (pivot 目標 = 既定目標 + 残差), mode (>0.5 で turn)]。

既定目標 (`TrackingGoalCommand.pivot_default`): look では見つめる対象の方位、pose では全経由点通過後は
ゴール yaw、それ以前は次の目標点の方位。v1 (残差なし、Δψ を一から学習) では乱数の Δψ で pivot が
役に立たず、方策が pivot を使わなくなった (turn 時間 47% → 2%、iter 300) ため残差方式にした。"""

TASK_POSE = 0
TASK_LOOK = 1


# ---------------------------------------------------------------------------
# 下位へのコマンド: 上位の出力で駆動する TransitionCommand
# ---------------------------------------------------------------------------
class ExternalTransitionCommand(TransitionCommand):
    """`TransitionCommand` の自律遷移を外し、action 項 ``walk_command`` の値で mode / コマンドを決める。

    * 切替は最小滞在時間 ``min_dwell_s`` を満たした env だけ受け付ける (mode のばたつき防止。
      切替のたびに履歴コマンド枠が 0 埋めされるので、ばたつくと履歴が壊れ続ける)。
    * turn 突入時に ``Δψ = 既定目標 (tracking コマンドの pivot_default) + 上位の残差`` を現在 yaw 基準で
      ラッチし、turn 中は固定目標として残り角を渡す。
    * 早期打ち切り (turn_early_exit_rad) / yaw assist は使わない (上位が自分で歩き出す)。
    """

    cfg: "ExternalTransitionCommandCfg"

    def __init__(self, cfg: "ExternalTransitionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.dwell = torch.zeros(self.num_envs, device=self.device)
        """現在のモードに入ってからの経過時間 [s]。"""
        # turn から始まる env の目標角を既定目標にそろえるためのフラグ (reset 時点では tracking 項が
        # まだ再サンプルされていないので、最初の compute で設定する)
        self._init_turn_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, dt: float):
        self._update_metrics()
        self.switched[:] = False
        self.dwell += dt

        act = self._env.action_manager.get_term(self.cfg.action_name).processed_actions
        if self._init_turn_pending.any():
            # turn 始まりのエピソード: 目標角 = 既定目標 (pivot の「お手本」状態分布。v2 で pivot が
            # 使われなかったので、pivot 中の価値を正しく学ばせるために入れた)
            ids = self._init_turn_pending.nonzero(as_tuple=False).flatten()
            default = self._env.command_manager.get_term(self.cfg.goal_command_name).pivot_default()[ids]
            self.turn.heading_target[ids] = math_utils.wrap_to_pi(self.robot.data.heading_w[ids] + default)
            self._init_turn_pending[:] = False
        # turn の到達メトリクス (turn_success_rate / turn_settle_time_s) は親と同じ定義で積む
        turn = self.is_turn
        self._turn_elapsed = torch.where(turn, self._turn_elapsed + dt, self._turn_elapsed)
        newly_settled = turn & (self.turn.heading_error.abs() < float(self.turn.cfg.success_threshold)) & ~self._turn_settled
        self._turn_settled |= newly_settled
        self._settle_sum += torch.where(newly_settled, self._turn_elapsed, torch.zeros_like(self._turn_elapsed))
        self._settle_cnt += newly_settled.float()

        want_turn = act[:, 4] > 0.5
        can_switch = self.dwell >= float(self.cfg.min_dwell_s)
        to_turn = self.is_walk & want_turn & can_switch
        to_walk = self.is_turn & ~want_turn & can_switch
        if to_turn.any():
            ids = to_turn.nonzero(as_tuple=False).flatten()
            self._enter_turn(ids, count_switch=True)
            # _enter_turn が引いた乱数目標を「既定目標 + 上位の残差」で上書き (ラッチ)
            default = self._env.command_manager.get_term(self.cfg.goal_command_name).pivot_default()[ids]
            self.turn.heading_target[ids] = math_utils.wrap_to_pi(self.robot.data.heading_w[ids] + default + act[ids, 3])
        if to_walk.any():
            self._enter_walk(to_walk.nonzero(as_tuple=False).flatten(), count_switch=True, resample_walk=False)
        self.dwell[self.switched] = 0.0

        self.walk.vel_command_b[:] = act[:, :3]
        self.turn._update_command()
        self._update_command()
        if self.cfg.zero_command_history_on_switch and self.switched.any():
            self._zero_command_history(self.switched.nonzero(as_tuple=False).flatten())

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = self._to_index_tensor(env_ids)
        self.dwell[ids] = 0.0
        self._init_turn_pending[ids] = self.is_turn[ids]
        # 親が歩行生成器の乱数コマンドを引くので、リセット直後は停止指令に戻す (action 項も 0)
        self.walk.vel_command_b[ids] = 0.0
        self._update_command()


@configclass
class ExternalTransitionCommandCfg(TransitionCommandCfg):
    class_type: type = ExternalTransitionCommand

    action_name: str = "walk_command"
    """上位の出力を保持する action 項 (dim=5)。"""

    min_dwell_s: float = 0.3
    """モード切替後、次の切替を受け付けるまでの最小滞在時間 [s]。C++ 側と同値にする。"""

    goal_command_name: str = "tracking"
    """pivot の既定目標 (`TrackingGoalCommand.pivot_default`) を読むコマンド項。"""


def lowlevel_state(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """上位の観測: [mode one-hot (2), 滞在時間 (clip 1 s) (1), turn 中の残り角 (1)]。"""
    term: ExternalTransitionCommand = env.command_manager.get_term(command_name)
    one_hot = F.one_hot(term.mode, num_classes=2).float()
    remaining = torch.where(term.is_turn, term.heading_error, torch.zeros_like(term.heading_error))
    return torch.cat([one_hot, term.dwell.clamp(max=1.0).unsqueeze(1), remaining.unsqueeze(1)], dim=1)


# ---------------------------------------------------------------------------
# コマンド項: 経由点 + ゴール + 見つめる対象
# ---------------------------------------------------------------------------
class TrackingGoalCommand(WaypointCommand):
    """`WaypointCommand` に task_mode (姿勢到達 / 見ながら到達) と見つめる対象を足したもの。

    通過半径は 1 値 (fast / strict は廃止)。観測 (base yaw 系, 2K+10 次元):
    ``[次の K 点 xy (2K), ゴール xy (2), ゴール yaw 誤差 cos/sin (2, look では 0),
    見つめる対象 xy (2, pose では 0), task_mode one-hot (2), pivot 既定目標 cos/sin (2)]``。
    """

    cfg: "TrackingGoalCommandCfg"

    def __init__(self, cfg: "TrackingGoalCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n, k = self.num_envs, cfg.max_waypoints
        self.task_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.look_at_w = torch.zeros(n, 2, device=self.device)
        self.cmd_buf = torch.zeros(n, 2 * k + 10, device=self.device)
        self._look_steps = torch.zeros(n, device=self.device)
        self._look_violations = torch.zeros(n, device=self.device)
        self.prev_heading_err = torch.zeros(n, device=self.device)  # heading_progress 報酬のポテンシャル
        for name in ("look_at_err", "look_at_violation_frac"):
            self.metrics[name] = torch.zeros(n, device=self.device)

    def is_look(self) -> torch.Tensor:
        return self.task_mode == TASK_LOOK

    def look_at_err(self) -> torch.Tensor:
        """機体 yaw から見つめる対象への方位角誤差 [rad] (±π)。"""
        d = self.look_at_w - self._pos()
        return math_utils.wrap_to_pi(torch.atan2(d[:, 1], d[:, 0]) - self.robot.data.heading_w)

    def pivot_default(self) -> torch.Tensor:
        """pivot の既定目標 (現在 yaw からの相対角 [rad]): look = 見つめる対象の方位、
        pose = 全経由点通過後はゴール yaw 誤差、それ以前は次の目標点の方位。"""
        target = self.points_w[self._env_idx, self.target_idx] - self._pos()
        to_target = math_utils.wrap_to_pi(torch.atan2(target[:, 1], target[:, 0]) - self.robot.data.heading_w)
        pose_default = torch.where(self.all_passed(), self.goal_yaw_err(), to_target)
        return torch.where(self.is_look(), self.look_at_err(), pose_default)

    def heading_err(self) -> torch.Tensor:
        """向きの誤差 [rad] (heading_progress 報酬用): look = 見つめる対象への方位誤差、
        pose = 全経由点通過後のゴール yaw 誤差 (それ以前は 0 = 接近中の向きは自由)。"""
        pose = torch.where(self.all_passed(), self.goal_yaw_err().abs(), torch.zeros_like(self.goal_yaw_w))
        return torch.where(self.is_look(), self.look_at_err().abs(), pose)

    def _update_metrics(self):
        super()._update_metrics()
        look = self.is_look()
        err = self.look_at_err().abs()
        self._look_steps += look.float()
        self._look_violations += (look & (err > self.cfg.look_at_limit)).float()
        self.metrics["goal_yaw_err"] = torch.where(look, torch.zeros_like(err), self.metrics["goal_yaw_err"])
        self.metrics["look_at_err"] = torch.where(look, err, torch.zeros_like(err))
        self.metrics["look_at_violation_frac"] = self._look_violations / self._look_steps.clamp(min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        self.pass_radius[env_ids] = self.cfg.pass_radius
        self.task_mode[env_ids] = (torch.rand(n, device=self.device) < self.cfg.look_prob).long()
        # 見つめる対象: 経路 (現在位置〜ゴール) の中点から距離 look_at_range、方向一様。
        # ただし経路上の点 (現在位置・経由点・ゴール) から look_at_clearance 未満は引き直す:
        # 対象の真上を通ると方位が瞬時に 180° 反転して違反が避けられず、方策が経由点の手前で
        # 立ち止まる (v2 評価: look の未到達 2 % の大半が経由点で停止)。
        mid = 0.5 * (self._pos()[env_ids] + self.goal_w()[env_ids])
        pts = torch.cat([self._pos()[env_ids].unsqueeze(1), self.points_w[env_ids]], dim=1)  # (n, k+2, 2)
        look = torch.zeros(n, 2, device=self.device)
        todo = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(8):
            if not todo.any():
                break
            r = torch.empty(n, device=self.device).uniform_(*self.cfg.look_at_range)
            th = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
            cand = mid + r.unsqueeze(1) * torch.stack([torch.cos(th), torch.sin(th)], dim=1)
            look = torch.where(todo.unsqueeze(1), cand, look)
            clear = torch.norm(pts - look.unsqueeze(1), dim=-1).min(dim=1).values >= self.cfg.look_at_clearance
            todo &= ~clear
        self.look_at_w[env_ids] = look
        self._look_steps[env_ids] = 0.0
        self._look_violations[env_ids] = 0.0
        self.prev_heading_err[env_ids] = self.heading_err()[env_ids]

    def _update_command(self):
        pos = self._pos()
        target = self.points_w[self._env_idx, self.target_idx]
        near = torch.norm(pos - target, dim=1) < self.pass_radius
        self.target_idx += (near & (self.target_idx < self.num_waypoints)).long()

        look = self.is_look()
        yaw_ok = look | (self.goal_yaw_err().abs() < self.cfg.goal_yaw_tolerance)
        at_goal = self.all_passed() & (self.goal_dist() < self.cfg.goal_pos_tolerance) & yaw_ok
        newly = at_goal & ~self.reached_goal
        self.time_to_goal[newly] = self._env.episode_length_buf[newly].float() * self._env.step_dt
        self.reached_goal |= at_goal

        k = self.cfg.max_waypoints
        idx = (self.target_idx.unsqueeze(1) + self._window.unsqueeze(0)).clamp(max=k)
        sel = torch.gather(self.points_w, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        rel = torch.cat([sel, self.goal_w().unsqueeze(1), self.look_at_w.unsqueeze(1)], dim=1) - pos.unsqueeze(1)
        heading = self.robot.data.heading_w
        c, s = torch.cos(heading).unsqueeze(1), torch.sin(heading).unsqueeze(1)
        rel_b = torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], dim=-1)
        yaw_err = self.goal_yaw_err()
        look_f = look.float().unsqueeze(1)
        pose_f = 1.0 - look_f
        pivot = self.pivot_default()
        self.cmd_buf[:] = torch.cat(
            [
                rel_b[:, : k + 1].flatten(1),
                pose_f * torch.cos(yaw_err).unsqueeze(1),
                pose_f * torch.sin(yaw_err).unsqueeze(1),
                look_f * rel_b[:, k + 1],
                pose_f,
                look_f,
                torch.cos(pivot).unsqueeze(1),
                torch.sin(pivot).unsqueeze(1),
            ],
            dim=1,
        )


@configclass
class TrackingGoalCommandCfg(WaypointCommandCfg):
    class_type: type = TrackingGoalCommand

    pass_radius: float = 0.3
    """経由点の通過半径 [m] (旧 fast 0.6 / strict 0.15 の中間、2026-11-05 ユーザー決定)。"""
    look_prob: float = 0.5
    """task_mode 1 (見ながら到達) になる確率。"""
    look_at_range: tuple[float, float] = (1.5, 5.0)
    """見つめる対象の、経路中点からの距離 [m]。"""
    look_at_limit: float = math.pi / 2
    """首の可動域 [rad]。見つめる対象への yaw 誤差がこれを超えると違反。"""
    look_at_clearance: float = 1.0
    """見つめる対象と経路上の点 (現在位置・経由点・ゴール) の最小距離 [m]。"""

    goal_pos_tolerance: float = 0.1
    goal_yaw_tolerance: float = 0.1


# ---------------------------------------------------------------------------
# 報酬
# ---------------------------------------------------------------------------
def track_goal_position_exp(env: "ManagerBasedRLEnv", std: float, command_name: str = "tracking") -> torch.Tensor:
    """全経由点通過後、ゴール位置に近いほど高い。look では見つめ誤差が可動域内のときだけ。"""
    term: TrackingGoalCommand = env.command_manager.get_term(command_name)
    gate = term.all_passed() & (~term.is_look() | (term.look_at_err().abs() <= term.cfg.look_at_limit))
    return gate.float() * torch.exp(-torch.square(term.goal_dist() / std))


def track_goal_heading_exp(
    env: "ManagerBasedRLEnv", std_yaw: float, std_pos: float, command_name: str = "tracking"
) -> torch.Tensor:
    """pose モードのみ: 全経由点通過後、ゴール近傍でゴール向きに合っているほど高い。"""
    term: TrackingGoalCommand = env.command_manager.get_term(command_name)
    gate = (term.all_passed() & ~term.is_look()).float() * torch.exp(-torch.square(term.goal_dist() / std_pos))
    return gate * torch.exp(-torch.square(term.goal_yaw_err() / std_yaw))


def look_at_penalty(
    env: "ManagerBasedRLEnv", deadzone: float = math.pi / 3, command_name: str = "tracking"
) -> torch.Tensor:
    """look モードのみ: |誤差| ≤ deadzone で 0、deadzone〜可動域で 2 乗増加、可動域超で +2 の段差。"""
    term: TrackingGoalCommand = env.command_manager.get_term(command_name)
    err = term.look_at_err().abs()
    limit = float(term.cfg.look_at_limit)
    ramp = torch.square(((err - deadzone) / (limit - deadzone)).clamp(min=0.0, max=1.0))
    out = torch.where(err > limit, ramp + 2.0, ramp)
    return term.is_look().float() * out


def heading_progress(env: "ManagerBasedRLEnv", command_name: str = "tracking") -> torch.Tensor:
    """向き誤差 (`TrackingGoalCommand.heading_err`) の減少量 [rad/step] (ポテンシャル形式)。

    pivot (180° を 0.3 s) は歩行の ωz 1.0 rad/s より桁違いに速く向きを合わせられるのに、v2/v3 の方策は
    pivot を使わなかった (3 %)。向きを早く合わせること自体に密な報酬を与えて pivot の価値を見えやすくする。
    look では見つめる対象へ向くこと、pose では全経由点通過後にゴール yaw へ向くことが元の目的に含まれるので
    最適方策を歪めない (接近中の向きには掛けない)。"""
    term: TrackingGoalCommand = env.command_manager.get_term(command_name)
    err = term.heading_err()
    # pose で全経由点を通過した瞬間はポテンシャルが 0 → |yaw 誤差| に跳ぶので、その step は 0 にする
    delta = torch.where(term.prev_heading_err > 0.0, term.prev_heading_err - err, torch.zeros_like(err))
    term.prev_heading_err = err
    return delta


def track_command_rate_l2(env: "ManagerBasedRLEnv", action_name: str = "walk_command") -> torch.Tensor:
    """歩行コマンド (先頭 3 次元) の 1 step 変化量 ||c_t − c_{t−1}||²。Δψ / mode は含めない。"""
    term: WalkCommandAction = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(term.raw_actions[:, :3] - term.prev_actions[:, :3]), dim=1)


def mode_switch(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """この step で walk ⇄ turn を切り替えた env に 1。"""
    term: ExternalTransitionCommand = env.command_manager.get_term(command_name)
    return term.switched.float()
