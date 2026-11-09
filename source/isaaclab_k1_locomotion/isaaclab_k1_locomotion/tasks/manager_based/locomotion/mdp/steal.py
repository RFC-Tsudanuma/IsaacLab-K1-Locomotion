# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール奪取タスク (Isaac-Steal-K1) の MDP 部品 (2026-11-03)。

円柱障害物 (敵ロボット想定) の周囲 ``ball_radius_range`` 以内 (``ball_side_angle`` の弧) に静止ボールを置き、凍結歩行 expert +
関節残差の方策 (agents/steal_policy.py) でボールへ近づき、足で触れて敵から離す。
:class:`StealCommand` が各エピソード開始時に障害物とボールを配置し、

* **lost** (失敗): ボールが障害物に触れた (中心距離 < 障害物半径 + ボール半径 + margin)
* **no_foot** (不成立): **足で触れないまま** ボールが ``steal_radius`` の外へ出た (脛・胴体で押し出すのは不可。
  2026-11-03 v2 の評価で足接触は 27% しか無かったので追加)。lost と同じ −200 にすると新規学習が
  「ボールに触らない」方策 (timeout 95%) に落ちたので、別の終了条件にして弱い罰則にする (v5)
* **stolen** (成功): 足 (接触センサ ``contact_balls_*``) でボールに触れた後、ボールが障害物中心から
  ``steal_radius`` の外へ出て ``hold_s`` 秒、転倒も lost もせずに立ち続けた
  (Block と同じラッチ方式: 触った瞬間に終わらせない)

を判定する。報酬は終端 (is_terminated_term) が主で、shaping は胴体→ボールの接近ポテンシャルと
ボールが障害物から離れる速度、時間罰則のみ。

方向指定 (``directed=True``, Isaac-StealDir-K1, 2026-11-05): エピソード開始時にボールを出す目標方向を
ロボット初期進行方向基準の 前 / 左 / 後 / 右 から 1 つ抽選し (ボール→障害物方向から ``target_min_angle``
未満の候補は除外 = 敵のいる側へは出さない)、ボールが ``steal_radius`` を出た時点の変位が目標方向から
``target_tolerance`` 以内なら奪取、外れていれば **wrong_dir** (不成立) で終了する。方策は目標方向を
base yaw 座標系の単位ベクトルとして観測する (:func:`steal_target_dir`)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StealCommand(CommandTerm):
    """障害物 + ボールを配置し、奪取 / 失敗を判定するコマンド項。

    観測 (``command``, base yaw 座標系): ``[障害物中心の相対位置 xy (2)]``。
    """

    cfg: "StealCommandCfg"

    def __init__(self, cfg: "StealCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self.obstacle: RigidObject = env.scene[cfg.obstacle_name]
        self.foot_sensors = [env.scene[name] for name in cfg.foot_sensor_names]
        self._feet_idx = self.robot.find_bodies(cfg.feet_body_names)[0]
        n = self.num_envs
        self.foot_touched = torch.zeros(n, dtype=torch.bool, device=self.device)  # このエピソードで足がボールに触れた
        self.cmd_buf = torch.zeros(n, 2, device=self.device)
        self.stolen_at = torch.full((n,), -1.0, device=self.device)  # 最初に圏外へ出た時刻 [s] (未達 −1)
        self.prev_body_dist = torch.zeros(n, device=self.device)  # 胴体→ボール距離 (接近ポテンシャル用)
        self.target_dir_w = torch.zeros(n, 2, device=self.device)  # 目標方向 (world 単位ベクトル)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=self.device)  # 0 前 / 1 左 / 2 後 / 3 右 (初期進行方向基準)
        self.ball_start_xy = torch.zeros(n, 2, device=self.device)
        self.ball_side0 = torch.zeros(n, device=self.device)  # 配置時のボール方位 (障害物→ロボット方向からの角度) [rad] 評価用
        for name in ("stolen", "lost", "lost_no_foot", "lost_after_touch", "time_to_steal", "wrong_dir"):
            self.metrics[name] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.cmd_buf

    # -- 幾何 ---------------------------------------------------------------------
    def ball_xy(self) -> torch.Tensor:
        return self.ball.data.root_pos_w[:, :2]

    def obstacle_xy(self) -> torch.Tensor:
        return self.obstacle.data.root_pos_w[:, :2]

    def ball_obstacle_dist(self) -> torch.Tensor:
        return torch.norm(self.ball_xy() - self.obstacle_xy(), dim=1)

    def body_ball_dist(self) -> torch.Tensor:
        return torch.norm(self.ball_xy() - self.robot.data.root_pos_w[:, :2], dim=1)

    def foot_ball_dist(self) -> torch.Tensor:
        """最寄りの足 (foot link 原点) とボール中心の距離 (N,)。"""
        feet = self.robot.data.body_link_pos_w[:, self._feet_idx, :]
        return torch.norm(feet - self.ball.data.root_pos_w.unsqueeze(1), dim=-1).min(dim=1).values

    def ball_away_speed(self) -> torch.Tensor:
        """ボールが障害物から遠ざかる速度 (半径方向成分, 負は 0)。"""
        rel = self.ball_xy() - self.obstacle_xy()
        rel = rel / torch.norm(rel, dim=1, keepdim=True).clamp(min=1e-6)
        return torch.sum(self.ball.data.root_lin_vel_w[:, :2] * rel, dim=1).clamp(min=0.0)

    def elapsed(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    def target_dir_b(self) -> torch.Tensor:
        """目標方向を base yaw 座標系で (N, 2)。"""
        h = self.robot.data.heading_w
        c, s = torch.cos(h), torch.sin(h)
        d = self.target_dir_w
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=1)

    def dir_ok(self) -> torch.Tensor:
        """ボールの初期位置からの変位が目標方向から target_tolerance 以内 (変位ゼロは False)。"""
        disp = self.ball_xy() - self.ball_start_xy
        return torch.sum(disp * self.target_dir_w, dim=1) >= torch.norm(disp, dim=1) * math.cos(self.cfg.target_tolerance)

    def foot_contact_now(self) -> torch.Tensor:
        """いずれかの足がボールに触れている (接触センサのフィルタ済み力 > 閾値)。"""
        hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for sensor in self.foot_sensors:
            f = sensor.data.force_matrix_w
            if f is not None:
                hit |= torch.norm(f, dim=-1).amax(dim=(1, 2)) > self.cfg.foot_force_threshold
        return hit

    # -- 判定 ---------------------------------------------------------------------
    def lost_obstacle(self) -> torch.Tensor:
        return self.ball_obstacle_dist() < self.cfg.obstacle_radius + self.cfg.ball_radius + self.cfg.touch_margin

    def lost_no_foot(self) -> torch.Tensor:
        return (self.ball_obstacle_dist() > self.cfg.steal_radius) & ~self.foot_touched

    def lost(self) -> torch.Tensor:
        return self.lost_obstacle()

    def wrong_dir(self) -> torch.Tensor:
        """足で触れた後、目標方向から外れて圏外へ出た (directed のときだけ終了条件に使う)。"""
        return self.foot_touched & (self.ball_obstacle_dist() > self.cfg.steal_radius) & (self.stolen_at < 0.0) & ~self.dir_ok()

    def stolen(self) -> torch.Tensor:
        """成功 (終了条件): 最初に圏外へ出てから hold_s 秒、失敗せずに経過した。"""
        held = (self.stolen_at >= 0.0) & (self.elapsed() - self.stolen_at >= self.cfg.hold_s)
        return held & ~self.lost()

    # -- CommandTerm hooks ----------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # 終了した step の状態でメトリクスを確定させてから配置し直す (基底は 1 step 前の値をログする)
        self._update_metrics()
        return super().reset(env_ids)

    def _update_metrics(self):
        stolen, lost = self.stolen(), self.lost()
        self.metrics["stolen"] = stolen.float()
        self.metrics["lost"] = lost.float()
        self.metrics["lost_no_foot"] = self.lost_no_foot().float()
        self.metrics["lost_after_touch"] = ((self.stolen_at >= 0.0) & ~stolen & ~lost).float()
        self.metrics["time_to_steal"] = torch.where(stolen, self.stolen_at, torch.zeros_like(self.stolen_at))
        self.metrics["wrong_dir"] = self.wrong_dir().float()

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        cfg, dev = self.cfg, self.device
        pos = self.robot.data.root_pos_w[env_ids, :2]
        heading = self.robot.data.heading_w[env_ids]
        z0 = self._env.scene.env_origins[env_ids, 2]

        # 障害物: ロボット前方 ±bearing、距離 distance_range
        dist = torch.empty(n, device=dev).uniform_(*cfg.distance_range)
        bearing = heading + torch.empty(n, device=dev).uniform_(-cfg.bearing, cfg.bearing)
        obs_xy = pos + dist.unsqueeze(1) * torch.stack([torch.cos(bearing), torch.sin(bearing)], dim=1)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = obs_xy
        pose[:, 2] = z0 + cfg.obstacle_height / 2
        pose[:, 3] = 1.0
        self.obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)

        # ボール: 障害物中心からロボット側 (障害物→ロボット方向 ±ball_side_angle) に半径 ball_radius_range
        to_robot = torch.atan2(pos[:, 1] - obs_xy[:, 1], pos[:, 0] - obs_xy[:, 0])
        side = torch.empty(n, device=dev).uniform_(-cfg.ball_side_angle, cfg.ball_side_angle)
        ang = to_robot + side
        self.ball_side0[env_ids] = side
        r_min = max(cfg.ball_radius_range[0], cfg.obstacle_radius + cfg.ball_radius + cfg.touch_margin + 0.03)
        r = torch.empty(n, device=dev).uniform_(r_min, cfg.ball_radius_range[1])
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = obs_xy + r.unsqueeze(1) * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
        pose[:, 2] = z0 + cfg.ball_radius
        pose[:, 3] = 1.0
        self.ball.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        ball_xy = pose[:, :2]

        # 目標方向: 初期進行方向基準の 前 / 左 / 後 / 右 から、ボール→障害物方向に近い候補を除いて抽選
        cand_ang = heading.unsqueeze(1) + torch.arange(4, device=dev) * (math.pi / 2)  # (n, 4)
        cand = torch.stack([torch.cos(cand_ang), torch.sin(cand_ang)], dim=2)  # (n, 4, 2)
        to_obs = obs_xy - ball_xy
        to_obs = to_obs / torch.norm(to_obs, dim=1, keepdim=True).clamp(min=1e-6)
        allowed = torch.sum(cand * to_obs.unsqueeze(1), dim=2) < math.cos(cfg.target_min_angle)  # (n, 4)
        idx = torch.multinomial(allowed.float(), 1).squeeze(1)
        self.target_idx[env_ids] = idx
        self.target_dir_w[env_ids] = cand[torch.arange(n, device=dev), idx]
        self.ball_start_xy[env_ids] = ball_xy

        self.stolen_at[env_ids] = -1.0
        self.foot_touched[env_ids] = False
        self.prev_body_dist[env_ids] = torch.norm(pose[:, :2] - pos, dim=1)

    def _update_command(self):
        # 足接触の記録と成功のラッチ (毎 step、終了判定の後に呼ばれる → 次 step の終了判定が見る)
        self.foot_touched |= self.foot_contact_now()
        first = (self.stolen_at < 0.0) & ~self.lost_obstacle() & self.foot_touched & (self.ball_obstacle_dist() > self.cfg.steal_radius)
        if self.cfg.directed:
            first &= self.dir_ok()
        self.stolen_at[first] = self.elapsed()[first]

        rel = self.obstacle_xy() - self.robot.data.root_pos_w[:, :2]
        h = self.robot.data.heading_w
        c, s = torch.cos(h), torch.sin(h)
        self.cmd_buf[:, 0] = c * rel[:, 0] + s * rel[:, 1]
        self.cmd_buf[:, 1] = -s * rel[:, 0] + c * rel[:, 1]


@configclass
class StealCommandCfg(CommandTermCfg):
    class_type: type = StealCommand

    asset_name: str = MISSING
    ball_name: str = "soccer_ball"
    obstacle_name: str = "obstacle"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """エピソード中は再配置しない (reset 時のみ)。"""

    distance_range: tuple[float, float] = (1.0, 3.0)
    """ロボット初期位置から障害物中心までの距離 [m]。"""
    bearing: float = math.radians(45.0)
    """障害物の方位 (ロボット初期進行方向からの角度) の範囲 ±[rad]。"""
    obstacle_radius: float = 0.15
    obstacle_height: float = 1.0
    """シーン側の円柱と一致させること (steal_env_cfg)。"""
    ball_radius: float = 0.11
    ball_radius_range: tuple[float, float] = (0.0, 0.5)
    """障害物中心からボールまでの距離 [m]。下限は障害物 + ボール半径 + margin に切り上げる。"""
    ball_side_angle: float = math.radians(75.0)
    """ボールの方位 (障害物→ロボット方向からの角度) の範囲 ±[rad]。π で全周 (裏側も含む)。"""

    touch_margin: float = 0.02
    """ボール中心と障害物中心の距離がこの余裕以内なら「敵に触れた」= 失敗。"""
    foot_sensor_names: tuple[str, ...] = ("contact_balls_right", "contact_balls_left")
    feet_body_names: str = ".*_foot_link"
    """足とボールの接触センサ (dribble シーン由来)。"""
    foot_force_threshold: float = 1.0
    """この力 [N] 以上で「足で触れた」とみなす。"""
    steal_radius: float = 0.8
    """ボールが障害物中心からこの距離の外に出たら奪取 (ラッチ)。"""
    hold_s: float = 1.5
    """ラッチしてから転倒・失敗せずにこの秒数立ち続けたら成功 (終了)。"""

    directed: bool = False
    """True: 目標方向 (前/左/後/右) を抽選し、方向が合ったときだけ奪取とみなす (Isaac-StealDir-K1)。"""
    target_min_angle: float = math.radians(60.0)
    """ボール→障害物方向からこの角度未満の候補は抽選しない (ボール半径 0.31 m・敵+ボール半径 0.26 m で
    直進しても当たらない下限 ≈ 57°)。90° 間隔の 4 候補なので少なくとも 2 候補は残る。"""
    target_tolerance: float = math.radians(45.0)
    """圏外に出た時点の変位と目標方向の角度差がこれ以内なら方向成立。"""


# ---------------------------------------------------------------------------
# 終了条件
# ---------------------------------------------------------------------------
def ball_stolen(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).stolen()


def ball_lost(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).lost()


def ball_no_foot(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).lost_no_foot()


def ball_wrong_dir(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).wrong_dir()


# ---------------------------------------------------------------------------
# 観測
# ---------------------------------------------------------------------------
def steal_target_dir(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """目標方向の単位ベクトル (base yaw 座標系, 2)。"""
    return env.command_manager.get_term(command_name).target_dir_b()


# ---------------------------------------------------------------------------
# 報酬
# ---------------------------------------------------------------------------
def body_ball_progress(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """胴体→ボール距離の減少量 [m/step] (ポテンシャル形式: 総和 = 初期距離 − 最終距離)。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    dist = term.body_ball_dist()
    delta = term.prev_body_dist - dist
    term.prev_body_dist = dist
    return delta


def foot_near_ball(env: "ManagerBasedRLEnv", std: float = 0.2, command_name: str = "steal") -> torch.Tensor:
    """足がボールに近いほど高い [0, 1]。足で触れるまでだけ有効 (触れた後に足元に留まる動機を作らない)。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    return (~term.foot_touched).float() * torch.exp(-torch.square(term.foot_ball_dist() / std))


def ball_away_speed(env: "ManagerBasedRLEnv", max_speed: float = 3.0, command_name: str = "steal") -> torch.Tensor:
    """ボールが障害物から遠ざかる速度 [m/s] (上限 max_speed)。足で触れた後だけ有効 (足以外で押すのは報酬なし)。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    return term.ball_away_speed().clamp(max=max_speed) * term.foot_touched.float()


def ball_target_speed(env: "ManagerBasedRLEnv", max_speed: float = 3.0, command_name: str = "steal") -> torch.Tensor:
    """ボール速度の目標方向成分 [m/s] (負は 0、上限 max_speed)。足で触れた後だけ有効。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    v = torch.sum(term.ball.data.root_lin_vel_w[:, :2] * term.target_dir_w, dim=1)
    return v.clamp(min=0.0, max=max_speed) * term.foot_touched.float()


def trunk_obstacle_contact(env: "ManagerBasedRLEnv", threshold: float = 1.0, sensor_name: str = "contact_trunk_obstacle") -> torch.Tensor:
    """胴体が障害物 (敵) に触れている (フィルタ済み接触力 > threshold) 間 1。"""
    f = env.scene[sensor_name].data.force_matrix_w
    return (torch.norm(f, dim=-1).amax(dim=(1, 2)) > threshold).float()
