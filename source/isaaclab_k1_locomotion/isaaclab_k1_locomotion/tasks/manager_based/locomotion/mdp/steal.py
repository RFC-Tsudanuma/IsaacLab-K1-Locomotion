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


PH_PRE, PH_ON, PH_ACTIVE, PH_OFF, PH_POST = 0, 1, 2, 3, 4
"""試合条件の段階: 起動前 (歩行) / 混合の立ち上げ / 作動 / 解除の混合 / 解除後。"""


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
        self.prev_circ_ang = torch.zeros(n, device=self.device)  # ボールから見たロボットの方位 (回り込み量の罰用)
        self.target_dir_w = torch.zeros(n, 2, device=self.device)  # 目標方向 (world 単位ベクトル)
        self.target_idx = torch.zeros(n, dtype=torch.long, device=self.device)  # 0 前 / 1 左 / 2 後 / 3 右 (初期進行方向基準)
        self.ball_start_xy = torch.zeros(n, 2, device=self.device)
        self.ball_side0 = torch.zeros(n, device=self.device)  # 配置時のボール方位 (障害物→ロボット方向からの角度) [rad] 評価用
        self.bearing0 = torch.zeros(n, device=self.device)  # 配置時の敵の方位 (ロボット初期向きからの角度) [rad] 評価用
        # 動く敵とボール (ドリブル想定, 2026-11-11)。速度は配置時に抽選して world 系で一定 (直進)
        self.move_speed_max = float(cfg.move_speed_max)  # カリキュラムで更新 (steal_move_speed_curriculum)
        self.obstacle_vel = torch.zeros(n, 2, device=self.device)
        self.ball_drive_vel = torch.zeros(n, 2, device=self.device)
        self.ball_driven = torch.zeros(n, dtype=torch.bool, device=self.device)  # ロボットが近づく / 触れるまで駆動
        self.moving_ep = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.follow_offset = torch.zeros(n, 2, device=self.device)  # 解放後〜足で触れるまで: 敵 = ボール + offset
        self.ball_touch_xy = torch.zeros(n, 2, device=self.device)  # 最初に足で触れた時のボール位置 (方向判定の基準)
        # 試合条件 (match_enable, 2026-11-22, STEAL_HANDOFF「次の再学習」): 歩行中からの開始・途中解除の混合、
        # 敵が寄ってくる配置・蹴られるボール、ボール / 敵の観測をノードの代替則と劣化で作る
        self.phase = torch.full((n,), PH_ACTIVE, dtype=torch.long, device=self.device)
        self.w = torch.ones(n, device=self.device)  # 歩行 (0) ↔ 奪取 (1) の混合重み
        self.cmd_ext = torch.zeros(n, 3, device=self.device)  # w < 1 で混ぜる歩行コマンド (base 系)
        self.t_act = torch.zeros(n, device=self.device)  # 奪取の起動時刻 [s]
        self.pre_until = torch.zeros(n, device=self.device)
        self.act_dist = torch.zeros(n, device=self.device)
        self.walk_speed = torch.zeros(n, device=self.device)
        self.walk_turn_gain = torch.zeros(n, device=self.device)
        self.release_at = torch.full((n,), float("inf"), device=self.device)  # 起動からの経過 [s]
        self.release_cmd = torch.zeros(n, 3, device=self.device)
        self.post_since = torch.zeros(n, device=self.device)
        self.contest = torch.zeros(n, dtype=torch.bool, device=self.device)  # 敵がボールへ寄ってくる配置
        self.approaching = torch.zeros(n, dtype=torch.bool, device=self.device)  # 敵がまだボールに届いていない
        self.kick_on_reach = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.kicked = torch.zeros(n, dtype=torch.bool, device=self.device)
        # 観測の劣化 (方策の観測のみ、critic は真値)
        h = int(cfg.obs_delay_steps[1]) + 1
        self._hist_len = h
        self._hist_ptr = 0
        self.hist_ball = torch.zeros(h, n, 4, device=self.device)  # world (x, y, vx, vy)
        self.hist_opp = torch.zeros(h, n, 2, device=self.device)
        self.obs_degrade = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self.seen_ball_w = torch.zeros(n, 2, device=self.device)  # 最後に見えたボール (world)
        self.jump_left = torch.zeros(n, device=self.device)
        self.jump_off = torch.zeros(n, 2, device=self.device)
        self.veldrop_left = torch.zeros(n, device=self.device)
        self.miss_left = torch.zeros(n, device=self.device)
        self.fake_left = torch.zeros(n, device=self.device)
        self.fake_off = torch.zeros(n, 2, device=self.device)
        self.obs_ball_b = torch.zeros(n, 4, device=self.device)  # 方策の観測: ボール位置・速度 (base yaw 系)
        self.obs_opp_b = torch.zeros(n, 2, device=self.device)  # 方策の観測: 敵の位置 (base yaw 系)
        self.opp_virtual = torch.zeros(n, dtype=torch.bool, device=self.device)
        # 観測履歴 (2026-11-22, Block 型 encoder 用): world 系で持ち、観測時に現在の base yaw 系へ直す (ノードは odom で同じことをする)
        T = int(cfg.obs_history)
        self._oh_len, self._oh_ptr = T, 0
        self.oh_obs = torch.zeros(T, n, 8, device=self.device)  # [ボール位置 2, 速度 2, 敵位置 2, ボールが見えている, 敵が仮想]
        self.oh_true = torch.zeros(T, n, 6, device=self.device)  # [ボール位置 2, 速度 2, 敵位置 2] (真値)
        self.oh_teach = torch.zeros(T, n, 8, device=self.device)  # 教師 (劣化なしで学習) が見る履歴 = 劣化なしエピソードの oh_obs と同じ作り
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
        """最初に足で触れた時のボール位置からの変位が目標方向から target_tolerance 以内 (変位ゼロは False)。
        静止ボールでは初期位置と同じ。動くボールではドリブルでの移動分を含めないため触れた位置を基準にする。"""
        disp = self.ball_xy() - self.ball_touch_xy
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

    def ball_out(self) -> torch.Tensor:
        """ボールが「出た」(奪取・方向・足以外の判定に使う)。敵がボールを持っている間は従来どおり敵の中心から steal_radius の外。
        持っていない間 (寄ってくる敵がまだ届いていない / 蹴られた後) は、足で触れた位置 (触れる前は初期位置) から
        free_out_dist 以上動いたら (2026-11-22、試合条件)。"""
        possessed = ~(self.approaching | self.kicked)
        out_poss = self.ball_obstacle_dist() > self.cfg.steal_radius
        ref = torch.where(self.foot_touched.unsqueeze(1), self.ball_touch_xy, self.ball_start_xy)
        out_free = torch.norm(self.ball_xy() - ref, dim=1) > self.cfg.free_out_dist
        return torch.where(possessed, out_poss, out_free)

    def lost_no_foot(self) -> torch.Tensor:
        return self.ball_out() & ~self.foot_touched & ~self.kicked

    def lost(self) -> torch.Tensor:
        return self.lost_obstacle()

    def wrong_dir(self) -> torch.Tensor:
        """足で触れた後、目標方向から外れて圏外へ出た (directed のときだけ終了条件に使う)。"""
        return self.foot_touched & self.ball_out() & (self.stolen_at < 0.0) & ~self.dir_ok()

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

        ids = torch.as_tensor(env_ids, device=dev)
        m = cfg.match_enable
        walkin = (torch.rand(n, device=dev) < cfg.walkin_prob) if m else torch.zeros(n, dtype=torch.bool, device=dev)
        contest = (torch.rand(n, device=dev) < cfg.contest_prob) if m else torch.zeros(n, dtype=torch.bool, device=dev)
        extra = torch.where(walkin, torch.empty(n, device=dev).uniform_(*cfg.walkin_extra_dist), torch.zeros(n, device=dev))

        # 障害物: ロボット前方 ±bearing、距離 distance_range (歩行中開始はその分遠くに置いて歩いて近づく)
        dist = torch.empty(n, device=dev).uniform_(*cfg.distance_range) + extra
        rel_bearing = torch.empty(n, device=dev).uniform_(-cfg.bearing, cfg.bearing)
        self.bearing0[env_ids] = rel_bearing
        bearing = heading + rel_bearing
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
        if contest.any():
            # 敵が寄ってくる配置: ボールはロボットから 0.3〜2.0 m (+ 歩く分)、敵はボールから 0.5〜2.0 m の任意の方向
            c = contest
            d_b = torch.empty(n, device=dev).uniform_(*cfg.contest_ball_dist) + extra
            ball_c = pos + d_b.unsqueeze(1) * torch.stack([torch.cos(bearing), torch.sin(bearing)], dim=1)
            a_e = torch.rand(n, device=dev) * 2 * math.pi
            d_e = torch.empty(n, device=dev).uniform_(*cfg.contest_enemy_dist)
            enemy_c = ball_c + d_e.unsqueeze(1) * torch.stack([torch.cos(a_e), torch.sin(a_e)], dim=1)
            pose[c, :2] = ball_c[c]
            obs_xy = torch.where(c.unsqueeze(1), enemy_c, obs_xy)
            opose = torch.zeros(n, 7, device=dev)
            opose[:, :2] = obs_xy
            opose[:, 2] = z0 + cfg.obstacle_height / 2
            opose[:, 3] = 1.0
            self.obstacle.write_root_pose_to_sim(opose, env_ids=env_ids)
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
        if cfg.target_mode == "robot_lr":
            self._target_lr(ids, pos, ball_xy, obs_xy)
        self.ball_start_xy[env_ids] = ball_xy

        # 動き: 敵は速さ U(0, move_speed_max)・向き一様で直進。ボールは敵の速度 + 最大 ball_rel_speed_max の差
        ids = torch.as_tensor(env_ids, device=dev)
        speed = torch.rand(n, device=dev) * self.move_speed_max
        speed[torch.rand(n, device=dev) < cfg.static_prob] = 0.0  # 静止の敵も一定割合で残す (静止での奪取率の維持)
        ang_v = torch.rand(n, device=dev) * 2 * math.pi
        self.obstacle_vel[ids] = speed.unsqueeze(1) * torch.stack([torch.cos(ang_v), torch.sin(ang_v)], dim=1)
        rel = torch.sqrt(torch.rand(n, device=dev)) * torch.minimum(speed, torch.full_like(speed, cfg.ball_rel_speed_max))
        ang_r = torch.rand(n, device=dev) * 2 * math.pi
        self.ball_drive_vel[ids] = self.obstacle_vel[ids] + rel.unsqueeze(1) * torch.stack([torch.cos(ang_r), torch.sin(ang_r)], dim=1)
        if contest.any():
            # 寄ってくる敵: ボールへ向かって直進 (届いたらドリブル or キック、_move_opponent_and_ball)
            to_ball = ball_xy - obs_xy
            to_ball = to_ball / torch.norm(to_ball, dim=1, keepdim=True).clamp(min=1e-6)
            hi = max(cfg.contest_enemy_speed[1], float(self.move_speed_max))
            v_c = torch.empty(n, device=dev).uniform_(cfg.contest_enemy_speed[0], hi).unsqueeze(1) * to_ball
            self.obstacle_vel[ids] = torch.where(contest.unsqueeze(1), v_c, self.obstacle_vel[ids])
            speed = torch.where(contest, torch.ones_like(speed), speed)  # moving_ep にする
        self.ball_driven[ids] = (speed > 1e-3) & ~contest
        self.moving_ep[ids] = speed > 1e-3
        self.contest[ids] = contest
        self.approaching[ids] = contest
        self.kick_on_reach[ids] = contest & (torch.rand(n, device=dev) < cfg.kick_prob)
        self.kicked[ids] = False
        self.ball_touch_xy[ids] = ball_xy
        if self.ball_driven[ids].any():
            self._write_ball_vel(ids, torch.where(self.ball_driven[ids].unsqueeze(1), self.ball_drive_vel[ids], torch.zeros_like(self.ball_drive_vel[ids])))

        # 起動の段階: 歩行中開始は PRE (w=0 で歩行 expert がボールへ歩く) から、それ以外は最初から作動 (w=1)
        self.phase[ids] = torch.where(walkin, torch.full_like(self.phase[ids], PH_PRE), torch.full_like(self.phase[ids], PH_ACTIVE))
        self.w[ids] = torch.where(walkin, torch.zeros(n, device=dev), torch.ones(n, device=dev))
        self.t_act[ids] = 0.0
        self.pre_until[ids] = torch.empty(n, device=dev).uniform_(*cfg.walkin_time)
        self.act_dist[ids] = torch.empty(n, device=dev).uniform_(*cfg.act_dist)
        self.walk_speed[ids] = torch.empty(n, device=dev).uniform_(*cfg.walkin_speed)
        self.walk_turn_gain[ids] = torch.empty(n, device=dev).uniform_(*cfg.walkin_turn_gain)
        rel_at = torch.empty(n, device=dev).uniform_(*cfg.release_after)
        do_rel = (torch.rand(n, device=dev) < cfg.release_prob) if m else torch.zeros(n, dtype=torch.bool, device=dev)
        self.release_at[ids] = torch.where(do_rel, rel_at, torch.full_like(rel_at, float("inf")))
        a_r = torch.rand(n, device=dev) * 2 * math.pi
        v_r = torch.empty(n, device=dev).uniform_(*cfg.release_move_speed)
        rc = torch.stack([v_r * torch.cos(a_r), v_r * torch.sin(a_r), torch.empty(n, device=dev).uniform_(-1.0, 1.0)], dim=1)
        rc[torch.rand(n, device=dev) < cfg.release_zero_prob] = 0.0
        self.release_cmd[ids] = rc
        self.cmd_ext[ids] = 0.0

        # 観測の劣化と履歴 (遅延用) の初期化
        self.obs_degrade[ids] = (torch.rand(n, device=dev) < cfg.obs_degrade_prob) if m else torch.zeros(n, dtype=torch.bool, device=dev)
        self.delay[ids] = torch.randint(int(cfg.obs_delay_steps[0]), int(cfg.obs_delay_steps[1]) + 1, (n,), device=dev)
        b4 = torch.cat([ball_xy, torch.zeros(n, 2, device=dev)], dim=1)
        self.hist_ball[:, ids] = b4.unsqueeze(0)
        self.hist_opp[:, ids] = obs_xy.unsqueeze(0)
        self.seen_ball_w[ids] = ball_xy
        f_true = torch.cat([ball_xy, torch.zeros(n, 2, device=dev), obs_xy], dim=1)
        self.oh_true[:, ids] = f_true.unsqueeze(0)
        self.oh_obs[:, ids] = torch.cat([f_true, torch.ones(n, 1, device=dev), torch.zeros(n, 1, device=dev)], dim=1).unsqueeze(0)
        self.oh_teach[:, ids] = self.oh_obs[:, ids]
        for buf in (self.jump_left, self.veldrop_left, self.miss_left, self.fake_left):
            buf[ids] = 0.0

        self.stolen_at[env_ids] = -1.0
        self.foot_touched[env_ids] = False
        self.prev_body_dist[env_ids] = torch.norm(pose[:, :2] - pos, dim=1)
        self.prev_circ_ang[env_ids] = torch.atan2(pos[:, 1] - pose[:, 1], pos[:, 0] - pose[:, 0])

    def _target_lr(self, ids: torch.Tensor, pos: torch.Tensor, ball_xy: torch.Tensor, obs_xy: torch.Tensor):
        """目標方向 "robot_lr" (2026-11-24): ロボット→ボールの線から左右どちらかへ target_lateral_range の角度
        (近づいた向きのまま左右へ出す)。ボール→敵から target_min_angle 以内の側は除く。リセット時と、
        歩行中開始では起動時 (ノードが方向を固定する tick) に引く。"""
        cfg, dev, n = self.cfg, self.device, len(ids)
        ref = ball_xy - pos
        off = torch.empty(n, device=dev).uniform_(*cfg.target_lateral_range)
        a = torch.atan2(ref[:, 1], ref[:, 0]).unsqueeze(1) + off.unsqueeze(1) * torch.tensor([1.0, -1.0], device=dev)  # 左, 右
        cand = torch.stack([torch.cos(a), torch.sin(a)], dim=2)  # (n, 2, 2)
        to_obs = obs_xy - ball_xy
        to_obs = to_obs / torch.norm(to_obs, dim=1, keepdim=True).clamp(min=1e-6)
        ok = torch.sum(cand * to_obs.unsqueeze(1), dim=2) < math.cos(cfg.target_min_angle)
        ok[~ok.any(dim=1)] = True
        pick = torch.multinomial(ok.float(), 1).squeeze(1)
        self.target_dir_w[ids] = cand[torch.arange(n, device=dev), pick]
        self.target_idx[ids] = torch.where(pick == 0, 1, 3).to(self.target_idx.dtype)  # 評価表の左 / 右

    def _write_ball_vel(self, ids: torch.Tensor, vel_xy: torch.Tensor):
        """ボールに水平速度を与える (z 速度は保持、角速度は転がり条件 ω = z × v / r)。"""
        v = self.ball.data.root_lin_vel_w[ids].clone()
        v[:, :2] = vel_xy
        w = torch.zeros_like(v)
        w[:, 0] = -vel_xy[:, 1] / self.cfg.ball_radius
        w[:, 1] = vel_xy[:, 0] / self.cfg.ball_radius
        self.ball.write_root_velocity_to_sim(torch.cat([v, w], dim=1), env_ids=ids)

    def _move_opponent_and_ball(self):
        """動くエピソード: (1) ロボットが近づくまで敵は直進・ボールは駆動、(2) 解放後〜足で触れるまで敵はボールに
        一定の offset で付いていく (ボールを保持し続けるドリブラー。自然に離れて no_foot / lost になるのを防ぐ)、
        (3) 足で触れた後は敵はそのまま直進し、ボールは物理のまま。"""
        cfg = self.cfg
        near = (torch.norm(self.ball_xy() - self.robot.data.root_pos_w[:, :2], dim=1) < cfg.ball_release_body_dist) | (
            self.foot_ball_dist() < cfg.ball_release_foot_dist
        )
        # 寄ってくる敵がボールに届いた: キック (任意の方向に 2〜6 m/s) か、ドリブル (以後は従来の動く敵と同じ)
        reach = self.approaching & (self.ball_obstacle_dist() < cfg.contest_reach_dist)
        if reach.any():
            ids = reach.nonzero().flatten()
            self.approaching[ids] = False
            k = self.kick_on_reach[ids] & ~self.foot_touched[ids]
            kids, dids = ids[k], ids[~k]
            if len(kids) > 0:
                a = torch.rand(len(kids), device=self.device) * 2 * math.pi
                s = torch.empty(len(kids), device=self.device).uniform_(*cfg.kick_speed)
                self._write_ball_vel(kids, s.unsqueeze(1) * torch.stack([torch.cos(a), torch.sin(a)], dim=1))
                self.kicked[kids] = True
            if len(dids) > 0:
                self.ball_drive_vel[dids] = self.obstacle_vel[dids]
                self.ball_driven[dids] = ~self.foot_touched[dids]
        released = self.ball_driven & (near | self.foot_touched)
        self.follow_offset[released] = self.obstacle_xy()[released] - self.ball_xy()[released]
        self.ball_driven &= ~released
        follow = self.moving_ep & ~self.ball_driven & ~self.foot_touched & ~self.approaching & ~self.kicked

        if self.moving_ep.any():
            ids = self.moving_ep.nonzero().flatten()
            pose = self.obstacle.data.root_pose_w[ids].clone()
            xy = pose[:, :2] + self.obstacle_vel[ids] * self._env.step_dt
            f = follow[ids]
            xy[f] = self.ball_xy()[ids][f] + self.follow_offset[ids][f]
            pose[:, :2] = xy
            pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            self.obstacle.write_root_pose_to_sim(pose, env_ids=ids)
        if self.ball_driven.any():
            ids = self.ball_driven.nonzero().flatten()
            vel = self.ball_drive_vel[ids].clone()
            # 駆動中に敵へ当たる / 圏外へ流れる (ロボットのせいでない終了) を防ぐ: 帯の外では敵と同じ速度に戻す
            d = self.ball_obstacle_dist()[ids]
            r_min = cfg.obstacle_radius + cfg.ball_radius + cfg.touch_margin + 0.03
            off_band = (d < r_min) | (d > cfg.steal_radius - 0.15)
            vel[off_band] = self.obstacle_vel[ids][off_band]
            self._write_ball_vel(ids, vel)

    def body_ball_b(self) -> torch.Tensor:
        """真のボール位置 (base yaw 系, N×2)。"""
        return self._to_body(self.ball_xy() - self.robot.data.root_pos_w[:, :2])

    def _to_body(self, v_w: torch.Tensor) -> torch.Tensor:
        h = self.robot.data.heading_w
        c, s = torch.cos(h), torch.sin(h)
        return torch.stack([c * v_w[:, 0] + s * v_w[:, 1], -s * v_w[:, 0] + c * v_w[:, 1]], dim=1)

    def _walk_to_ball_cmd(self) -> torch.Tensor:
        """起動前の歩行コマンド: ボールへ向かって上限付近の速さ・旋回込み (試合での起動直前の分布を模す)。"""
        cfg = self.cfg
        b = self.body_ball_b()
        d = b / torch.norm(b, dim=1, keepdim=True).clamp(min=1e-6)
        v = self.walk_speed.unsqueeze(1) * d
        vy = v[:, 1].clamp(-cfg.walk_cmd_limits[1], cfg.walk_cmd_limits[1])
        t = ((vy.abs() - cfg.walk_lateral_knee) / (cfg.walk_cmd_limits[1] - cfg.walk_lateral_knee)).clamp(0.0, 1.0)
        x_lim = cfg.walk_cmd_limits[0] - t * (cfg.walk_cmd_limits[0] - cfg.walk_lateral_x_min)
        vx = torch.maximum(torch.minimum(v[:, 0], x_lim), -x_lim)
        wz = (self.walk_turn_gain * torch.atan2(b[:, 1], b[:, 0])).clamp(-cfg.walk_cmd_limits[2], cfg.walk_cmd_limits[2])
        return torch.stack([vx, vy, wz], dim=1)

    def _update_phase(self):
        """歩行 ↔ 奪取の段階と混合重み w (ノードの steal_step と同じ 0.3 s の線形混合)。"""
        cfg = self.cfg
        dt = self._env.step_dt
        t = self.elapsed()
        pre = self.phase == PH_PRE
        start = pre & ((t >= self.pre_until) | (self.body_ball_dist() <= self.act_dist))
        self.phase[start] = PH_ON
        self.t_act[start] = t[start]
        if cfg.target_mode == "robot_lr" and start.any():
            i = start.nonzero().squeeze(1)
            self._target_lr(i, self.robot.data.root_pos_w[i, :2], self.ball_xy()[i], self.obstacle_xy()[i])
        on = self.phase == PH_ON
        self.w[on] = (self.w[on] + dt / cfg.blend_s).clamp(max=1.0)
        self.phase[on & (self.w >= 1.0)] = PH_ACTIVE
        act = self.phase == PH_ACTIVE
        rel = act & (t - self.t_act >= self.release_at)
        self.phase[rel] = PH_OFF
        off = self.phase == PH_OFF
        self.w[off] = (self.w[off] - dt / cfg.blend_s).clamp(min=0.0)
        to_post = off & (self.w <= 0.0)
        self.phase[to_post] = PH_POST
        self.post_since[to_post] = t[to_post]
        pre = self.phase == PH_PRE
        self.cmd_ext[:] = 0.0
        if pre.any():
            self.cmd_ext[pre] = self._walk_to_ball_cmd()[pre]
        out = (self.phase == PH_OFF) | (self.phase == PH_POST)
        self.cmd_ext[out] = self.release_cmd[out]

    def _update_observations(self):
        """方策が見るボール・敵 (ノードと同じ代替則 + 劣化)。劣化なしのエピソードは真値 (従来と同じ)。"""
        cfg, dev, n = self.cfg, self.device, self.num_envs
        dt = self._env.step_dt
        root = self.robot.data.root_pos_w[:, :2]
        ball_true = torch.cat([self.ball_xy(), self.ball.data.root_lin_vel_w[:, :2]], dim=1)
        self._hist_ptr = (self._hist_ptr + 1) % self._hist_len
        self.hist_ball[self._hist_ptr] = ball_true
        self.hist_opp[self._hist_ptr] = self.obstacle_xy()
        idx = (self._hist_ptr - self.delay) % self._hist_len
        ar = torch.arange(n, device=dev)
        ball_d = self.hist_ball[idx, ar]
        opp_d = self.hist_opp[idx, ar]
        deg = self.obs_degrade
        ball_w = torch.where(deg.unsqueeze(1), ball_d, ball_true)
        opp_w = torch.where(deg.unsqueeze(1), opp_d, self.obstacle_xy())

        # 跳び (自己位置の補正・誤検出): 確率的に 0.2〜1.0 m のオフセットを 0.1〜0.5 s 足す
        start = deg & (self.jump_left <= 0) & (torch.rand(n, device=dev) < cfg.jump_prob)
        if start.any():
            k = int(start.sum())
            a = torch.rand(k, device=dev) * 2 * math.pi
            mag = torch.empty(k, device=dev).uniform_(*cfg.jump_mag)
            self.jump_off[start] = mag.unsqueeze(1) * torch.stack([torch.cos(a), torch.sin(a)], dim=1)
            self.jump_left[start] = torch.empty(k, device=dev).uniform_(*cfg.jump_dur)
        jumping = self.jump_left > 0
        pos_w = ball_w[:, :2] + torch.where(jumping.unsqueeze(1), self.jump_off, torch.zeros_like(self.jump_off))
        self.jump_left = (self.jump_left - dt).clamp(min=0.0)

        # 視野: 首はボールを追う (首 ±neck_limit) + 水平視野 ±fov_half。足元の近距離も見えない
        rel_b = self._to_body(pos_w - root)
        bear = torch.atan2(rel_b[:, 1], rel_b[:, 0])
        head = bear.clamp(-cfg.neck_limit, cfg.neck_limit)
        visible = ((bear - head).abs() <= cfg.fov_half) & (torch.norm(rel_b, dim=1) >= cfg.near_blind_dist)
        visible |= ~deg
        self.seen_ball_w[visible] = pos_w[visible]
        hold = ~visible
        pos_w = torch.where(hold.unsqueeze(1), self.seen_ball_w, pos_w)
        vel_w = ball_w[:, 2:]
        start = deg & (self.veldrop_left <= 0) & (torch.rand(n, device=dev) < cfg.veldrop_prob)
        if start.any():
            self.veldrop_left[start] = torch.empty(int(start.sum()), device=dev).uniform_(*cfg.veldrop_dur)
        vel_zero = hold | (self.veldrop_left > 0)
        vel_w = torch.where(vel_zero.unsqueeze(1), torch.zeros_like(vel_w), vel_w)
        self.veldrop_left = (self.veldrop_left - dt).clamp(min=0.0)
        ball_b = self._to_body(pos_w - root)
        self.obs_ball_b = torch.cat([ball_b, self._to_body(vel_w)], dim=1)

        # 敵: 検出ロボットのうち (観測した) ボールに最も近いもの、ボールから opp_max 以内のみ。無ければ仮想の敵
        opp_b = self._to_body(opp_w - root)
        opp_vis = ((torch.atan2(opp_b[:, 1], opp_b[:, 0]) - head).abs() <= cfg.fov_half) | ~deg
        start = deg & (self.miss_left <= 0) & (torch.rand(n, device=dev) < cfg.opp_miss_prob)
        if start.any():
            self.miss_left[start] = torch.empty(int(start.sum()), device=dev).uniform_(*cfg.opp_miss_dur)
        opp_vis &= self.miss_left <= 0
        self.miss_left = (self.miss_left - dt).clamp(min=0.0)
        d_opp = torch.norm(opp_w - pos_w, dim=1)
        cand_ok = opp_vis & (d_opp <= cfg.opponent_max_dist)
        best_b = opp_b.clone()
        best_d = torch.where(cand_ok, d_opp, torch.full_like(d_opp, float("inf")))
        start = deg & (self.fake_left <= 0) & (torch.rand(n, device=dev) < cfg.opp_fake_prob)
        if start.any():
            k = int(start.sum())
            a = torch.rand(k, device=dev) * 2 * math.pi
            r = torch.empty(k, device=dev).uniform_(0.3, cfg.opponent_max_dist)
            self.fake_off[start] = r.unsqueeze(1) * torch.stack([torch.cos(a), torch.sin(a)], dim=1)
            self.fake_left[start] = torch.empty(k, device=dev).uniform_(*cfg.opp_fake_dur)
        faking = self.fake_left > 0
        fake_d = torch.norm(self.fake_off, dim=1)
        use_fake = faking & (fake_d < best_d)
        best_b = torch.where(use_fake.unsqueeze(1), self._to_body(pos_w + self.fake_off - root), best_b)
        found = cand_ok | use_fake
        self.fake_left = (self.fake_left - dt).clamp(min=0.0)
        nb = torch.norm(ball_b, dim=1, keepdim=True).clamp(min=1e-3)
        virtual_b = ball_b * (1.0 + cfg.virtual_opponent_m / nb)
        # 劣化なしのエピソードは従来どおり真の敵 (ボールから遠くても仮想にしない)
        self.opp_virtual = deg & ~found
        self.obs_opp_b = torch.where(self.opp_virtual.unsqueeze(1), virtual_b, best_b)
        # 履歴に world 系で積む
        self._oh_ptr = (self._oh_ptr + 1) % self._oh_len
        opp_obs_w = root + self._to_world(self.obs_opp_b)
        rn = lambda x, sd: x + sd * torch.randn_like(x)  # 検出の揺れ (現在フレームの一様ノイズに相当)
        self.oh_obs[self._oh_ptr] = torch.cat(
            [rn(pos_w, cfg.hist_pos_noise), torch.where(vel_zero.unsqueeze(1), vel_w, rn(vel_w, cfg.hist_vel_noise)), rn(opp_obs_w, cfg.hist_pos_noise),
             visible.float().unsqueeze(1), self.opp_virtual.float().unsqueeze(1)], dim=1
        )
        self.oh_true[self._oh_ptr] = torch.cat([ball_true, self.obstacle_xy()], dim=1)
        ones = torch.ones(self.num_envs, 1, device=self.device)
        self.oh_teach[self._oh_ptr] = torch.cat(
            [rn(ball_true[:, :2], cfg.hist_pos_noise), rn(ball_true[:, 2:], cfg.hist_vel_noise),
             rn(self.obstacle_xy(), cfg.hist_pos_noise), ones, torch.zeros_like(ones)], dim=1
        )

    def _to_world(self, v_b: torch.Tensor) -> torch.Tensor:
        h = self.robot.data.heading_w
        c, s = torch.cos(h), torch.sin(h)
        return torch.stack([c * v_b[:, 0] - s * v_b[:, 1], s * v_b[:, 0] + c * v_b[:, 1]], dim=1)

    def history_b(self, true: bool, teacher: bool = False) -> torch.Tensor:
        """観測履歴を現在の base yaw 系で (N, T × dim)、古い順。位置は現在のロボット位置からの相対、速度は回転のみ。
        teacher=True は教師 (劣化なしで学習) 用の劣化なし履歴。"""
        buf = self.oh_teach if teacher else (self.oh_true if true else self.oh_obs)
        T = self._oh_len
        idx = (self._oh_ptr + 1 + torch.arange(T, device=self.device)) % T
        h = buf[idx].clone()  # (T, N, dim)
        root = self.robot.data.root_pos_w[:, :2]
        c, s = torch.cos(self.robot.data.heading_w), torch.sin(self.robot.data.heading_w)
        for k, off in ((0, root), (2, 0.0), (4, root)):
            x, y = (h[..., k : k + 2] - off).unbind(-1)
            h[..., k], h[..., k + 1] = c * x + s * y, -s * x + c * y
        return h.permute(1, 0, 2).reshape(self.num_envs, -1)

    def neutral_end(self) -> torch.Tensor:
        """罰なしの打ち切り (time_out 扱い): 解除後 post_s 倒れなかった / 触れる前にボールが ball_far_dist より遠くへ行った。"""
        post_done = (self.phase == PH_POST) & (self.elapsed() - self.post_since >= self.cfg.post_s)
        far = (self.phase != PH_PRE) & ~self.foot_touched & (self.body_ball_dist() > self.cfg.ball_far_dist) & self.kicked
        return post_done | far

    def _update_command(self):
        self._move_opponent_and_ball()
        if self.cfg.match_enable:
            self._update_phase()
        self._update_observations()
        # 足接触の記録と成功のラッチ (毎 step、終了判定の後に呼ばれる → 次 step の終了判定が見る)
        new_touch = self.foot_contact_now() & ~self.foot_touched
        self.ball_touch_xy[new_touch] = self.ball_xy()[new_touch]
        self.foot_touched |= new_touch
        first = (self.stolen_at < 0.0) & ~self.lost_obstacle() & self.foot_touched & self.ball_out()
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

    move_speed_max: float = 0.0
    """敵の直進速度の上限の初期値 [m/s] (0 で静止)。steal_move_speed_curriculum が move_speed_limit まで上げる。"""
    move_speed_limit: float = 1.0
    static_prob: float = 0.3
    """動く敵のカリキュラム中も、この割合のエピソードは敵・ボールを静止させる (速さ一様だと静止がほぼ無く、
    静止での奪取率が 94 → 90.6% に落ちた)。"""
    ball_rel_speed_max: float = 0.2
    """ボールの速度 = 敵の速度 + この大きさ以内の差 [m/s] (ボールと敵は別々の速度で動く)。"""
    ball_release_body_dist: float = 0.45
    ball_release_foot_dist: float = 0.3
    """ロボットの胴体 / 足がボールにこの距離まで近づいたらボールの駆動をやめて物理に任せる [m]。"""

    # --- 試合条件 (2026-11-22, STEAL_HANDOFF.md「次の再学習」の 1〜6)。False なら従来どおり ---
    match_enable: bool = False
    walkin_prob: float = 0.75
    """1. 歩行中からの開始の割合。残りは立位スタート (最初から w=1)。"""
    walkin_extra_dist: tuple[float, float] = (1.0, 2.5)
    """歩行中開始では配置をこの分だけ遠くし、歩いて近づく [m]。"""
    walkin_time: tuple[float, float] = (0.5, 2.0)
    act_dist: tuple[float, float] = (0.5, 2.0)
    """起動: walkin_time 経つか、ボールまで act_dist 以内に入ったら 0.3 s かけて奪取方策へ混合 (試合の起動条件 2.0 m)。"""
    walkin_speed: tuple[float, float] = (1.0, 2.3)
    walkin_turn_gain: tuple[float, float] = (1.0, 3.0)
    """起動前の歩行コマンド: ボール方向へこの速さ (クランプ・膝あり)、旋回 = gain × ボール方位 (±2.0)。"""
    walk_cmd_limits: tuple[float, float, float] = (2.0, 1.9, 2.0)
    walk_lateral_knee: float = 1.0
    walk_lateral_x_min: float = 0.3
    blend_s: float = 0.3
    """ノード (steal::BLEND_ON_S / BLEND_OFF_S) と同じ混合時間 [s]。"""
    contest_prob: float = 0.5
    """2. 敵がボールへ寄ってくる配置の割合 (従来は敵がボールを持っている配置だけ)。"""
    contest_ball_dist: tuple[float, float] = (0.3, 2.0)
    contest_enemy_dist: tuple[float, float] = (0.5, 2.0)
    contest_enemy_speed: tuple[float, float] = (0.2, 1.0)
    contest_reach_dist: float = 0.35
    kick_prob: float = 0.3
    kick_speed: tuple[float, float] = (2.0, 6.0)
    """3. 寄ってきた敵がボールに届いたとき、この割合で任意の方向へ蹴る (残りはドリブル)。"""
    free_out_dist: float = 0.6
    """敵がボールを持っていない間の「出た」判定: 足で触れた位置 (触れる前は初期位置) からの移動距離 [m]。"""
    ball_far_dist: float = 2.0
    """蹴られたボールが、触れる前にロボットからこの距離より遠くなったら罰なしで打ち切る (試合の終了条件)。"""
    release_prob: float = 0.4
    release_after: tuple[float, float] = (1.0, 4.0)
    release_zero_prob: float = 0.5
    release_move_speed: tuple[float, float] = (0.3, 1.5)
    post_s: float = 1.0
    """6. 起動から release_after の時点で歩行へ混合で戻す (コマンド 0 か任意の移動)。その後 post_s 倒れなければ罰なしで打ち切る。"""
    obs_degrade_prob: float = 0.8
    """4・5. 方策の観測を劣化させるエピソードの割合 (残りは真値)。"""
    obs_delay_steps: tuple[int, int] = (2, 10)
    obs_history: int = 30
    """Block 型 encoder に渡す観測履歴の長さ [step] (50 Hz で 0.6 s)。"""
    hist_pos_noise: float = 0.03
    hist_vel_noise: float = 0.1
    """履歴に積む観測の位置・速度のガウスノイズ σ。"""
    neck_limit: float = math.radians(90.0)
    fov_half: float = math.radians(52.5)
    near_blind_dist: float = 0.2
    jump_prob: float = 0.01
    jump_mag: tuple[float, float] = (0.2, 1.0)
    jump_dur: tuple[float, float] = (0.1, 0.5)
    veldrop_prob: float = 0.02
    veldrop_dur: tuple[float, float] = (0.2, 1.0)
    opponent_max_dist: float = 1.0
    virtual_opponent_m: float = 0.4
    """ノード (steal::OPPONENT_MAX_M / steal_virtual_opponent_m) と同じ。"""
    opp_miss_prob: float = 0.005
    opp_miss_dur: tuple[float, float] = (0.2, 1.0)
    opp_fake_prob: float = 0.005
    opp_fake_dur: tuple[float, float] = (0.2, 1.0)

    target_mode: str = "robot4"
    """目標方向の抽選。"robot4" = 開始時のロボット向き基準の前後左右から敵方向に近いものを除く (従来)、
    "robot_lr" = ロボット→ボールの線から左右どちらかへ target_lateral_range (敵方向に近い側は除く、2026-11-24)。
    ボール→敵の線の横 ("lateral"、2026-11-23) は敵側へ入る立ち位置が 12 → 28% に増えたので不採用・削除。"""
    target_lateral_range: tuple[float, float] = (math.radians(60.0), math.radians(120.0))

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


def steal_timeout(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """エピソード長の時間切れ (time_out 項) で 1。is_terminated_term は time_out の項を 0 にするので別に作る。
    罰なしの打ち切り (steal_neutral) は含めない。"""
    return env.termination_manager.get_term("time_out").float()


def steal_neutral_end(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    return env.command_manager.get_term(command_name).neutral_end()


def steal_obs_ball_pos(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """方策が見るボール位置 (base yaw 系, 2)。劣化なしなら ball_pos_rel と同じ。"""
    return env.command_manager.get_term(command_name).obs_ball_b[:, :2]


def steal_obs_ball_vel(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """方策が見るボール速度 (base yaw 系, 2)。劣化なしなら ball_vel と同じ。"""
    return env.command_manager.get_term(command_name).obs_ball_b[:, 2:]


def steal_obs_opponent(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """方策が見る敵の位置 (base yaw 系, 2)。ノードの steal_opponent と同じ規則。劣化なしなら真値。"""
    return env.command_manager.get_term(command_name).obs_opp_b


def steal_hist_obs(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """actor 用の観測履歴 (N, T × 8): 方策が見たボール位置・速度・敵位置・ボールが見えているか・敵が仮想か。"""
    return env.command_manager.get_term(command_name).history_b(true=False)


def steal_hist_teacher(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """教師-生徒の蒸留用: 教師が見る劣化なしの観測履歴 (N, T × 8)。"""
    return env.command_manager.get_term(command_name).history_b(true=False, teacher=True)


def steal_hist_true(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """critic 用の真値履歴 (N, T × 6): ボール位置・速度・敵位置。"""
    return env.command_manager.get_term(command_name).history_b(true=True)


def steal_ctrl(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """方策の外の制御量 [w, 歩行コマンド (3)]。StealActorCritic.env_actions が混合に使う (actor / critic には入れない)。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    return torch.cat([term.w.unsqueeze(1), term.cmd_ext], dim=1)


def steal_phase_obs(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """critic 用: [w, 起動前, 解除中/後] (3)。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    pre = (term.phase == PH_PRE).float()
    out = ((term.phase == PH_OFF) | (term.phase == PH_POST)).float()
    return torch.stack([term.w, pre, out], dim=1)


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


def ball_circling(env: "ManagerBasedRLEnv", min_dist: float = 0.25, command_name: str = "steal") -> torch.Tensor:
    """ボールのまわりを回り込んだ角度 [rad/step] (ボールから見たロボットの方位の変化量)。奪取方策が作動中で、
    足で触れる前だけ数える。ボールの真上付近 (min_dist 未満) は方位が定まらないので数えない。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    rel = term.robot.data.root_pos_w[:, :2] - term.ball_xy()
    ang = torch.atan2(rel[:, 1], rel[:, 0])
    delta = torch.remainder(ang - term.prev_circ_ang + math.pi, 2 * math.pi) - math.pi
    term.prev_circ_ang = ang
    return delta.abs() * ((term.w > 0.0) & ~term.foot_touched & (torch.norm(rel, dim=1) > min_dist)).float()


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


# ---------------------------------------------------------------------------
# 観測 (critic のみ) とカリキュラム
# ---------------------------------------------------------------------------
def steal_obstacle_vel(env: "ManagerBasedRLEnv", command_name: str = "steal") -> torch.Tensor:
    """敵の速度 (base yaw 系, 2)。方策には入れず critic だけが見る (実機ノードの入力契約を変えないため)。"""
    term: StealCommand = env.command_manager.get_term(command_name)
    h = term.robot.data.heading_w
    c, s = torch.cos(h), torch.sin(h)
    v = term.obstacle_vel
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=1)


def steal_move_speed_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "steal",
    step: float = 0.1,
    up_threshold: float = 0.75,
    min_episodes: int = 20000,
) -> float:
    """敵・ボールの速度上限を、終了したエピソードの奪取率が up_threshold 以上なら step ずつ上げる。
    学習中の奪取率は探索ノイズ込みで決定的評価より約 10 pt 低い (評価 95% で学習中 83%) ので 0.75 (0.85 では 0.2〜0.3 で停滞)。

    min_episodes 本ごとに判定 (下げない)。戻り値 (現在の上限) は Curriculum/ に記録される。
    """
    term: StealCommand = env.command_manager.get_term(command_name)
    if not hasattr(term, "_cur_n"):
        term._cur_n, term._cur_ok = 0, 0
    if len(env_ids) > 0:
        ok = env.termination_manager.get_term("ball_stolen")[env_ids]
        term._cur_n += len(env_ids)
        term._cur_ok += int(ok.sum())
    if term._cur_n >= min_episodes:
        if term._cur_ok / term._cur_n >= up_threshold and term.move_speed_max < term.cfg.move_speed_limit:
            term.move_speed_max = min(term.move_speed_max + step, term.cfg.move_speed_limit)
        term._cur_n, term._cur_ok = 0, 0
    return term.move_speed_max
