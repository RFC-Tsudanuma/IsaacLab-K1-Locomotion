# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""(B-Human に無い、既定は無効) 歩行方策で遠くのボールへ近づき、途中でキック方策へ渡す「受け渡し」を学習に入れる。

実機では普段は歩行方策が動いていて、ボールが ``ball_approach_m`` (今は 1.2 m) 以内に入るとキック方策へ切り替える
(deploy リポジトリ ``rl_policy_isaaclab_node.cpp`` の KickBehavior)。そのとき:

* 歩行が動いている間も、キック方策の 100 ステップの履歴は毎制御周期積まれている (``observe_walk_tick``)。
  歩行の 49 項は歩行方策が見ているのと同じ値 (速度指令の欄 = 歩行の速度指令、位相 = 歩行の位相、
  前の action = 歩行の脚 12)、13 番目 (歩行周波数) の前の action は 0。キックの状態 (ball_close_to_kick_pose など) も進める。
* 切り替えた瞬間 (``on_enter``) は履歴・位相・前の action をそのまま引き継ぎ、そこからはキック方策が自分の速度指令
  (walk_command) で動く。

今までの学習にはこの受け渡しが無く (キック方策はボールまで自分で vx ≤ 1.0 で歩く、リセット時の胴体の速さ ±0.5 m/s、
ボールは 0.4〜1.5 m)、実機で 2 m/s 前後で歩いて来て 1.2 m で渡すとキック方策がボールへ突っ込んだ。

ここでは新しいボールのうち ``handoff_prob`` の割合を 1.5〜4 m に置き、凍結した歩行方策 (model_52700 の actor と観測の
正規化) にボールの方へ歩かせ、ボールまでの距離 (胴体の中心から) が受け渡し距離 (0.8〜2.0 m、1 回ごとに引く) 以下に
なったらキック方策へ渡す。歩行が動かしている間の観測・位相・前の action は上の実機と同じになるように作る:

* env.step に入ってきた action のうち、歩行の env の分を「歩行方策の脚 12 + 周波数 0」に置き換えてから action manager に
  渡す。こうすると last_joint_action / gait_frequency_action の観測、位相の積分 (周波数のずらし 0)、関節の目標がすべて
  歩行のものになる。
* 歩行の入力は直前の観測 (``env.obs_buf``) の command グループ + policy 履歴の先頭 49 項 × 100 ステップ。キックの policy
  履歴は項ごとに (100 × 次元) のブロックが並び、歩行の 7 項が先頭にあるので (history_layout.KICK_POLICY_TERM_SPECS)、
  先頭 4900 個がそのまま歩行の policy 履歴と同じ並びになる。
* 速度指令 (base_velocity、観測の速度指令の欄と位相の周波数の元) は歩行の env では歩行の速度指令にする
  (walk_command.BallWalkVelocityCommand)。

歩行が動かしたステップはキック方策の PPO の標本にしない。env はそのステップの報酬を 0 にし (episode の報酬の集計からも
引く)、``env.extras["bwk_walk_driven"]`` (そのステップの action が歩行のものだったか) と ``env.extras["bwk_walk_mode"]``
(次の action が歩行のものになるか) を出す。PPO 側 (agents/ppo_actor_freeze.py) がこれを見て、方策・価値の損失と
観測の正規化の更新から外す。
"""

from __future__ import annotations

import math
import types
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from .kick_command import BHumanKickCommand


def resolve_repo_path(path: str) -> Path:
    """相対パスはこのリポジトリの一番上から見る (vast.ai などへ rsync した先でも同じに読めるように、CWD は見ない)。"""
    p = Path(path)
    if p.is_absolute():
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "frozen_policies").is_dir() and (parent / "source").is_dir():
            return parent / p
    raise FileNotFoundError(f"リポジトリの一番上 (frozen_policies/ と source/ がある所) が見つかりません: {here}")


class FrozenWalkActor(torch.nn.Module):
    """歩行の checkpoint (HistoryActorCritic、1 ステップ 49) の actor と観測の正規化だけを読んだもの。重みは変えない。"""

    def __init__(self, ckpt_path: Path, device: str):
        super().__init__()
        from ...locomotion.agents.history_actor_critic import HistoryEncoderHead
        from ...locomotion.history_layout import HISTORY_LENGTH, MLP_HISTORY_STEPS, POLICY_TERM_SPECS, COMMAND_DIM

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        actor = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
        hidden = []
        i = 0
        while f"mlp.{i + 2}.weight" in actor:
            hidden.append(actor[f"mlp.{i}.weight"].shape[0])
            i += 2
        self.head = HistoryEncoderHead(
            direct_dim=COMMAND_DIM, term_specs=POLICY_TERM_SPECS, output_dim=actor[f"mlp.{i}.weight"].shape[0],
            hidden_dims=hidden, activation="elu", history_length=HISTORY_LENGTH, mlp_history_steps=MLP_HISTORY_STEPS,
            cnn_kernel_sizes=(8, 4), cnn_filter_sizes=(32, 16), cnn_stride_sizes=(4, 2),
        )
        self.head.load_state_dict(actor, strict=True)
        # rsl_rl EmpiricalNormalization: (x - mean) / (std + 1e-2)
        self.register_buffer("mean", sd["actor_obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["actor_obs_normalizer._std"].clone())
        self.input_dim = self.mean.shape[-1]
        self.history_dim = self.input_dim - COMMAND_DIM
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, command: torch.Tensor, policy_history: torch.Tensor) -> torch.Tensor:
        x = torch.cat((command, policy_history[:, : self.history_dim]), dim=-1)
        return self.head((x - self.mean) / (self.std + 1e-2))


class WalkHandoff:
    """受け渡しの状態 (env ごとに「今は歩行が動かしているか」) と、action・報酬の差し替え。BHumanKickCommand が持つ。"""

    def __init__(self, kick: BHumanKickCommand):
        self.kick = kick
        self.cfg = kick.cfg
        self.env = kick._env
        n, dev = kick.num_envs, kick.device
        self.walk_mode = torch.zeros(n, dtype=torch.bool, device=dev)  # 次の action を歩行が出すか
        self.walk_driven = torch.zeros(n, dtype=torch.bool, device=dev)  # いま実行したステップの action が歩行のものか
        self.entry_dist = torch.zeros(n, device=dev)
        self.speed = torch.zeros(n, device=dev)
        self.walk_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.max_walk_steps = int(round(self.cfg.handoff_max_walk_s / kick.dt))
        self.walk = FrozenWalkActor(resolve_repo_path(self.cfg.handoff_walk_checkpoint), dev)
        self._installed = False
        # 確認用 (学習のログ Metrics/kick/handoff_*): 学習開始からの受け渡しの回数と、渡した瞬間の胴体の水平の速さ・
        # ボールまでの距離の平均 (直近 50 回くらいの指数移動平均)
        self.stats = {"handoff_count": 0.0, "handoff_speed": 0.0, "handoff_dist": 0.0}
        self._stats_alpha = 0.02

    # ------------------------------------------------------------------ #
    # env への差し込み (action manager / reward manager はコマンドより後に作られるので、最初の reset で入れる)
    # ------------------------------------------------------------------ #
    def install(self):
        if self._installed:
            return
        env = self.env
        if not (hasattr(env, "action_manager") and hasattr(env, "reward_manager")):
            return  # まだ作られていない (次の reset / compute でもう一度呼ぶ)
        am, rm = env.action_manager, env.reward_manager
        orig_process = am.process_action
        orig_compute = rm.compute
        h = self

        def process_action(_self, action: torch.Tensor):
            h.walk_driven[:] = h.walk_mode
            env.extras["bwk_walk_driven"] = h.walk_driven
            env.extras["bwk_walk_mode"] = h.walk_mode
            ids = h.walk_mode.nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                obs = env.obs_buf
                a = h.walk(obs["command"][ids], obs["policy"][ids])
                action = action.clone()
                action[ids, : a.shape[1]] = a
                action[ids, a.shape[1]:] = 0.0  # 歩行周波数のずらし (実機: 歩行中の 13 番目の前の action は 0)
            return orig_process(action)

        def compute(_self, dt: float):
            r = orig_compute(dt)
            m = h.walk_driven
            if m.any():
                # 歩行が動かしたステップの報酬は 0。episode の項ごとの集計 (Episode_Reward/*) からも引く。
                for idx, name in enumerate(_self._term_names):
                    _self._episode_sums[name][m] -= _self._step_reward[m, idx] * dt
                _self._step_reward[m] = 0.0
                r[m] = 0.0
            return r

        am.process_action = types.MethodType(process_action, am)
        rm.compute = types.MethodType(compute, rm)
        self._installed = True

    # ------------------------------------------------------------------ #
    def on_new_ball(self, ids: torch.Tensor):
        """新しいボール (エピソード開始・キックの後) のうち handoff_prob の割合を遠くに置き、歩行で近づかせる。

        ボールの置き直し (_reset_ball_root_states) の後、キック課題の引き直し (_resample_kick_ball) の前に呼ぶ。
        """
        cfg, k, dev = self.cfg, self.kick, self.kick.device
        self.walk_mode[ids] = False
        self.walk_steps[ids] = 0
        pick = ids[torch.rand(len(ids), device=dev) < cfg.handoff_prob]
        m = len(pick)
        if m == 0:
            return
        robot_xy = k.robot.data.root_pos_w[pick, :2]
        lo, hi = cfg.handoff_ball_dist
        dist = lo + (hi - lo) * torch.rand(m, device=dev)
        ang = (torch.rand(m, device=dev) * 2.0 - 1.0) * math.pi
        pos = torch.zeros(m, 3, device=dev)
        pos[:, 0] = robot_xy[:, 0] + dist * torch.cos(ang)
        pos[:, 1] = robot_xy[:, 1] + dist * torch.sin(ang)
        pos[:, 2] = cfg.ball_radius + cfg.ball_radius_margin + 0.01
        quat = torch.zeros(m, 4, device=dev)
        quat[:, 0] = 1.0
        vel = torch.zeros(m, 6, device=dev)  # 止まったボール
        k.ball.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=pick)
        k.ball.write_root_velocity_to_sim(vel, env_ids=pick)
        k.ball_pos_w[pick] = pos
        k.ball_vel_w[pick] = vel
        k.last_ball_vel_w[pick] = 0.0
        k.buffered_global_ball_pos[pick] = pos.unsqueeze(1).expand(-1, k.buffered_global_ball_pos.shape[1], -1)
        robot_yaw = k.yaw[pick]
        rel = torch.stack((dist * torch.cos(ang - robot_yaw), dist * torch.sin(ang - robot_yaw)), dim=-1)
        k.ball_rel_root[pick] = rel
        k.ball_rel_root_prev[pick] = rel

        elo, ehi = cfg.handoff_entry_dist
        entry = elo + (ehi - elo) * torch.rand(m, device=dev)
        slo, shi = cfg.handoff_speed_range
        self.entry_dist[pick] = entry
        self.speed[pick] = slo + (shi - slo) * torch.rand(m, device=dev)
        # 受け渡し距離より内側から始まるボールはすぐキック方策 (今までと同じ)
        self.walk_mode[pick] = dist > entry

    def update(self):
        """物理ステップの後 (BHumanKickCommand._update、キック状態を進めた後・ボール再配置の判定の前) に呼ぶ。"""
        k = self.kick
        wm = self.walk_mode
        if not wm.any():
            return
        self.walk_steps[wm] += 1
        dist = torch.norm(k.ball_pos_w[:, :2] - k.base_pos[:, :2], dim=-1)
        # 受け渡し: 受け渡し距離以内 (どの向きでも)。歩行のままボールに触れた / 歩く時間が長すぎた場合も渡す。
        enter = wm & ((dist <= self.entry_dist) | (k.ball_kicked_counter > 0) | (self.walk_steps >= self.max_walk_steps))
        # 歩行中に左右の足の判定が入れ替わっても行き過ぎ (ball_walk_target_overshoot、キック方策への罰) を残さない
        k.ball_walk_target_overshoot[wm] = 0.0
        ids = enter.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            v = torch.norm(k.root_vel_w[ids, :2], dim=-1)
            # 最初の 1 回目は平均そのものから始める (0 から始めると立ち上がりが遅い)
            a = 1.0 if self.stats["handoff_count"] == 0 else 1.0 - (1.0 - self._stats_alpha) ** len(ids)
            self.stats["handoff_speed"] += a * (v.mean().item() - self.stats["handoff_speed"])
            self.stats["handoff_dist"] += a * (dist[ids].mean().item() - self.stats["handoff_dist"])
            self.stats["handoff_count"] += len(ids)
            self.walk_mode[ids] = False

    def walk_command(self) -> torch.Tensor:
        """歩行の env に渡す速度指令 (vx, vy, wz): ボールの方へ向きを変えながら、引いた速さで進む。

        歩行の学習の範囲 (vx ±2.0、wz ±1.0) の中。ボールが正面から外れているほど vx を cos で落とす (真横より後ろなら
        その場で向きを変える)。
        """
        k, cfg = self.kick, self.cfg
        rel = quat_apply_inverse(k.base_quat_z, k.ball_pos_w - k.base_pos)[:, :2]
        err = torch.atan2(rel[:, 1], rel[:, 0])
        vx = self.speed * torch.cos(err).clamp(min=0.0)
        wz = (cfg.handoff_yaw_gain * err).clamp(-1.0, 1.0)
        return torch.stack((vx, torch.zeros_like(vx), wz), dim=-1)
