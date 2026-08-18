# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール履歴版の報酬。**予測式を一切含まない**。

直接版の ``target_reach_velocity_direct`` は「手書きの外挿で求めた到達点へ
どれだけ速く近づいたか」を測っていた。実機の経路から外挿式を消しても、
**報酬に残っていれば方策はその戦略を模倣する**ので、線形外挿が表現できない
振る舞い (バウンド球、回転球、遅い球を引きつける判断) は学習されない。
式が学習の天井として残ってしまう。

ここでは予測を捨て、**ボールの真の現在位置**だけを使う潜在関数ベースの
整形報酬にする::

    φ(s) = −|robot_y − ball_y|          (ゴール座標系、真値)
    r    = φ(s') − φ(s)                 = 横方向の距離を縮めたぶん

「どう先読みするか」は指定しない。先読みが有効なら、スパース報酬
(save_touch_bonus) との組み合わせで方策が自分で獲得する。

★ 潜在関数の差分形なので、理論上は最適方策を変えない (reward shaping の
  ポテンシャル定理)。密な誘導を与えつつ、解を歪めない形。
★ 真値を使ってよい。報酬は実機では動かない (critic に真値を渡すのと同じ扱い)。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..mdp.observations import ball_pos_goal, gk_buffers, robot_pos_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_PREV_ATTR = "_gk_ballhist_prev_potential"
_PREV_STEP_ATTR = "_gk_ballhist_prev_potential_step"


def ball_lateral_progress(
    env: "ManagerBasedRLEnv",
    max_step_m: float = 0.03,
    deadband: float = 0.12,
    stop_speed: float = 0.5,
) -> torch.Tensor:
    """ボールとの横方向距離を縮めたぶんの報酬 (N,)。**予測を含まない**。

    Args:
        deadband: この誤差 [m] 以内を「到達済み」とみなし、停止報酬へ切り替える。
        stop_speed: 停止報酬がゼロになるベース速度 [m/s]。
        max_step_m: 1 ステップあたりの差分のクリップ [m]。
            ★ 2026-08-18: 0.1 → 0.03 に絞った。0.1 は 5 m/s 相当で、**歩くより
              ボールに向かって倒れ込むほうが速く距離を縮められる**ため、方策が
              ダイブを学習してしまった (実測: base_height 終了 26%、転倒・逸脱の
              合計が 41%)。0.03 は 1.5 m/s 相当で、横移動の実力 (1.3〜1.5 m/s) の
              すぐ上。正常な歩行移動はクリップに触れず、倒れ込みだけが頭打ちになる。
            リセットやボール再発射で潜在関数が不連続に飛んだときの保護も兼ねる。

    ★ ボール非アクティブ時は 0 を返す (追う対象が無い)。
    """
    bufs = gk_buffers(env)
    active = bufs["ball_active"]

    # ★ 2026-08-18: ボールの y は **ゴール幅にクランプする**。
    #
    #   クランプしないと、枠外へ飛ぶ球 (状況の多様化で追加) の y にどこまでも
    #   近づこうとして、フィールド境界 (±2.2m) を越えて追いかける。
    #   実測: out_of_bounds 終了が 11.5% -> 18.6% に増え、最大の失敗要因になった。
    #   GK が守るのはゴール幅の中なので、外へ出た球を追う必要は無い。
    #   compute_target_y も同じ理由で ±max_y にクランプしている。
    max_y = float(getattr(env.cfg.goalkeeper, "goal_half_width", 1.3))
    ball_y = ball_pos_goal(env)[:, 1].clamp(-max_y, max_y)
    robot_y = robot_pos_goal(env)[:, 1]
    potential = -(robot_y - ball_y).abs()

    prev = getattr(env, _PREV_ATTR, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = potential.clone()
        setattr(env, _PREV_ATTR, prev)

    # 報酬は 1 ステップ 1 回しか呼ばれないが、念のため冪等にしておく
    if getattr(env, _PREV_STEP_ATTR, -1) == int(env.common_step_counter):
        return torch.zeros_like(potential)
    setattr(env, _PREV_STEP_ATTR, int(env.common_step_counter))

    delta = (potential - prev).clamp(-max_step_m, max_step_m)
    setattr(env, _PREV_ATTR, potential.clone())

    # ★ 2026-08-18: **到達したら止まる** 項を足す。
    #
    #   差分だけだと「近づいたぶん」しか見ないので、到達しても止まる理由が無く、
    #   行き過ぎても罰が無い。**倒れ込んでも近づいたぶんは満額もらえる**ため、
    #   方策はダイブを選ぶ (実測: base_height 終了 26% / 転倒・逸脱 合計 42%)。
    #   直接版の target_reach_velocity は同じ問題を「deadband 内は止まっているほど
    #   高い」に切り替えることで塞いでいる。ここでも同じ構造にする。
    #   ただし外挿点ではなく **ボールの真の現在位置** を基準にするので、予測は含まない。
    #
    #   スケールを差分側と揃える: r_stop は [0,1] なので max_step_m 倍して、
    #   同じ weight で桁が合うようにする。
    err = (robot_y - ball_y).abs()
    speed = torch.norm(env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
    r_stop = (1.0 - speed / stop_speed).clamp(0.0, 1.0) * max_step_m
    delta = torch.where(err <= deadband, r_stop, delta)

    # リセット直後は前ステップが別エピソードの値なので無効化する
    fresh = env.episode_length_buf < 2
    out = torch.where(fresh | (~active), torch.zeros_like(delta), delta)
    return out
