# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick のキック報酬。

B-Human "A Modular Ball Kicking Behavior with Reinforcement Learning" の
報酬テーブルを K1 向けに実装したもの。latch 状態は :mod:`.kick_state` が管理する。

フェーズゲート:
  * pre-latch  (kick_done=false): 項1-3 = 0、項4/5 有効
  * L 発火時                    : τ_direction, v_ball, p_style を凍結、kick_done=true、G を P_kick に固定
  * post-latch (kick_done=true) : 項1-3 を凍結値で毎ステップ dense に払う、項5 = 0、項4 継続
  * 項6 (overshoot) は常時有効
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _r_direction(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float,
) -> tuple[torch.Tensor, dict]:
    """r_direction = (f(τ_direction) − 0.5) * 2 * p_style （いずれも凍結値）。

    f(τ) = exp(−τ² / 2σ²) なので τ=0 (方向ぴったり) で f=1 → r_direction = +p_style、
    方向が大きく外れると f→0 → r_direction = −p_style。
    pre-latch では凍結値が 0 のままなので、呼び出し側で kick_done ゲートを掛けること。

    NOTE: 負値は 0 にクリップする。項1-3 は post-latch に凍結値を毎ステップ dense で払うが、
          転倒すると base_contact でエピソードが終わり支払いも止まる。負のまま払うと
          「方向を外したキックの後は早く転んで損失を止めた方が得」という抜け道ができるため
          (転倒罰は -100 * dt = -2.0 の一度きりなのに対し、負の dense 払いは窓の秒数だけ
          累積する)。クリップの代償として、方向を外したキックは「罰される」のではなく
          「報われない」(= 蹴らないのと同値) 扱いになる。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    tau = state["tau_direction_frozen"]
    f_dir = torch.exp(-(tau**2) / (2.0 * sigma_direction**2))
    r_dir = (f_dir - 0.5) * 2.0 * state["p_style_frozen"]
    r_dir = torch.clamp(r_dir, min=0.0)

    # pre-latch は 0
    return r_dir * state["kick_done"].float(), state


def kick_direction(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
) -> torch.Tensor:
    """項1. Kick Direction。凍結した飛翔方向誤差 × 凍結 p_style。shape: (N,)"""
    r_dir, _ = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)
    return r_dir


def kick_velocity_scaled(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    sigma_velocity: float = 1.0,
) -> torch.Tensor:
    """項2. Kick Velocity Scaled = r_direction * f(v_ball)。shape: (N,)

    f(v_ball) は「要求された蹴り速度 v_target にどれだけ一致したか」を測る。
    指令速度に対する一致度を見ないと可変キック強度が学習できないため、
    f(v) = exp(−((v − v_target) / σ)²) とした。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    v_err = state["v_ball_frozen"] - state["v_target"]
    f_vel = torch.exp(-((v_err / sigma_velocity) ** 2))
    return r_dir * f_vel


def kick_velocity_strong(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
) -> torch.Tensor:
    """項3. Kick Velocity Strong = r_direction * v_ball（生の速度）。shape: (N,)"""
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)
    return r_dir * state["v_ball_frozen"]


def walk_speed(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_walk: float = 0.5,
    sigma_walk_potential: float = 0.5,
) -> torch.Tensor:
    """項4. Walk Speed = (f(τ_walk) − 0.5) * 2 * p_walk。shape: (N,)

    τ_walk はロボット速度の G 方向成分（符号付き）。f = sigmoid(τ_walk / σ) なので、
    G に向かって進めば正、遠ざかれば負になる。p_walk = exp(−d(robot, G) / σ_pot) は
    G への接近度 (1 = 到達)。G は P_kick で下限クランプされるので、キック立ち位置に
    着いた時点で p_walk が飽和し、この項は self-gate する。凍結しない。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    f_walk = torch.sigmoid(state["tau_walk"] / sigma_walk)
    p_walk = torch.exp(-state["d_to_G"] / sigma_walk_potential)
    return (f_walk - 0.5) * 2.0 * p_walk


def approach_penalty(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_sole: float = 0.35,
    sigma_pose: float = 0.3,
) -> torch.Tensor:
    """項5. Approach Penalty = f(d_soleToBall) * p_kickPose。負の重みで使う。shape: (N,)

    * f(d_soleToBall) = 1 − exp(−(d/σ)²) : 足裏がボールから **遠いほど大きい** (近い=0, 遠い=1)
    * p_kickPose                          : 理想キック姿勢からの **ズレほど大きい** (合致=0, ズレ=1)

    理想（足がボールに近い × 姿勢が P_kick と一致）で 0、最悪（遠い × ズレ）で最大の罰。
    pre-latch のみ有効（kick_done で 0 ゲート）。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    # 遠いほど 1 に近づく
    f_sole = 1.0 - torch.exp(-((state["d_sole_to_ball"] / sigma_sole) ** 2))

    # 理想キック姿勢 = P_kick に立ち、蹴り方向を向いている。合致度が高いほど 0 に近づく。
    pose_match = torch.exp(-((state["d_to_P_kick"] / sigma_pose) ** 2)) * state["p_style"]
    p_kick_pose = 1.0 - pose_match

    return f_sole * p_kick_pose * (~state["kick_done"]).float()


def kick_pose_overshoot(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
) -> torch.Tensor:
    """項6. Kick Pose Overshoot。後方レイ R を跨いだ瞬間だけ 1。負の重みで使う。shape: (N,)

    エピソード開始時に記録した s = dot(base_pos − ball_pos, right_vec) の符号を初期側とし、
    符号が反転したら発火して latch する。戻っても解除せず、1 エピソード最大 1 回だけ罰する。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    return state["overshoot_event"]
