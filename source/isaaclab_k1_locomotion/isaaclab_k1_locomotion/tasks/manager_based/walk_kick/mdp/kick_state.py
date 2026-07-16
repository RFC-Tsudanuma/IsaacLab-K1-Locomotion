# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""キック報酬が共有する latch 状態。

値 latch (凍結) と状態 latch (フラグ) を厳密に分けて保持する。

* 値 latch: トリガー L (``v_ball > v_thresh``) の発火時に τ_direction / v_ball / p_style を
  **同時に**スナップショットして固定する。以降はその凍結値で dense に払う。
* 状態 latch: ``kick_done`` (L 発火) と ``overshoot_fired`` (後方レイ R の左右跨ぎ)。
  いずれもエピソード内で一度立ったら解除しない。

状態はステップごとに一度だけ更新する (``common_step_counter`` でステップ境界を検出)。
どの項から先に呼ばれても同じ結果になるので、報酬項の評価順に依存しない。

NOTE: RewardManager は weight==0 の項をスキップするので、カリキュラムで weight を 0 から
      立ち上げる項だけに更新を任せると Phase 1 の間ずっと状態が更新されない。そのため
      **常に有効な termination 項** (:func:`..terminations.kick_finished`) からも
      :func:`kick_state` を呼び、毎ステップの更新を保証している。TerminationManager は
      RewardManager より先に走るので、報酬項が読む時点で状態は最新になっている。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ATTR = "_kick_latch_state"


def kick_state(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    command_name: str = "kick_direction",
    ball_name: str = "soccer_ball",
) -> dict:
    """キック関連の共有状態を返す。同一ステップ内では一度しか更新しない。"""
    step = int(env.common_step_counter)
    state = getattr(env, _ATTR, None)
    if state is not None and state["step"] == step:
        return state

    robot = env.scene["robot"]
    ball = env.scene[ball_name]
    device = env.device

    ball_pos = ball.data.root_pos_w[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_w[:, :2]

    # kick_direction コマンドは [sin θ, cos θ, v_target] (θ は world frame)
    cmd = env.command_manager.get_command(command_name)
    kick_dir = torch.stack([cmd[:, 1], cmd[:, 0]], dim=-1)  # (cos θ, sin θ), 単位ベクトル
    v_target = cmd[:, 2]
    # kick_dir を水平面で -90° 回した右向き単位ベクトル: R(-90)·(x, y) = (y, -x)
    right_vec = torch.stack([kick_dir[:, 1], -kick_dir[:, 0]], dim=-1)

    # ロボット胴体のヨー方向
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    forward = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

    if state is None:
        state = {
            "step": -1,
            "P_kick": torch.zeros(env.num_envs, 2, device=device),
            "init_side": torch.ones(env.num_envs, device=device),
            "kick_done": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "overshoot_fired": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "overshoot_event": torch.zeros(env.num_envs, device=device),
            "tau_direction_frozen": torch.zeros(env.num_envs, device=device),
            "v_ball_frozen": torch.zeros(env.num_envs, device=device),
            "p_style_frozen": torch.zeros(env.num_envs, device=device),
            "G": torch.zeros(env.num_envs, 2, device=device),
            "p_walk": torch.zeros(env.num_envs, device=device),
            "tau_walk": torch.zeros(env.num_envs, device=device),
            "d_sole_to_ball": torch.zeros(env.num_envs, device=device),
            "p_kick_pose": torch.zeros(env.num_envs, device=device),
            "v_target": torch.zeros(env.num_envs, device=device),
        }
        setattr(env, _ATTR, state)
        # 初回はまだエピソードが始まっていないので全 env を初期化対象にする
        just_reset = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    else:
        # reward / termination は episode_length_buf の加算後、_reset_idx の前に走るので、
        # 新エピソードの 1 歩目は episode_length_buf == 1 になる。
        just_reset = env.episode_length_buf == 1

    state["step"] = step
    state["v_target"] = v_target

    # ------------------------------------------------------------------ #
    # エピソード開始時のリセット: 凍結値・フラグ・P_kick・初期側符号
    # ------------------------------------------------------------------ #
    # 符号付き横距離 s: 後方レイ R からロボットがどちら側にいるか
    s = ((robot_pos - ball_pos) * right_vec).sum(dim=-1)

    if just_reset.any():
        # P_kick: R 上、ボールから後方 r_stance の点。エピソード終了まで固定。
        state["P_kick"][just_reset] = (ball_pos - r_stance * kick_dir)[just_reset]
        # 初期側の符号 (s == 0 のときは + 側とみなす)
        state["init_side"][just_reset] = torch.where(
            s >= 0.0, torch.ones_like(s), -torch.ones_like(s)
        )[just_reset]
        state["kick_done"][just_reset] = False
        state["overshoot_fired"][just_reset] = False
        state["tau_direction_frozen"][just_reset] = 0.0
        state["v_ball_frozen"][just_reset] = 0.0
        state["p_style_frozen"][just_reset] = 0.0

    # ------------------------------------------------------------------ #
    # p_style: 胴体の向きが蹴り方向にどれだけ正対しているか (1 = 正対)
    # ------------------------------------------------------------------ #
    p_style = torch.clamp((forward * kick_dir).sum(dim=-1), min=0.0, max=1.0)

    # ------------------------------------------------------------------ #
    # 値 latch: L = (v_ball > v_thresh) の立ち上がりで τ_direction, v_ball, p_style を同時凍結
    # ------------------------------------------------------------------ #
    v_ball = ball_vel.norm(dim=-1)
    trigger = (v_ball > v_thresh) & (~state["kick_done"])

    if trigger.any():
        # τ_direction: ボールの飛翔方向と蹴り方向の角度誤差 [rad]
        ball_dir = ball_vel / (v_ball.unsqueeze(-1) + 1e-6)
        cos_err = torch.clamp((ball_dir * kick_dir).sum(dim=-1), min=-1.0, max=1.0)
        sin_err = ball_dir[:, 0] * kick_dir[:, 1] - ball_dir[:, 1] * kick_dir[:, 0]
        tau_direction = torch.abs(torch.atan2(sin_err, cos_err))

        state["tau_direction_frozen"] = torch.where(
            trigger, tau_direction, state["tau_direction_frozen"]
        )
        state["v_ball_frozen"] = torch.where(trigger, v_ball, state["v_ball_frozen"])
        state["p_style_frozen"] = torch.where(trigger, p_style, state["p_style_frozen"])
        state["kick_done"] = state["kick_done"] | trigger

    kick_done = state["kick_done"]

    # ------------------------------------------------------------------ #
    # 目標終端 G: R 上をボール側へ滑る点。latch 後は P_kick に固定して飛翔ボールを追わせない。
    # ------------------------------------------------------------------ #
    dist_robot_ball = (robot_pos - ball_pos).norm(dim=-1)
    reach = torch.clamp(alpha * dist_robot_ball, min=r_stance, max=0.5)
    G = ball_pos - reach.unsqueeze(-1) * kick_dir
    G = torch.where(kick_done.unsqueeze(-1), state["P_kick"], G)
    state["G"] = G

    # τ_walk: ロボット速度の G 方向成分 (符号付き)
    to_G = G - robot_pos
    d_to_G = to_G.norm(dim=-1)
    dir_to_G = to_G / (d_to_G.unsqueeze(-1) + 1e-6)
    state["tau_walk"] = (robot_vel * dir_to_G).sum(dim=-1)
    state["d_to_G"] = d_to_G

    # ------------------------------------------------------------------ #
    # 状態 latch: overshoot。後方レイ R を跨いで初期側と反対側へ入ったら発火。
    # 前後位置・0.5m・G とは無関係。base_link の水平位置のみで判定する。
    # ------------------------------------------------------------------ #
    crossed = (s * state["init_side"]) < 0.0
    newly_fired = crossed & (~state["overshoot_fired"])
    state["overshoot_event"] = newly_fired.float()  # 発火したステップだけ 1 (1エピソード最大1回)
    state["overshoot_fired"] = state["overshoot_fired"] | crossed

    # ------------------------------------------------------------------ #
    # d_soleToBall: 左右の足裏のうちボールに近い方の距離
    # ------------------------------------------------------------------ #
    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[:, foot_ids, :]  # (N, 2, 3)
    ball_pos_3d = ball.data.root_pos_w[:, :3].unsqueeze(1)  # (N, 1, 3)
    state["d_sole_to_ball"] = (foot_pos - ball_pos_3d).norm(dim=-1).min(dim=1).values

    state["p_style"] = p_style
    state["d_to_P_kick"] = (robot_pos - state["P_kick"]).norm(dim=-1)

    return state


_FOOT_IDS_ATTR = "_kick_foot_body_ids"


def _foot_body_ids(env: ManagerBasedRLEnv, robot) -> list[int]:
    """左右の足リンクの body index を一度だけ解決してキャッシュする。"""
    ids = getattr(env, _FOOT_IDS_ATTR, None)
    if ids is None:
        left = robot.find_bodies("left_foot_link")[0][0]
        right = robot.find_bodies("right_foot_link")[0][0]
        ids = [left, right]
        setattr(env, _FOOT_IDS_ATTR, ids)
    return ids
