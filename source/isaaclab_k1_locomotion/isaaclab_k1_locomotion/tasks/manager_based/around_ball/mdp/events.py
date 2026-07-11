# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用のイベント関数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# 「蹴ったらボールだけ再配置」用の per-env バッファ名
_BALL_SPAWN_POS_ATTR = "_ball_spawn_pos_w"          # 直近スポーン位置 (N, 2)
_BALL_RELOCATE_CD_ATTR = "_ball_relocate_countdown"  # 再配置カウントダウン (N,), -1=非発火


def _place_ball_in_front_cone(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    dist_range: tuple[float, float],
    half_angle: float,
    ball_radius: float,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """ロボットの現在の向き基準の正面扇形にボールを静止配置し、新しい xy 位置を返す。"""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    n = len(env_ids)

    dist = torch.empty(n, device=env.device).uniform_(float(dist_range[0]), float(dist_range[1]))
    ang = torch.empty(n, device=env.device).uniform_(-float(half_angle), float(half_angle))
    theta = robot.data.heading_w[env_ids] + ang

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = robot.data.root_pos_w[env_ids, 0] + dist * torch.cos(theta)
    pose[:, 1] = robot.data.root_pos_w[env_ids, 1] + dist * torch.sin(theta)
    pose[:, 2] = ball_radius  # 平面地形 (z=0) 前提
    pose[:, 3:7] = ball.data.default_root_state[env_ids, 3:7]  # 単位クォータニオン

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    return pose[:, :2]


def _ball_tracking_buffers(env: "ManagerBasedEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """スポーン位置・再配置カウントダウンのバッファを (遅延生成して) 返す。"""
    spawn = getattr(env, _BALL_SPAWN_POS_ATTR, None)
    if spawn is None or spawn.shape != (env.num_envs, 2):
        spawn = torch.zeros(env.num_envs, 2, device=env.device)
        setattr(env, _BALL_SPAWN_POS_ATTR, spawn)
    cd = getattr(env, _BALL_RELOCATE_CD_ATTR, None)
    if cd is None or cd.shape != (env.num_envs,):
        cd = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        setattr(env, _BALL_RELOCATE_CD_ATTR, cd)
    return spawn, cd


def reset_ball_in_front_cone(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    dist_range: tuple[float, float] = (0.6, 2.0),
    half_angle: float = 1.0,
    ball_radius: float = 0.11,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """リセット後のロボットの正面扇形内にボールを静止状態で配置する。

    walk_kick の ``reset_ball_in_front_of_robot`` と同方式: ロボットの現在位置・向き
    (heading) を基準に、距離 d ∈ ``dist_range``・方位 φ ∈ ±``half_angle`` [rad] を
    一様サンプリングして配置する。ロボット基準なので、ロボットの初期 yaw を全周
    ランダム化しても「ボールは必ず視野内 (half_angle ≤ FOV)」が保証される。

    Note: reset_base (ロボットの姿勢リセット) より後に実行されること
    (EventManager は cfg の定義順に reset イベントを適用する)。
    """
    new_pos = _place_ball_in_front_cone(
        env, env_ids, dist_range, half_angle, ball_radius, robot_cfg, ball_cfg
    )
    # 蹴り検出用のスポーン位置を記録し、再配置カウントダウンを解除
    spawn, cd = _ball_tracking_buffers(env)
    spawn[env_ids] = new_pos
    cd[env_ids] = -1


def relocate_ball_after_kick(
    env: "ManagerBasedEnv",
    command_name: str = "kick_direction",
    kick_dist_threshold: float = 0.3,
    delay_steps: int = 150,
    dist_range: tuple[float, float] = (0.6, 2.0),
    half_angle: float = 1.0,
    ball_radius: float = 0.11,
    standoff: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールを蹴れたら ``delay_steps`` 後にボールだけ再配置する疑似報酬項 (常に 0 を返す)。

    walk_kick の ``reset_ball_after_kick`` と同じ「毎ステップ呼ばれる場所」として
    RewardManager を使うためのハック。**RewTerm(weight=1.0) で登録すること**
    (weight=0 だと RewardManager にスキップされて実行されない)。

    毎ステップの処理:
        1. 蹴り検出: ボールが直近スポーン位置から ``kick_dist_threshold`` [m] 以上
           動いたらカウントダウン開始 (接触センサー不要の変位ベース検出)。
        2. カウントダウンが 0 になったら、ロボットの現在の向き基準の正面扇形に
           ボールを再配置し、キック方向コマンドを再サンプリングする。
           エピソードはリセットしない (残り時間で次の回り込みを練習できる)。
        3. ボールの瞬間移動でポテンシャル報酬 (standoff_point_progress) の
           「前回距離」が古くなり偽スパイクが出るので、バッファを新しい距離で上書き。

    タイムアウトや転倒による通常のエピソードリセットには関与しない (その場合は
    reset_ball_in_front_cone がスポーン位置とカウントダウンを初期化する)。
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    spawn, cd = _ball_tracking_buffers(env)

    # 1. 蹴り検出 → カウントダウン開始
    moved = torch.norm(ball.data.root_pos_w[:, :2] - spawn, dim=1) > kick_dist_threshold
    cd[moved & (cd < 0)] = int(delay_steps)

    # 2. カウントダウン進行 → 0 になった env のボールを再配置
    cd[cd > 0] -= 1
    fire = cd == 0
    if fire.any():
        env_ids = fire.nonzero(as_tuple=False).squeeze(-1)
        new_pos = _place_ball_in_front_cone(
            env, env_ids, dist_range, half_angle, ball_radius, robot_cfg, ball_cfg
        )
        spawn[env_ids] = new_pos
        cd[env_ids] = -1

        # キック方向を再抽選 (しないと既に揃った向きのまま「突っ込むだけ」になり、
        # 2 回目以降の回り込み練習にならない)
        env.command_manager.get_term(command_name)._resample(env_ids)

        # 3. ポテンシャル報酬の前回距離バッファを新しい目標点距離で上書き (偽スパイク防止)
        prev = getattr(env, "_prev_standoff_dist", None)
        if prev is not None and prev.shape == (env.num_envs,):
            kick_dir = env.command_manager.get_term(command_name).command[env_ids]  # (n, 2)
            target = new_pos - standoff * kick_dir
            prev[env_ids] = torch.norm(target - robot.data.root_pos_w[env_ids, :2], dim=1)

    return torch.zeros(env.num_envs, device=env.device)


def reset_ball_last_seen(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
):
    """リセットされた env の ``_ball_last_seen_pos_b`` バッファ (hold-last-seen 観測) を 0 にする。

    バッファ実体は観測関数 :func:`mdp.observations.ball_pos_rel_fov` が遅延生成するので、
    まだ無ければ no-op。リセットイベントは観測計算より先に走るため、ここで 0 化しても
    ボールが視野内にスポーンする限り新エピソード最初の観測で即座に真値へ上書きされる。
    """
    buf = getattr(env, "_ball_last_seen_pos_b", None)
    if buf is None:
        return
    if env_ids is None:
        buf.zero_()
        return
    buf[env_ids] = 0.0
