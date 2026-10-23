# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for K1 locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_after_ball_contact(
    env: ManagerBasedRLEnv,
    time_after_contact: float = 3.0,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """ボールに最初に接触してから一定時間経過した環境を終了する。

    左右足の contact sensor を確認し、初回接触時刻を `_custom_buffers` に記録する。
    現在時刻 - 初回接触時刻 が ``time_after_contact`` を超えたら True を返す。

    リセット検知: episode_length_buf がリセットされると現在時刻が記録時刻より小さく
    なるため、その場合に記録をクリアする (event登録不要で自己復元する)。
    """
    sensor_right = env.scene.sensors["contact_balls_right"]
    sensor_left = env.scene.sensors["contact_balls_left"]
    force_right = torch.norm(sensor_right.data.force_matrix_w[:, 0, 0], p=2, dim=1)
    force_left = torch.norm(sensor_left.data.force_matrix_w[:, 0, 0], p=2, dim=1)
    has_contact = (force_right > contact_threshold) | (force_left > contact_threshold)

    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    buffer_key = "first_ball_contact_time"
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = torch.full(
            (env.num_envs,), -1.0, device=env.device
        )
    first_contact_time = env._custom_buffers[buffer_key]

    current_time = env.episode_length_buf.float() * env.step_dt

    # 環境リセット後は現在時刻が記録時刻より小さくなる → 記録を無効化
    reset_mask = current_time < first_contact_time
    first_contact_time = torch.where(
        reset_mask, torch.full_like(first_contact_time, -1.0), first_contact_time
    )

    # 未接触かつ今 接触 → 初回接触時刻として記録
    new_contact_mask = (first_contact_time < 0) & has_contact
    first_contact_time = torch.where(new_contact_mask, current_time, first_contact_time)

    env._custom_buffers[buffer_key] = first_contact_time

    elapsed = current_time - first_contact_time
    return (first_contact_time >= 0) & (elapsed > time_after_contact)


__all__ = [
    "time_after_ball_contact",
]


# ---------------------------------------------------------------------------
# ドリブル (Isaac-Velocity-Flat-K1-Dribble-Target-v0) 用の終了条件
# ---------------------------------------------------------------------------
# NOTE: ball_reached_target は DoneTerm と RewTerm (ball_reached_target_bonus) の両方から
# 同じ関数・同じ distance_threshold で参照されるので、判定基準がズレない。


def ball_reached_target(
    env: ManagerBasedRLEnv,
    command_name: str = "dribble_target",
    distance_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ボールとロボットの両方が目標点から ``distance_threshold`` 以内に入ったら成功。

    2026-10-11: ボールだけで判定すると「蹴って転がして放置」で成功できたため、
    ロボット自身も目標まで来ることを要求する。
    経路 (L字・Z字など) の途中のウェイポイントでは成功にしない (最終ウェイポイントのときだけ判定)。
    """
    cmd = env.command_manager.get_term(command_name)
    target = cmd.pos_command_w[:, :2]
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    ball_ok = torch.norm(ball.data.root_pos_w[:, :2] - target, dim=1) < distance_threshold
    robot_ok = torch.norm(robot.data.root_pos_w[:, :2] - target, dim=1) < distance_threshold
    reached = ball_ok & robot_ok
    is_final = getattr(cmd, "is_final_waypoint", None)
    if is_final is not None:
        reached = reached & is_final
    return reached


def ball_too_far(
    env: ManagerBasedRLEnv,
    max_distance: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボットとボールの xy 距離が ``max_distance`` を超えたら終了 (ボールを見失った)。"""
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    return torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1) > max_distance


def obstacle_collision(
    env: ManagerBasedRLEnv,
    obstacle_cfgs: list[SceneEntityCfg],
    robot_distance: float = 0.45,
    ball_distance: float = 0.31,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """ロボット / ボールの中心が障害物の中心から xy で 0.45m / 0.31m 以内に入ったら終了。

    0.31m = 障害物半径 0.2m + ボール半径 0.11m。障害物の無いレベルでは障害物が遠方に
    退避しているので発火しない。
    """
    robot_xy = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    ball_xy = env.scene[ball_cfg.name].data.root_pos_w[:, :2]
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in obstacle_cfgs:
        obs_xy = env.scene[cfg.name].data.root_pos_w[:, :2]
        hit |= torch.norm(robot_xy - obs_xy, dim=1) < robot_distance
        hit |= torch.norm(ball_xy - obs_xy, dim=1) < ball_distance
    return hit


def dribble_time_out(env: ManagerBasedRLEnv, command_name: str = "dribble_target") -> torch.Tensor:
    """エピソードごとの制限時間の time_out (DoneTerm(time_out=True) で使う)。

    制限時間は WaypointPose2dCommand が経路を作るときに決める:
    max(30 秒, 経路の長さ ÷ 0.5 m/s + 5 秒)、上限 60 秒 (WaypointPose2dCommandCfg.time_limit_*)。
    time_out は進級の判定で失敗に数えるので、遅くても完走できる時間にしている (2026-10-23)。
    環境の episode_length_s は time_limit_max_s 以上にしておくこと。
    """
    cmd = env.command_manager.get_term(command_name)
    return env.episode_length_buf >= torch.round(cmd.time_limit_s / env.step_dt).long()
