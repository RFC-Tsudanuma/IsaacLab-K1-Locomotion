# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用の報酬関数。

exp 型の距離報酬は σ を 1 本にすると「遠い目標で勾配が消えて足踏み均衡に落ちる」
既知の問題があるため、σ の異なる 2 項 (粗い/細かい) を cfg 側で重ねて使う
(multi-scale。exp 報酬の諦め問題対策)。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from ...around_ball.mdp.observations import _high_action_cmd
from .observations import compute_target_y, gk_buffers, robot_pos_goal
from .terminations import update_save_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _target_y_error(env: "ManagerBasedRLEnv", max_y: float) -> torch.Tensor:
    """robot_y − target_y (ゴール座標系) の符号付き誤差 (N,)。"""
    return robot_pos_goal(env)[:, 1] - compute_target_y(env, max_y=max_y)


def track_target_y(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    max_y: float = 1.25,
) -> torch.Tensor:
    """目標 y への距離のガウス報酬 exp(-err²/σ²) ∈ [0, 1]。

    cfg 側で std の違う 2 項 (例: 0.5 と 0.15) を重ねてマルチスケールにする。
    目標はステージ1 = ランダム点、ステージ2 以降 = ボール到達予測点 (compute_target_y)。
    """
    err = _target_y_error(env, max_y)
    return torch.exp(-torch.square(err) / std**2)


def target_reach_velocity(
    env: "ManagerBasedRLEnv",
    deadband: float = 0.12,
    v_cap: float = 0.6,
    max_y: float = 1.25,
) -> torch.Tensor:
    """目標方向への横移動速度に比例する密報酬 ∈ [-1, 1]。

    目標が遠くても勾配が一定に出る (exp 型の勾配消失の補完)。deadband 内では
    満額 1 (到達を減点しない)。逆方向への移動は負になる。
    """
    err = _target_y_error(env, max_y)
    robot = env.scene["robot"]
    v_y = robot.data.root_lin_vel_w[:, 1]
    toward = -torch.sign(err) * v_y  # 誤差を減らす向きの速度
    r = (toward / v_cap).clamp(-1.0, 1.0)
    return torch.where(err.abs() <= deadband, torch.ones_like(r), r)


def hold_at_target(
    env: "ManagerBasedRLEnv",
    pos_std: float = 0.2,
    cmd_std: float = 0.1,
    lin_vel_std: float = 0.3,
    yaw_rate_weight: float = 0.25,
    max_y: float = 1.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """目標に到達した状態で「コマンドを 0 に落として静止」するほど高い報酬 [0, 1]。

    around_ball_ready の stop_when_ready と同じ構造 (gate × stop_cmd × stop_body)。
    stop_cmd (上位コマンドノルムのガウス) が足踏み局所最適対策の本命:
    frozen が初期姿勢で立つのはコマンドノルム < 0.05 (gait_phase ゼロ埋め閾値) の
    ときだけなので、実ベース速度だけを見る停止報酬では「コマンド 0.1〜0.3 を残した
    その場足踏み」がほぼ満額を取ってしまい、0.05 の壁を越える勾配が出ない。
    """
    robot = env.scene[robot_cfg.name]
    err = _target_y_error(env, max_y)
    gate = torch.exp(-torch.square(err) / pos_std**2)

    cmd_norm = torch.norm(_high_action_cmd(env)[:, :3], dim=1)
    stop_cmd = torch.exp(-torch.square(cmd_norm) / cmd_std**2)

    lin_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    yaw_rate = robot.data.root_ang_vel_w[:, 2].abs()
    motion = lin_speed + yaw_rate_weight * yaw_rate
    stop_body = torch.exp(-torch.square(motion) / lin_vel_std**2)

    return gate * stop_cmd * stop_body


def save_touch_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """ボールに触れて弾いた瞬間の一回限りのボーナス (エピソードにつき 1 回)。"""
    newly = update_save_state(env)
    bufs = gk_buffers(env)
    fire = newly & (~bufs["touch_rewarded"])
    bufs["touch_rewarded"][fire] = True
    return fire.float()


def return_to_center_after_save(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
) -> torch.Tensor:
    """ボールを弾いた後、ゴール中央 (y=0) へ戻るほど高い報酬 (タッチ後のみ有効)。"""
    bufs = gk_buffers(env)
    y = robot_pos_goal(env)[:, 1]
    r = torch.exp(-torch.square(y) / std**2)
    gate = bufs["ball_active"] & bufs["touched"]
    return r * gate.float()


def stay_on_goal_line(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    x_offset: float = 0.0,
) -> torch.Tensor:
    """ゴールライン近傍 (x ≈ x_offset) に留まるほど高い報酬 [0, 1]。

    横移動セーブのタスクなので前後方向はライン上が定位置。前へ飛び出す/後退して
    ゴール内に入る動きを抑える shaping。
    """
    x = robot_pos_goal(env)[:, 0]
    return torch.exp(-torch.square(x - x_offset) / std**2)


def face_field(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """フィールド側 (+x, yaw=0) を向き続けるほど高い報酬 [0, 1]。

    セーブは vy の横ステップで行う想定なので、体の向きを固定して
    「横歩き」のコマンド意味論を保つ shaping。
    """
    robot = env.scene[asset_cfg.name]
    heading = robot.data.heading_w  # env はワールド軸に沿って配置されるので yaw=0 が +x
    return torch.exp(-torch.square(heading) / std**2)
