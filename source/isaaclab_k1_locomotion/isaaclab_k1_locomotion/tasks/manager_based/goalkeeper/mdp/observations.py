# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク専用の観測関数。

座標系の規約 (goalkeeper_env_cfg.py と共有):
    * 「ゴール座標系」= env ローカル座標 (env origin がゴール中央・ゴールライン上)。
      +x がフィールド側 (ボールが来る方向)、y がゴールライン方向。
    * ゴールライン: x = 0。失点判定はボール中心 x < -(ボール半径)。
    * ロボットはゴール中央 (原点付近) に +x 向きでスポーンする。

ステージ間で観測次元を固定するため、ボール観測 (相対位置・速度) はステージ1
(ボールなし) でもスロットを確保し、非アクティブ時は 0 (ダミー値) を返す。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# per-env 状態バッファ (遅延生成)
# ---------------------------------------------------------------------------

def gk_buffers(env: "ManagerBasedRLEnv") -> dict[str, torch.Tensor]:
    """goalkeeper タスクの per-env 状態バッファを (無ければ生成して) 返す。

    - ``_gk_target_y``      : ステージ1 のランダム目標 y [m] (ゴール座標系)
    - ``_gk_ball_active``   : ボールがアクティブ (発射済み) か。False = ステージ1 の
                              パーク状態 (観測はダミー 0)
    - ``_gk_touched``       : このエピソードでロボットがボールに触れたか
    - ``_gk_touch_rewarded``: save_touch_bonus を既に払ったか (一回限りの報酬用)
    - ``_gk_save_cd``       : セーブ成功終了までのカウントダウン [-1=非発火]
    - ``_gk_hold_ctr``      : ステージ1 の目標保持カウンタ
    """
    n = env.num_envs
    if getattr(env, "_gk_target_y", None) is None or env._gk_target_y.shape != (n,):
        env._gk_target_y = torch.zeros(n, device=env.device)
        env._gk_ball_active = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_touched = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_touch_rewarded = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._gk_save_cd = torch.full((n,), -1, dtype=torch.long, device=env.device)
        env._gk_hold_ctr = torch.zeros(n, dtype=torch.long, device=env.device)
    return {
        "target_y": env._gk_target_y,
        "ball_active": env._gk_ball_active,
        "touched": env._gk_touched,
        "touch_rewarded": env._gk_touch_rewarded,
        "save_cd": env._gk_save_cd,
        "hold_ctr": env._gk_hold_ctr,
    }


# ---------------------------------------------------------------------------
# 座標ヘルパ
# ---------------------------------------------------------------------------

def ball_pos_goal(env: "ManagerBasedRLEnv", ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball")) -> torch.Tensor:
    """ボール位置をゴール座標系 (env origin 基準) で返す (N, 3)。"""
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w[:, :3] - env.scene.env_origins


def robot_pos_goal(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ロボット base 位置をゴール座標系で返す (N, 3)。"""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins


def compute_target_y(
    env: "ManagerBasedRLEnv",
    max_y: float = 1.25,
    approach_vx_threshold: float = -0.05,
) -> torch.Tensor:
    """ロボットが向かうべき目標 y 座標 (ゴール座標系) を返す (N,)。

    * ボール非アクティブ (ステージ1): ``_gk_target_y`` バッファのランダム目標。
    * ボールがゴールへ接近中 (vx < approach_vx_threshold): 現在のボール位置・速度から
      ゴールライン (x=0) への到達 y を外挿して返す (±max_y にクランプ)。
      転がりの減速は進行方向に沿って一様に効くため軌道は直線のままであり、
      「到達するなら到達点 y」は等速外挿と一致する (減速で届かない場合も
      目標としては正しい遮断位置になる)。
    * ボールが接近していない (弾いた後・停止後): ゴール中央 0 (復帰)。

    観測と報酬の両方からこの関数で同じ目標を参照する (整合を一箇所に集約)。
    """
    bufs = gk_buffers(env)
    pos = ball_pos_goal(env)
    ball: RigidObject = env.scene["soccer_ball"]
    vel = ball.data.root_com_vel_w[:, :3]

    approaching = vel[:, 0] < approach_vx_threshold
    # x=0 到達までの時間 (接近中のみ意味を持つ。ゼロ割り防止で clamp)
    t = pos[:, 0] / (-vel[:, 0]).clamp(min=1e-3)
    y_pred = (pos[:, 1] + vel[:, 1] * t).clamp(-max_y, max_y)

    target = torch.where(approaching, y_pred, torch.zeros_like(y_pred))
    return torch.where(bufs["ball_active"], target, bufs["target_y"])


# ---------------------------------------------------------------------------
# 観測項
# ---------------------------------------------------------------------------

def gk_ball_pos_rel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ボール相対位置 (base yaw frame, 2D)。ボール非アクティブ時はダミー 0。"""
    bufs = gk_buffers(env)
    ball: RigidObject = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    offset_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    offset_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), offset_w)[:, :2]
    return torch.where(bufs["ball_active"].unsqueeze(1), offset_b, torch.zeros_like(offset_b))


def gk_ball_vel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ボール速度 (base yaw frame, 2D)。ボール非アクティブ時はダミー 0。"""
    bufs = gk_buffers(env)
    ball: RigidObject = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    vel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), ball.data.root_com_vel_w[:, :3])[:, :2]
    return torch.where(bufs["ball_active"].unsqueeze(1), vel_b, torch.zeros_like(vel_b))


def gk_ball_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """ボールがアクティブ (発射済み) かのフラグ (N, 1)。ステージ1 では常に 0。"""
    return gk_buffers(env)["ball_active"].float().unsqueeze(1)


def gk_target_y(env: "ManagerBasedRLEnv", max_y: float = 1.25) -> torch.Tensor:
    """目標 y 座標 (ゴール座標系) の観測 (N, 1)。:func:`compute_target_y` 参照。"""
    return compute_target_y(env, max_y=max_y).unsqueeze(1)


def gk_self_state(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """自機のゴール座標系状態 (N, 4): (x オフセット, y オフセット, sin(yaw), cos(yaw))。

    yaw=0 はフィールド側 (+x, ボールが来る方向) を向いた姿勢。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    pos = robot_pos_goal(env, asset_cfg)
    heading = robot.data.heading_w
    return torch.stack([pos[:, 0], pos[:, 1], torch.sin(heading), torch.cos(heading)], dim=1)
