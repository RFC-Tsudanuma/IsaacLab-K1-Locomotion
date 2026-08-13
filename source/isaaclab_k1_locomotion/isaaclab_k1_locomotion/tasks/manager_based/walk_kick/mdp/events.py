from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_in_front_of_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    dist_range: tuple[float, float] = (0.5, 0.8),
    half_angle: float = math.pi / 3,
    ball_radius: float = 0.11,
    spawn_clearance: float = 0.0,
    speed_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """ボールをロボット前方コーン内にリセットする。

    ロボットの実際のリセット後の姿勢を基準に、±half_angle のコーン内・dist_range の距離で
    ランダム配置する。``speed_range`` を与えると水平方向のランダム初速も乗せる。

    NOTE: 基準は ``default_root_state`` ではなく ``root_pos_w`` / ``root_quat_w``。
          reset_base (reset_root_state_uniform) がロボットを default から x,y ±0.5m・
          yaw ±π だけランダムにずらすため、default を基準にするとボールがロボットから見て
          まったく別の場所 (最悪は足の間) に湧く。このイベントは reset_base より後に走り
          (``__post_init__`` で後から追加されるので ``__dict__`` の末尾に入る)、
          ``write_root_link_pose_to_sim`` が内部バッファを即時更新するので、ここでは
          リセット済みの姿勢が読める。

    NOTE: dist_range の下限はロボットの足がボールに初期接触しない距離にすること。
          距離は base 中心からなので、ball_radius (0.11m) と足の張り出し (~0.15m) を
          考えると 0.3m では正面に湧いたときに接触する。

    Args:
        spawn_clearance: 接地高さからさらに浮かせる量 [m]。凹凸地形 (terrain generator) では
            env origin の z が「サブ地形中央付近の最大高さ」なので、そこから離れた xy では
            地面がそれより数 cm 低いことも高いこともある。地面より下に湧くとボールが
            めり込んだまま弾かれるため、地形の起伏ぶんだけ浮かせて落下させる。
            **平坦地形では 0.0 のままにすること** (既定値。既存タスクの挙動を変えない)。
        speed_range: 水平初速の大きさ [m/s] のサンプリング範囲。方向は 0-2π の一様乱数。
            既定 ``(0.0, 0.0)`` は静止配置 = 既存タスクの挙動。

            NOTE: 上限は必ず kick_state の ``v_thresh`` (既定 0.8 m/s) より **十分下**に
                  すること。初速が閾値を超えると、ロボットが触る前に値 latch が成立して
                  「蹴っていないのに τ_direction が凍結される」ことになる。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    n = len(env_ids)

    # リセット後の実際のロボット位置・向き
    robot_pos_w = robot.data.root_pos_w[env_ids]
    quat = robot.data.root_quat_w[env_ids]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    dist = torch.empty(n, device=env.device).uniform_(*dist_range)
    angle_offset = torch.empty(n, device=env.device).uniform_(-half_angle, half_angle)
    target_angle = yaw + angle_offset

    state = ball.data.default_root_state[env_ids].clone()
    state[:, 0] = robot_pos_w[:, 0] + dist * torch.cos(target_angle)
    state[:, 1] = robot_pos_w[:, 1] + dist * torch.sin(target_angle)
    # z の基準は env origin (= 地形原点)。terrain_type="plane" では 0.0 なので、
    # 平坦タスクではこれまでどおり絶対高さ ball_radius に置かれる。
    state[:, 2] = env.scene.env_origins[env_ids, 2] + ball_radius + spawn_clearance
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0

    # 水平初速 (転がるボール用)。既定 (0.0, 0.0) では静止のまま。
    if speed_range[1] > 0.0:
        speed = torch.empty(n, device=env.device).uniform_(*speed_range)
        vel_angle = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * math.pi)
        state[:, 7] = speed * torch.cos(vel_angle)
        state[:, 8] = speed * torch.sin(vel_angle)

    ball.write_root_state_to_sim(state, env_ids=env_ids)


def perturb_ball_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    distance_range: tuple[float, float] = (0.01, 0.05),
) -> None:
    """ボールをランダムな方向に distance_range [m] だけずらす。"""
    ball = env.scene[ball_cfg.name]
    n = len(env_ids)

    state = ball.data.root_state_w[env_ids].clone()

    dist = torch.empty(n, device=env.device).uniform_(*distance_range)
    angle = torch.empty(n, device=env.device).uniform_(0.0, 2.0 * torch.pi)
    state[:, 0] += dist * torch.cos(angle)
    state[:, 1] += dist * torch.sin(angle)

    ball.write_root_state_to_sim(state, env_ids=env_ids)
