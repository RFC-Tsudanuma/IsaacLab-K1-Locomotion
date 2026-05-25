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
    forward_range: tuple[float, float] = (0.3, 0.6),
    lateral_range: tuple[float, float] = (-0.2, 0.2),
    ball_radius: float = 0.11,
) -> None:
    """ボールをロボットの前方にリセットする。

    ロボットのヨー角に基づいてロボット座標系で forward_range, lateral_range を
    サンプリングし、ワールド座標に変換してボールを配置する。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    n = len(env_ids)

    robot_pos_w = robot.data.root_pos_w[env_ids]
    robot_quat_w = robot.data.root_quat_w[env_ids]

    # ロボットのヨー角を取得
    w, x, y, z = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # ロボット座標系でサンプリング
    fwd = torch.empty(n, device=env.device).uniform_(*forward_range)
    lat = torch.empty(n, device=env.device).uniform_(*lateral_range)

    # ワールド座標に変換
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx = fwd * cos_yaw - lat * sin_yaw
    dy = fwd * sin_yaw + lat * cos_yaw

    state = ball.data.root_state_w[env_ids].clone()
    state[:, 0] = robot_pos_w[:, 0] + dx
    state[:, 1] = robot_pos_w[:, 1] + dy
    state[:, 2] = ball_radius
    state[:, 7:] = 0.0  # 速度ゼロ

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
