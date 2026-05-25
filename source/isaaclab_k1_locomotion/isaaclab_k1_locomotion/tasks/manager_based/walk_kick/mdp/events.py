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
    dist_range: tuple[float, float] = (0.3, 0.8),
    half_angle: float = math.pi / 3,
    ball_radius: float = 0.11,
) -> None:
    """ボールをロボット前方コーン内にリセットする。

    ロボットの向きを基準に ±half_angle のコーン内、dist_range の距離でランダム配置する。
    リセットイベント時は default_root_state のヨー角を使用する。
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene["robot"]
    n = len(env_ids)

    env_origins = env.scene.env_origins[env_ids]

    robot_default = robot.data.default_root_state[env_ids]
    robot_reset_x = env_origins[:, 0] + robot_default[:, 0]
    robot_reset_y = env_origins[:, 1] + robot_default[:, 1]

    qw, qx, qy, qz = robot_default[:, 3], robot_default[:, 4], robot_default[:, 5], robot_default[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    dist = torch.empty(n, device=env.device).uniform_(*dist_range)
    angle_offset = torch.empty(n, device=env.device).uniform_(-half_angle, half_angle)
    target_angle = yaw + angle_offset

    state = ball.data.default_root_state[env_ids].clone()
    state[:, :3] += env_origins
    state[:, 0] = robot_reset_x + dist * torch.cos(target_angle)
    state[:, 1] = robot_reset_y + dist * torch.sin(target_angle)
    state[:, 2] = ball_radius
    state[:, 7:] = 0.0

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
