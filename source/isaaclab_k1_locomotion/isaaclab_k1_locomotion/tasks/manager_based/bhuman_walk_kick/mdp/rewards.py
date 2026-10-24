# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""bhuman_walk_kick の報酬: B-Human ``t1_ball.py`` のボール関係の報酬 (``_reward_ball_*``) と
歩行周波数の範囲外ペナルティ (``_reward_gait_phase_factor``)。

姿勢・関節・足の運びの報酬は土台の walk (K1FlatEnvCfg) のものをそのまま使うので、ここには
置かない (B-Human の同種の項は使わない)。重みは呼び出し側 (RewTerm.weight) で掛ける。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .kick_command import _yaw_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .kick_command import BHumanKickCommand


def _kick(env: ManagerBasedRLEnv, command_name: str = "kick") -> BHumanKickCommand:
    term = env.command_manager.get_term(command_name)
    term.ensure_updated()
    return term


def gait_phase_factor(env: ManagerBasedRLEnv, limit: tuple[float, float] = (-0.5, 0.5)) -> torch.Tensor:
    a = env.action_manager.action[:, -1]
    return torch.square(a - a.clamp(*limit))


# --------------------------------------------------------------------------- #
# ボール
# --------------------------------------------------------------------------- #
def _walk_target(k: BHumanKickCommand) -> torch.Tensor:
    """B-Human ``_helper_walk_target``: キック立ち位置 (蹴り足側にずらす) と回り込み点の混合。"""
    cfg = k.cfg
    kick_pose_in_robot = quat_apply_inverse(k.base_quat_z, k.ball_kick_pose - k.base_pos)
    extra = ((k.kick_pose_translation_ratio - 0.5) / 0.5).clip(min=0.0) * (cfg.ball_radius + 0.1)
    x_off = -torch.sin(k.direction_angle_current) * (cfg.kick_pose_y_off + extra)
    y_off = torch.cos(k.direction_angle_current) * (cfg.kick_pose_y_off + extra)
    left = kick_pose_in_robot.clone()
    right = kick_pose_in_robot.clone()
    left[:, 0] += x_off
    left[:, 1] += y_off
    right[:, 0] -= x_off
    right[:, 1] -= y_off
    walk_target = torch.where((k.filtered_ball_pos[:, 1] < 0).unsqueeze(-1), left, right)

    gdir = k.robot_direction_ref + k.direction_angle * torch.pi
    dq = _yaw_quat(gdir)
    bir = quat_apply_inverse(dq, k.ball_pos_w - k.base_pos)
    ratio = torch.where(k.ball_close_to_kick_pose < 1, 0.0, (-bir[:, 0] / 0.3).clip(0.0, 1.0))
    bir[:, 0] += (torch.abs(bir[:, 1]) / 0.3).clip(0.0, 1.0) * -0.2 + torch.max(
        -bir[:, 0], 1.0 - torch.abs(bir[:, 1]) / 0.3
    ).clip(0.0, 1.0) * 0.3
    bir[:, 1] += torch.where(k.filtered_ball_pos[:, 1] < 0, 1.0, -1.0) * cfg.ball_avoidance_range
    side_target = quat_apply_inverse(k.base_quat_z, quat_apply(dq, bir))
    return walk_target * (1.0 - ratio).unsqueeze(-1) + side_target * ratio.unsqueeze(-1)


def _ball_vision_cone(k: BHumanKickCommand, sigma: float) -> torch.Tensor:
    bir = quat_apply_inverse(k.base_quat_z, k.ball_pos_w - k.base_pos)
    return torch.exp(-torch.square(torch.atan2(bir[:, 1], bir[:, 0])) / sigma)


def ball_kick_direction(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    return _kick(env, command_name).ball_kick_direction_reward


def ball_kick_velocity(env: ManagerBasedRLEnv, sigma_scale: float = 1.0, command_name: str = "kick") -> torch.Tensor:
    k = _kick(env, command_name)
    vel_diff = k.expected_vel() - k.ball_velocity_norm_at_kick
    reward = torch.exp(-torch.square(vel_diff) / sigma_scale) * (
        k.ball_kicked_counter > k.cfg.ball_kicked_start_counter
    ).float()
    d = k.ball_kick_direction_reward
    return torch.where(d < 0, d * 0.1 * k.ball_velocity_norm_at_kick, reward * d)


def ball_kick_velocity_strong(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    k = _kick(env, command_name)
    d = k.ball_kick_direction_reward
    reward = k.ball_velocity_norm_at_kick * torch.where(d < 0, d * 0.1, d)
    return reward * k.flag_is_strong_kick.float()


def ball_walk_speed(
    env: ManagerBasedRLEnv,
    vision_sigma: float = 0.75,
    direction_sigma: float = 0.75,
    clip_max_speed_near_ball: float = 0.5,
    negative_walk_speed_factor: float = 1.0,
    command_name: str = "kick",
) -> torch.Tensor:
    """B-Human ``_reward_ball_walk_speed``。式の癖 (clip(min=1, max=0) と演算子の優先順位) もそのまま。"""
    k = _kick(env, command_name)
    allow_positive = torch.where(k.ball_kick_direction_reward < 0, 0.0, 1.0)

    v = k.filtered_lin_vel[:, :3].clone()
    v[:, 2] = k.filtered_ang_vel[:, 2]
    too_much_rotation_ratio = 1.0 - ((torch.abs(v[:, 2]) - 4.0) / 0.5).clip(min=0.0)

    ball_vision_factor = (_ball_vision_cone(k, vision_sigma) - 0.5) * 2.0
    min_rotation_negative_factor = torch.min(
        -((torch.abs(v[:, 2]) - 0.3).clip(0.0, 1.0)), ball_vision_factor.clip(max=0.0)
    )
    # B-Human: ((x + 0.5) / 0.25).clip(min=1.0, max=0.0) は常に 0 → scaling = 1
    ball_vision_factor_scaling = 1.0 - torch.clamp((ball_vision_factor + 0.5) / 0.25, min=1.0, max=0.0)
    max_rotation_positive_factor = ball_vision_factor_scaling + (1.0 - ball_vision_factor_scaling) * (
        0.25 + 0.75 * (torch.norm(k.ball_pos, dim=-1) - 0.4 / 0.3).clip(0.0, 1.0)
    )
    rotation_sign_factor = torch.where(
        v[:, 2] * k.ball_pos[:, 1] >= 0.0, allow_positive * max_rotation_positive_factor, min_rotation_negative_factor
    )

    target = _walk_target(k)
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    walk_angle = torch.atan2(k.filtered_lin_vel[:, 1], k.filtered_lin_vel[:, 0])
    angle_diff = wrap_to_pi(target_angle - walk_angle)
    walk_dir = (torch.exp(-torch.square(angle_diff) / direction_sigma) - 0.5) * 2.0
    walk_dir = torch.where(walk_dir < 0, -1.0, walk_dir)
    walk_dir = walk_dir * torch.where(
        walk_dir < 0, (ball_vision_factor_scaling * 3.0 + 1.0) * negative_walk_speed_factor, 1.0
    )
    walk_dir = torch.where(k.ball_close_to_kick_pose < 1, walk_dir.clip(min=-1.0), walk_dir * allow_positive)

    v[:, 0] = torch.abs(v[:, 0])
    v[:, 1] = torch.abs(v[:, 1])
    v[:, 2] = torch.abs(v[:, 2]) * rotation_sign_factor
    v[:, 2] = v[:, 2] * torch.where(v[:, 2] < 0, 1.0, too_much_rotation_ratio)

    ball_distance = torch.norm(k.ball_pos[:, :2], dim=-1)
    ratio = torch.where((k.ball_close_to_kick_pose < 1.0) & (ball_distance < 0.6), 0.0, 1.0)
    cap = clip_max_speed_near_ball * (1.0 - ratio) + 3 * ratio
    v[:, 0] = torch.where(walk_dir < 0, v[:, 0], torch.minimum(torch.abs(v[:, 0]), cap))
    v[:, 1] = torch.where(walk_dir < 0, v[:, 1], torch.minimum(torch.abs(v[:, 1]), cap))
    rot_cap = 3 * ((ratio - 0.5) * 2).clip(min=0.0)
    v[:, 2] = torch.where(v[:, 2] < 0, v[:, 2], torch.minimum(torch.abs(v[:, 2]), rot_cap))

    return torch.norm(v[:, :2], dim=-1) * walk_dir + v[:, 2]


def ball_walk_target_overshoot(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    k = _kick(env, command_name)
    return (
        (k.ball_walk_target_overshoot.clip(min=0.0) / 10.0)
        * (1.0 - ((k.ball_velocity_norm - 0.3) / 0.7).clip(0.0, 1.0))
        * (1.0 - (k.ball_kicked_counter > 0).float())
    )


def ball_sole_yaw(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    k = _kick(env, command_name)
    f = torch.square(k.inside_kick_pose_counter)
    f[k.flag_allow_deviation] = 1.0
    return f * k.kick_sole_yaw_ratio * (k.ball_kick_direction_reward > 0).float()


# --------------------------------------------------------------------------- #
# B-Human に無い項 (Stage 1 Boot だけで使う)
# --------------------------------------------------------------------------- #
def ball_progress(env: ManagerBasedRLEnv, lateral_weight: float = 0.5, max_speed: float = 4.0,
                  command_name: str = "kick") -> torch.Tensor:
    """蹴った後 (ball_kicked_counter > 0) の「キックで変わったボール速度」の
    「目標方向の成分 - 横成分 × lateral_weight」。

    変化量 (今の速度 - 触れる直前の速度) で見るのは、半分の env ではボールが転がった状態から
    始まり、転がってきたボールがロボットに当たっただけでもキック扱いになるため。速度そのものを
    使った版はそれに引きずられて平均が負のままだった (2026-10-21)。

    B-Human のキック報酬は「立ち位置で約 0.8 秒止まってから、目標から 48° 以内に蹴った」ときだけ
    正になるので、ボールを押し出す程度の弱いキック (向きが 55° ずれていた) には何も払われず、
    キックが学習の中で一度も強化されなかった。弱くても目標方向へ転がせば払う項を足して、
    キックの動きそのものを先に覚えさせる。
    """
    k = _kick(env, command_name)
    a = k.direction_angle_current
    dv_w = k.ball_vel_w[:, :3] - k.ball_vel_w_before_kick
    v = quat_apply_inverse(k.base_quat_z, dv_w)[:, :2]
    along = v[:, 0] * torch.cos(a) + v[:, 1] * torch.sin(a)
    lateral = torch.abs(-v[:, 0] * torch.sin(a) + v[:, 1] * torch.cos(a))
    r = along.clamp(-1.0, max_speed) - lateral_weight * lateral
    return r * (k.ball_kicked_counter > 0).float()
