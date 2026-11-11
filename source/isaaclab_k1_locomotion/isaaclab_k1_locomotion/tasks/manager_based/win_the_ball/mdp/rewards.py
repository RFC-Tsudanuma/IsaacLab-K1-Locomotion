# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""win_the_ball 固有の報酬。"""

import math
import torch

from ...walk_kick.mdp.kick_state import _foot_body_ids, kick_state


def quick_accurate_kick(
    env,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    time_scale: float = 4.0,
    sigma_direction: float = 0.35,
    sigma_velocity: float = 0.35,
) -> torch.Tensor:
    """成立した瞬間だけ、早く正確なキックほど大きいエピソード報酬を与える。"""
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    device = state["kick_done"].device
    rewarded = getattr(env, "_win_ball_quick_kick_rewarded", None)
    if rewarded is None or rewarded.shape[0] != env.num_envs:
        rewarded = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env._win_ball_quick_kick_rewarded = rewarded
        env._win_ball_kick_time = torch.zeros(env.num_envs, device=device)

    fresh = env.episode_length_buf <= 1
    rewarded[fresh] = False
    env._win_ball_kick_time[fresh] = 0.0

    just_kicked = state["kick_done"] & (~rewarded)
    elapsed = env.episode_length_buf.float() * env.step_dt
    env._win_ball_kick_time = torch.where(just_kicked, elapsed, env._win_ball_kick_time)
    rewarded |= state["kick_done"]

    time_quality = torch.exp(-elapsed / time_scale)
    direction_quality = torch.exp(-((state["tau_direction_frozen"] / sigma_direction) ** 2))
    velocity_error = state["v_ball_frozen"] - state["v_target"]
    velocity_quality = torch.exp(-((velocity_error / sigma_velocity) ** 2))
    # RewardManager の dt 乗算を打ち消し、weight をエピソード単位の最大ボーナスにする。
    return just_kicked.float() * time_quality * direction_quality * velocity_quality / env.step_dt


def direction_matched_foot_kick(
    env,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    sigma_velocity: float = 0.35,
    center_deadzone_deg: float = 5.0,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """右方向は右足、左方向は左足で正確に蹴った場合に追加報酬を与える。

    正面のdeadzone内では左右どちらも同点にして、不必要な左右バイアスを作らない。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
    )
    command = env.command_manager.get_term(command_name)
    env_ids = torch.arange(env.num_envs, device=state["kick_done"].device)
    relative_angle = command._relative_command_angle(env_ids)
    kick_done = state["kick_done"].float()
    used_right = state["kick_foot_frozen"] == 1.0
    preferred_right = relative_angle < 0.0
    center = relative_angle.abs() <= math.radians(center_deadzone_deg)
    foot_match = (center | (used_right == preferred_right)).float()
    direction_quality = torch.exp(-((state["tau_direction_frozen"] / sigma_direction) ** 2))
    velocity_error = state["v_ball_frozen"] - state["v_target"]
    velocity_quality = torch.exp(-((velocity_error / sigma_velocity) ** 2))
    return kick_done * foot_match * direction_quality * velocity_quality


def non_forward_facing_error(env, command_name: str = "kick_direction") -> torch.Tensor:
    """側方・後方指令中の開始時正面からのヨーずれを返す。負の重みで使う。"""
    command = env.command_manager.get_term(command_name)
    keep_facing = (command._is_side_command | command._is_rear_command).float()
    robot_quat = env.scene["robot"].data.root_quat_w
    qw, qx, qy, qz = (
        robot_quat[:, 0],
        robot_quat[:, 1],
        robot_quat[:, 2],
        robot_quat[:, 3],
    )
    current_yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    yaw_error = torch.atan2(
        torch.sin(current_yaw - command._reference_yaw),
        torch.cos(current_yaw - command._reference_yaw),
    )
    return (1.0 - torch.cos(yaw_error)) * keep_facing


def side_foot_kick(
    env,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    sigma_velocity: float = 0.35,
    side_saturation: float = 0.34,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """正面姿勢のまま、足の側面で側方へ蹴った動作を評価する。"""
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    command = env.command_manager.get_term(command_name)
    is_side = command._is_side_command.float()
    direction_quality = torch.exp(-((state["tau_direction_frozen"] / sigma_direction) ** 2))
    velocity_error = state["v_ball_frozen"] - state["v_target"]
    velocity_quality = torch.exp(-((velocity_error / sigma_velocity) ** 2))
    span = max(1.0 - side_saturation, 1e-6)
    contact_quality = torch.clamp(
        (1.0 - state["foot_kick_dot_frozen"].abs()) / span, min=0.0, max=1.0
    )
    measured = (state["touch_count"] > 0.0).float()
    return (
        is_side
        * state["kick_done"].float()
        * direction_quality
        * velocity_quality
        * contact_quality
        * measured
    )


def side_ball_orthogonality(
    env,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_dot: float = 0.25,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """側方キック時、胴体正面と実際のボール飛翔方向が直交するほど報酬を与える。"""
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    command = env.command_manager.get_term(command_name)
    kick_yaw = torch.atan2(command.command[:, 0], command.command[:, 1])
    # tau_signed > 0 はボールが指令方向より右なので、実球方向はkick_yawより時計回り。
    ball_yaw = kick_yaw - state["tau_signed_frozen"]

    # 開始時正面を使い、キック後に身体だけ回して直交報酬を稼ぐ抜け道を作らない。
    forward_dot_ball = torch.cos(ball_yaw - command._reference_yaw).abs()
    orthogonal_quality = torch.exp(-((forward_dot_ball / sigma_dot) ** 2))
    env._side_ball_orthogonality_quality = orthogonal_quality
    return command._is_side_command.float() * state["kick_done"].float() * orthogonal_quality


def rear_foot_over_ball_motion(
    env,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    min_clearance: float = 0.15,
    cross_distance: float = 0.13,
    lateral_tolerance: float = 0.18,
    straight_rear_halfwidth_deg: float = 20.0,
    straight_rear_gain: float = 3.0,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """後方ヒールキックの「上げる→跨ぐ→踵を戻す」を速度と通過イベントで誘導する。

    真後ろでは斜め後ろより跨ぎが難しいため、180度からの角度差に応じて報酬を増幅する。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    command = env.command_manager.get_term(command_name)
    is_rear = command._is_rear_command
    robot = env.scene["robot"]
    ball = env.scene["soccer_ball"]
    device = robot.data.root_pos_w.device
    env_ids = torch.arange(env.num_envs, device=device)

    relative_angle = command._relative_command_angle(env_ids)
    straight_rear_distance = math.pi - relative_angle.abs()
    straight_rear_quality = torch.clamp(
        1.0 - straight_rear_distance / math.radians(straight_rear_halfwidth_deg),
        min=0.0,
        max=1.0,
    )
    rear_reward_scale = 1.0 + straight_rear_gain * straight_rear_quality
    # 右後方は右足、左後方は左足。足bodyの順序は[left, right]。
    kick_foot = (relative_angle < 0.0).long()
    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[:, foot_ids, :][env_ids, kick_foot]

    reference_forward = torch.stack(
        [torch.cos(command._reference_yaw), torch.sin(command._reference_yaw)], dim=-1
    )
    reference_left = torch.stack(
        [-reference_forward[:, 1], reference_forward[:, 0]], dim=-1
    )
    foot_from_ball = foot_pos[:, :2] - ball.data.root_pos_w[:, :2]
    longitudinal = (foot_from_ball * reference_forward).sum(dim=-1)
    lateral = (foot_from_ball * reference_left).sum(dim=-1).abs()
    clearance = foot_pos[:, 2] - ball.data.root_pos_w[:, 2]

    crossed = getattr(env, "_rear_foot_crossed", None)
    if crossed is None or crossed.shape[0] != env.num_envs:
        crossed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env._rear_foot_crossed = crossed
        env._rear_cross_potential_prev = torch.zeros(env.num_envs, device=device)
        env._rear_return_longitudinal_prev = longitudinal.clone()
    fresh = env.episode_length_buf <= 1
    crossed[fresh] = False

    # 足を高くしつつ前へ運ぶ目標へのポテンシャル。差分で払うため、上下往復では相殺される。
    clearance_quality = torch.clamp(
        (clearance + 0.10) / (min_clearance + 0.10), min=0.0, max=1.0
    )
    forward_quality = torch.clamp(
        (longitudinal + 0.35) / (cross_distance + 0.35), min=0.0, max=1.0
    )
    lateral_quality = torch.exp(-((lateral / lateral_tolerance) ** 2))
    cross_potential = (
        0.45 * clearance_quality + 0.55 * clearance_quality * forward_quality
    ) * lateral_quality
    potential_prev = env._rear_cross_potential_prev
    return_longitudinal_prev = env._rear_return_longitudinal_prev
    potential_prev[fresh] = cross_potential[fresh]
    return_longitudinal_prev[fresh] = longitudinal[fresh]

    crossing_now = (
        is_rear
        & (~crossed)
        & (clearance >= min_clearance)
        & (longitudinal >= cross_distance)
        & (lateral <= lateral_tolerance)
    )
    crossed |= crossing_now

    pre_cross = is_rear & (~crossed) & (~state["kick_done"])
    post_cross = is_rear & crossed & (~state["kick_done"])
    cross_progress = torch.clamp((cross_potential - potential_prev) * 20.0, min=-1.0, max=1.0)
    # 跨いだ後は前後位置の差分で後退だけを促す。前へ戻れば負になる。
    return_progress = torch.clamp(
        (return_longitudinal_prev - longitudinal) / 0.02 / 0.8, min=-1.0, max=1.0
    )

    env._rear_cross_potential_prev = cross_potential.clone()
    env._rear_return_longitudinal_prev = longitudinal.clone()
    env._rear_crossing_clearance = clearance

    return (
        pre_cross.float() * cross_progress
        + crossing_now.float()
        + post_cross.float() * return_progress
    ) * rear_reward_scale


def rear_low_crossing_penalty(
    env,
    min_clearance: float = 0.15,
    danger_longitudinal: float = 0.16,
    lateral_tolerance: float = 0.18,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """後方指令で、足を低いままボール直上へ進める動作を罰する。"""
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    ball = env.scene["soccer_ball"]
    device = robot.data.root_pos_w.device
    env_ids = torch.arange(env.num_envs, device=device)
    relative_angle = command._relative_command_angle(env_ids)
    kick_foot = (relative_angle < 0.0).long()
    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[:, foot_ids, :][env_ids, kick_foot]

    reference_forward = torch.stack(
        [torch.cos(command._reference_yaw), torch.sin(command._reference_yaw)], dim=-1
    )
    reference_left = torch.stack(
        [-reference_forward[:, 1], reference_forward[:, 0]], dim=-1
    )
    foot_from_ball = foot_pos[:, :2] - ball.data.root_pos_w[:, :2]
    longitudinal = (foot_from_ball * reference_forward).sum(dim=-1)
    lateral = (foot_from_ball * reference_left).sum(dim=-1).abs()
    clearance = foot_pos[:, 2] - ball.data.root_pos_w[:, 2]
    crossed = getattr(env, "_rear_foot_crossed", torch.zeros_like(command._is_rear_command))

    horizontal_danger = (longitudinal.abs() < danger_longitudinal) & (lateral < lateral_tolerance)
    low_fraction = torch.clamp((min_clearance - clearance) / min_clearance, min=0.0, max=1.0)
    penalty = command._is_rear_command.float() * (~crossed).float() * horizontal_danger.float() * low_fraction
    env._rear_low_crossing_danger = penalty
    return penalty


def rear_heel_kick(
    env,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    sigma_velocity: float = 0.35,
    command_name: str = "kick_direction",
) -> torch.Tensor:
    """後方へ正確な速度で、踵側から当てたキックを評価する。"""
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    command = env.command_manager.get_term(command_name)
    is_rear = command._is_rear_command.float()
    kick_done = state["kick_done"].float()
    direction_quality = torch.exp(-((state["tau_direction_frozen"] / sigma_direction) ** 2))
    velocity_error = state["v_ball_frozen"] - state["v_target"]
    velocity_quality = torch.exp(-((velocity_error / sigma_velocity) ** 2))
    # foot_kick_dot=-1で踵が指令方向を向く。0以上（側面～つま先）は無得点。
    heel_quality = torch.clamp(-state["foot_kick_dot_frozen"], min=0.0, max=1.0)
    measured = (state["touch_count"] > 0.0).float()
    return (
        is_rear
        * kick_done
        * direction_quality
        * velocity_quality
        * heel_quality
        * measured
    )