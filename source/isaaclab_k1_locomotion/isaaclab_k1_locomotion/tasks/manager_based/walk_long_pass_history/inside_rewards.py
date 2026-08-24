# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_history 専用のインサイドキック報酬。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from ..walk_kick.mdp.curriculums import kick_rate_gated_speed_range
from ..walk_kick.mdp.kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_FORM_STATE_ATTR = "_inside_kick_form_state"
_STAGE_STATE_ATTR = "_inside_kick_stage_state"
_FOOT_IDS_ATTR = "_inside_kick_foot_body_ids"
_NORM_EPS = 1.0e-6
_MOVING_EPS = 1.0e-3


def _kick_state(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> dict:
    """親タスクと同じパラメータで共有 latch 状態を取得する。"""
    return kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )


def _foot_body_ids(env: ManagerBasedRLEnv, robot) -> list[int]:
    """左右の足リンクを ``[left, right]`` の順で一度だけ解決する。"""
    ids = getattr(env, _FOOT_IDS_ATTR, None)
    if ids is None:
        left = robot.find_bodies("left_foot_link")[0][0]
        right = robot.find_bodies("right_foot_link")[0][0]
        ids = [left, right]
        setattr(env, _FOOT_IDS_ATTR, ids)
    return ids


def _inside_form_state(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> tuple[dict, dict]:
    """接触直前の足速度と接触時の内側面向きを latch して返す。

    接触センサーは decimation 中の衝突を取りこぼし得るため、接触判定は共有
    :func:`kick_state` の ``pre_latch_touch_event``（球速増分）をそのまま使う。
    足速度だけは一つ前の制御ステップ値、姿勢は接触を検出した現在値を保存する。
    複数回接触した場合は、ボール速度が latch 閾値を超えた時点で最後に保存された
    接触を本命キックとして凍結する。
    """
    shared = _kick_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )

    step = int(env.common_step_counter)
    form = getattr(env, _FORM_STATE_ATTR, None)
    if form is not None and form["step"] == step:
        return shared, form

    robot = env.scene["robot"]
    ball = env.scene["soccer_ball"]
    device = env.device
    foot_ids = _foot_body_ids(env, robot)
    foot_pos = robot.data.body_pos_w[:, foot_ids, :]
    foot_vel = robot.data.body_lin_vel_w[:, foot_ids, :]
    foot_quat = robot.data.body_quat_w[:, foot_ids, :]

    if form is None:
        form = {
            "step": -1,
            "prev_foot_vel": foot_vel.clone(),
            "prev_kick_done": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "last_touch_valid": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "inside_contact_cos_last_touch": torch.zeros(env.num_envs, device=device),
            "contact_local_x_last_touch": torch.zeros(env.num_envs, device=device),
            "inside_face_cos_last_touch": torch.zeros(env.num_envs, device=device),
            "swing_cos_last_touch": torch.zeros(env.num_envs, device=device),
            "swing_speed_last_touch": torch.zeros(env.num_envs, device=device),
            "form_valid_frozen": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
            "inside_contact_cos_frozen": torch.zeros(env.num_envs, device=device),
            "contact_local_x_frozen": torch.zeros(env.num_envs, device=device),
            "inside_face_cos_frozen": torch.zeros(env.num_envs, device=device),
            "swing_cos_frozen": torch.zeros(env.num_envs, device=device),
            "swing_speed_frozen": torch.zeros(env.num_envs, device=device),
        }
        setattr(env, _FORM_STATE_ATTR, form)

    just_reset = env.episode_length_buf == 1
    if just_reset.any():
        # 新エピソードの最初の速度を「前ステップ」として入れ、ゼロ由来の偽角度を防ぐ。
        form["prev_foot_vel"][just_reset] = foot_vel[just_reset]
        form["prev_kick_done"][just_reset] = False
        form["last_touch_valid"][just_reset] = False
        form["inside_contact_cos_last_touch"][just_reset] = 0.0
        form["contact_local_x_last_touch"][just_reset] = 0.0
        form["inside_face_cos_last_touch"][just_reset] = 0.0
        form["swing_cos_last_touch"][just_reset] = 0.0
        form["swing_speed_last_touch"][just_reset] = 0.0
        form["form_valid_frozen"][just_reset] = False
        form["inside_contact_cos_frozen"][just_reset] = 0.0
        form["contact_local_x_frozen"][just_reset] = 0.0
        form["inside_face_cos_frozen"][just_reset] = 0.0
        form["swing_cos_frozen"][just_reset] = 0.0
        form["swing_speed_frozen"][just_reset] = 0.0

    env_ids = torch.arange(env.num_envs, device=device)
    ball_pos = ball.data.root_pos_w[:, :3]
    nearest_foot = (foot_pos - ball_pos.unsqueeze(1)).norm(dim=-1).argmin(dim=1)
    selected_pos = foot_pos[env_ids, nearest_foot]
    selected_quat = foot_quat[env_ids, nearest_foot]
    selected_pre_vel = form["prev_foot_vel"][env_ids, nearest_foot]
    ball_in_foot = quat_apply_inverse(selected_quat, ball_pos - selected_pos)

    # URDF の中立姿勢では左足の内側が local -Y、右足の内側が local +Y。
    inside_local = torch.zeros(env.num_envs, 3, device=device, dtype=foot_pos.dtype)
    inside_local[:, 1] = torch.where(nearest_foot == 0, -1.0, 1.0)
    inside_world_xy = quat_apply(selected_quat, inside_local)[:, :2]
    inside_world_xy = inside_world_xy / inside_world_xy.norm(dim=-1, keepdim=True).clamp_min(
        _NORM_EPS
    )

    # 目標方向とは無関係に「ボールが足の内側面側にあるか」を測る。
    foot_to_ball_xy = ball_pos[:, :2] - selected_pos[:, :2]
    foot_to_ball_xy = foot_to_ball_xy / foot_to_ball_xy.norm(dim=-1, keepdim=True).clamp_min(
        _NORM_EPS
    )
    inside_contact_cos = torch.clamp(
        (inside_world_xy * foot_to_ball_xy).sum(dim=-1), -1.0, 1.0
    )

    command = env.command_manager.get_command("kick_direction")
    target_dir = torch.stack([command[:, 1], command[:, 0]], dim=-1)
    inside_face_cos = torch.clamp((inside_world_xy * target_dir).sum(dim=-1), -1.0, 1.0)

    swing_xy = selected_pre_vel[:, :2]
    swing_speed = swing_xy.norm(dim=-1)
    swing_dir = swing_xy / swing_speed.unsqueeze(-1).clamp_min(_NORM_EPS)
    swing_cos = torch.clamp((swing_dir * target_dir).sum(dim=-1), -1.0, 1.0)

    touched = shared["pre_latch_touch_event"].bool()
    form["last_touch_valid"] = form["last_touch_valid"] | touched
    form["inside_contact_cos_last_touch"] = torch.where(
        touched, inside_contact_cos, form["inside_contact_cos_last_touch"]
    )
    form["contact_local_x_last_touch"] = torch.where(
        touched, ball_in_foot[:, 0], form["contact_local_x_last_touch"]
    )
    form["inside_face_cos_last_touch"] = torch.where(
        touched, inside_face_cos, form["inside_face_cos_last_touch"]
    )
    form["swing_cos_last_touch"] = torch.where(
        touched, swing_cos, form["swing_cos_last_touch"]
    )
    form["swing_speed_last_touch"] = torch.where(
        touched, swing_speed, form["swing_speed_last_touch"]
    )

    kick_event = shared["kick_done"] & (~form["prev_kick_done"])
    form["form_valid_frozen"] = torch.where(
        kick_event, form["last_touch_valid"], form["form_valid_frozen"]
    )
    form["inside_contact_cos_frozen"] = torch.where(
        kick_event, form["inside_contact_cos_last_touch"], form["inside_contact_cos_frozen"]
    )
    form["contact_local_x_frozen"] = torch.where(
        kick_event, form["contact_local_x_last_touch"], form["contact_local_x_frozen"]
    )
    form["inside_face_cos_frozen"] = torch.where(
        kick_event, form["inside_face_cos_last_touch"], form["inside_face_cos_frozen"]
    )
    form["swing_cos_frozen"] = torch.where(
        kick_event, form["swing_cos_last_touch"], form["swing_cos_frozen"]
    )
    form["swing_speed_frozen"] = torch.where(
        kick_event, form["swing_speed_last_touch"], form["swing_speed_frozen"]
    )

    form["prev_foot_vel"] = foot_vel.clone()
    form["prev_kick_done"] = shared["kick_done"].clone()
    form["step"] = step
    return shared, form


def kick_inside_contact(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_contact: float = math.radians(30.0),
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """目標方向に依存せず、ボールが足の内側面側で接触するほど高い報酬。"""
    shared, form = _inside_form_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    angle = torch.acos(torch.clamp(form["inside_contact_cos_frozen"], -1.0, 1.0))
    reward = torch.exp(-((angle / sigma_contact) ** 2))
    return reward * form["form_valid_frozen"].float() * shared["kick_done"].float()


def kick_ankle_contact(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    target_x: float = -0.004,
    sigma_x: float = 0.025,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """本命キックの接触位置を、踵から60 mm（足local X=-4 mm）へ誘導する。"""
    shared, form = _inside_form_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    error_x = (form["contact_local_x_frozen"] - target_x) / sigma_x
    reward = torch.exp(-(error_x**2))
    return reward * form["form_valid_frozen"].float() * shared["kick_done"].float()


def first_ball_touch(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    base_fraction: float = 0.25,
    speed_bonus_scale: float = 1.75,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """最初の接触を返し、接触後の球速が latch 閾値へ近いほど増幅する。"""
    state = _kick_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    touched = state["pre_latch_touch_event"].bool()
    first_touch = touched & (state["touch_count"] == 1.0)
    speed_ratio = torch.clamp(
        state["prev_v_ball"] / state["v_thresh_eff"].clamp_min(_NORM_EPS),
        min=0.0,
        max=1.0,
    )
    return first_touch.float() * (base_fraction + speed_bonus_scale * speed_ratio)


def _set_reward_weight(env: ManagerBasedRLEnv, term_name: str, weight: float) -> None:
    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - weight) > 1.0e-8:
        term.weight = weight


def inside_kick_stage_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    inside_contact_term_name: str,
    inside_face_term_name: str,
    straight_swing_term_name: str,
    opposite_direction_term_name: str,
    inside_contact_weight: float,
    stage2_inside_face_weight: float,
    stage2_straight_swing_weight: float,
    stage2_opposite_direction_weight: float,
    inside_contact_angle_deg: float,
    promote_kick_rate: float,
    promote_inside_contact_rate: float,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    speed_start_step: int,
    speed_end_step: int,
    steps_per_iteration: int = 0,
    ema_alpha: float = 0.01,
    speed_advance_above: float = 0.80,
    speed_retreat_below: float = 0.50,
    speed_advance_error_below_deg: float | None = None,
    speed_retreat_error_above_deg: float | None = None,
    speed_retreat_scale: float = 2.0,
) -> dict[str, float]:
    """内側接触を先に獲得し、その後に方向フォームと球速帯を有効化する。"""
    if steps_per_iteration > 0:
        now = env.common_step_counter / steps_per_iteration
    else:
        now = float(env.common_step_counter)

    state = getattr(env, _STAGE_STATE_ATTR, None)
    if state is None:
        state = {
            "stage": 1,
            "kick_rate_ema": 1.0,
            "inside_contact_rate_ema": 0.0,
            "first_touch_rate": 0.0,
            "extra_touch_count": 0.0,
            "touch_to_kick_rate": 0.0,
            "promoted_at": None,
        }
        setattr(env, _STAGE_STATE_ATTR, state)

    command_term = env.command_manager.get_term(command_name)
    kick_rate_metric = command_term.metrics.get("kick_rate", None)
    touch_count_metric = command_term.metrics.get("ball_touch_count", None)
    form = getattr(env, _FORM_STATE_ATTR, None)
    if kick_rate_metric is not None and env_ids is not None and len(env_ids) > 0:
        kick_done = kick_rate_metric[env_ids]
        kick_rate = float(kick_done.mean())
        state["kick_rate_ema"] += ema_alpha * (kick_rate - state["kick_rate_ema"])

        successful_kicks = float(kick_done.sum())
        if form is not None and successful_kicks > 0.0:
            valid = form["form_valid_frozen"][env_ids]
            contact_cos = form["inside_contact_cos_frozen"][env_ids]
            threshold_cos = math.cos(math.radians(inside_contact_angle_deg))
            inside_success = valid & (kick_done > 0.5) & (contact_cos >= threshold_cos)
            inside_contact_rate = float(inside_success.float().sum()) / successful_kicks
            state["inside_contact_rate_ema"] += ema_alpha * (
                inside_contact_rate - state["inside_contact_rate_ema"]
            )

        if touch_count_metric is not None:
            touch_count = touch_count_metric[env_ids]
            touched = touch_count > 0.0
            touched_count = float(touched.float().sum())
            state["first_touch_rate"] = float(touched.float().mean())
            state["extra_touch_count"] = float(
                torch.clamp(touch_count - 1.0, min=0.0).mean()
            )
            state["touch_to_kick_rate"] = (
                float(kick_done[touched].sum()) / touched_count if touched_count > 0.0 else 0.0
            )

    if (
        state["stage"] == 1
        and state["kick_rate_ema"] >= promote_kick_rate
        and state["inside_contact_rate_ema"] >= promote_inside_contact_rate
    ):
        state["stage"] = 2
        state["promoted_at"] = now

    _set_reward_weight(env, inside_contact_term_name, inside_contact_weight)
    if state["stage"] == 1:
        _set_reward_weight(env, inside_face_term_name, 0.0)
        _set_reward_weight(env, straight_swing_term_name, 0.0)
        _set_reward_weight(env, opposite_direction_term_name, 0.0)
        command_term.cfg.target_speed_range = start_range
        return {
            "stage": 1.0,
            "stage1_kick_rate_ema": state["kick_rate_ema"],
            "inside_contact_rate_ema": state["inside_contact_rate_ema"],
            "first_touch_rate": state["first_touch_rate"],
            "extra_touch_count": state["extra_touch_count"],
            "touch_to_kick_rate": state["touch_to_kick_rate"],
            "speed_min": start_range[0],
            "speed_max": start_range[1],
            "alpha": 0.0,
            "iterations_since_promotion": 0.0,
        }

    _set_reward_weight(env, inside_face_term_name, stage2_inside_face_weight)
    _set_reward_weight(env, straight_swing_term_name, stage2_straight_swing_weight)
    _set_reward_weight(env, opposite_direction_term_name, stage2_opposite_direction_weight)
    promoted_at = state["promoted_at"]
    speed = kick_rate_gated_speed_range(
        env,
        env_ids,
        command_name=command_name,
        start_range=start_range,
        end_range=end_range,
        start_step=promoted_at + speed_start_step,
        end_step=promoted_at + speed_end_step,
        steps_per_iteration=steps_per_iteration,
        advance_above=speed_advance_above,
        retreat_below=speed_retreat_below,
        advance_error_below_deg=speed_advance_error_below_deg,
        retreat_error_above_deg=speed_retreat_error_above_deg,
        ema_alpha=ema_alpha,
        retreat_scale=speed_retreat_scale,
    )
    return {
        "stage": 2.0,
        "stage1_kick_rate_ema": state["kick_rate_ema"],
        "inside_contact_rate_ema": state["inside_contact_rate_ema"],
        "first_touch_rate": state["first_touch_rate"],
        "extra_touch_count": state["extra_touch_count"],
        "touch_to_kick_rate": state["touch_to_kick_rate"],
        "iterations_since_promotion": now - promoted_at,
        **speed,
    }


def kick_inside_face_alignment(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_angle: float = 0.35,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """接触時のインサイド面法線が目標方向を向くほど高い報酬。"""
    shared, form = _inside_form_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    angle = torch.acos(torch.clamp(form["inside_face_cos_frozen"], -1.0, 1.0))
    reward = torch.exp(-((angle / sigma_angle) ** 2))
    return reward * form["form_valid_frozen"].float() * shared["kick_done"].float()


def kick_straight_swing(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_angle: float = 0.35,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """接触直前の蹴り足水平速度が目標方向を向くほど高い報酬。"""
    shared, form = _inside_form_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    angle = torch.acos(torch.clamp(form["swing_cos_frozen"], -1.0, 1.0))
    reward = torch.exp(-((angle / sigma_angle) ** 2))
    moving = form["swing_speed_frozen"] > _MOVING_EPS
    return (
        reward
        * moving.float()
        * form["form_valid_frozen"].float()
        * shared["kick_done"].float()
    )


def kick_velocity_independent(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_velocity: float = 1.0,
    use_3d_speed: bool = False,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """最終方向やフォームを掛けない、独立した球速追従報酬。"""
    state = _kick_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    speed = state["v_ball_3d_frozen"] if use_3d_speed else state["v_ball_frozen"]
    reward = torch.exp(-(((speed - state["v_target"]) / sigma_velocity) ** 2))
    return reward * state["kick_done"].float()


def kick_opposite_direction(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """目標の反対半球へ飛んだ量。非負値を返し、負の重みで使う。"""
    state = _kick_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    opposite = torch.clamp(-torch.cos(state["tau_direction_frozen"]), min=0.0, max=1.0)
    return opposite * state["kick_done"].float()


def kick_elevation_independent(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    phi_target: float = 0.52,
    sigma_phi: float = 0.25,
    phi_sat: float | None = None,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """最終方向を掛けず、既存の30度目標だけを維持する仰角報酬。"""
    state = _kick_state(
        env,
        r_stance,
        alpha,
        v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
    phi = state["phi_frozen"]
    if phi_sat is not None:
        reward = torch.clamp(phi / phi_sat, min=0.0, max=1.0)
    else:
        reward = torch.exp(-(((phi - phi_target) / sigma_phi) ** 2))
    return reward * state["kick_done"].float()
