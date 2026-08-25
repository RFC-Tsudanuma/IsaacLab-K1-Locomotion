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
from ..walk_kick.mdp.rewards import kick_direction as base_kick_direction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_FORM_STATE_ATTR = "_inside_kick_form_state"
_STAGE_STATE_ATTR = "_inside_kick_stage_state"
_BODY_FACING_STATE_ATTR = "_inside_kick_body_facing_state"
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


def _front_factor(state: dict, *, frozen: bool) -> torch.Tensor:
    """接触直前の胴体正面にボールがある度合い。横から後方は0、正面は1。"""
    if frozen:
        return torch.clamp(state["body_ball_cos_frozen"], min=0.0, max=1.0)
    return torch.clamp(state["body_ball_cos_pre_step"], min=0.0, max=1.0)


def body_ball_facing(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_angle: float = math.radians(60.0),
    sigma_pose: float = 0.3,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """最終接近の正対度がエピソード内の自己ベストを更新した分だけ払う。"""
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
    angle = torch.acos(torch.clamp(state["body_ball_cos"], -1.0, 1.0))
    alignment = torch.exp(-((angle / sigma_angle) ** 2))
    final_approach = torch.exp(-((state["d_to_P_kick"] / sigma_pose) ** 2))
    potential = alignment * final_approach

    # 状態値を毎step払い続けると、P_kickで正対したままtime-outまで報酬を稼げる。
    # エピソード内の最大値を更新した差分だけにすれば、正のshapingを保ちながら
    # 立ち止まりや姿勢の往復では追加報酬が出ない。差分の総和は最大1で有界。
    step = int(env.common_step_counter)
    facing_state = getattr(env, _BODY_FACING_STATE_ATTR, None)
    if facing_state is not None and facing_state["step"] == step:
        return facing_state["reward"]
    if facing_state is None:
        facing_state = {
            "step": -1,
            "best": potential.clone(),
            "reward": torch.zeros_like(potential),
        }
        setattr(env, _BODY_FACING_STATE_ATTR, facing_state)

    just_reset = env.episode_length_buf == 1
    previous_best = torch.where(just_reset, potential, facing_state["best"])
    improvement = torch.clamp(potential - previous_best, min=0.0)
    reward = improvement * (~just_reset).float() * (~state["kick_done"]).float()

    facing_state["best"] = torch.maximum(previous_best, potential)
    facing_state["reward"] = reward
    facing_state["step"] = step
    return reward


def kick_direction_front_gated(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    v_gate_frac: float = 0.0,
    sigma_gate: float = 0.05,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """既存の方向精度報酬を、キック時の胴体―ボール前方係数でゲートする。"""
    reward = base_kick_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction=sigma_direction,
        v_gate_frac=v_gate_frac,
        sigma_gate=sigma_gate,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )
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
    return reward * _front_factor(state, frozen=True)


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
    return (
        reward
        * form["form_valid_frozen"].float()
        * shared["kick_done"].float()
        * _front_factor(shared, frozen=True)
    )


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
    return (
        reward
        * form["form_valid_frozen"].float()
        * shared["kick_done"].float()
        * _front_factor(shared, frozen=True)
    )


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
    return (
        first_touch.float()
        * (base_fraction + speed_bonus_scale * speed_ratio)
        * _front_factor(state, frozen=False)
    )


def _set_reward_weight(env: ManagerBasedRLEnv, term_name: str, weight: float) -> None:
    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - weight) > 1.0e-8:
        term.weight = weight


def _set_reward_param(
    env: ManagerBasedRLEnv, term_name: str, param_name: str, value: float
) -> None:
    term = env.reward_manager.get_term_cfg(term_name)
    if abs(float(term.params[param_name]) - value) > 1.0e-8:
        term.params[param_name] = value


def inside_kick_stage_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    inside_contact_term_name: str,
    inside_face_term_name: str,
    straight_swing_term_name: str,
    opposite_direction_term_name: str,
    direction_accuracy_term_name: str,
    inside_contact_weight: float,
    stage2_inside_face_weight: float,
    stage2_straight_swing_weight: float,
    stage2_opposite_direction_weight: float,
    stage3_direction_accuracy_weight: float,
    stage3_promote_error_below_deg: float,
    stage3_direction_ramp_iterations: int,
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
    inside_face_rough_multiplier: float = 2.0,
    inside_face_precision_multiplier: float = 1.5,
    inside_face_maintain_multiplier: float = 1.0,
    inside_face_rough_sigma_angle: float = math.radians(45.0),
    inside_face_precision_sigma_angle: float = math.radians(30.0),
    inside_face_maintain_sigma_angle: float = math.radians(20.0),
    inside_face_precision_enter_below_deg: float = 30.0,
    inside_face_maintain_enter_below_deg: float = 10.0,
) -> dict[str, float]:
    """内側接触を先に獲得し、その後に方向フォームと球速帯を有効化する。

    Stage 2 の足内側面報酬は、成功キックの方向誤差 EMA に合わせて粗調整・精密化・
    維持の3段階で weight と Gaussian 幅を切り替える。一度進んだ段階は精度が悪化しても
    戻さず、粗調整から精密化、精密化から維持へ一方向にだけ進める。方向誤差 EMA が
    閾値を下回ると不可逆に Stage 3 へ進み、最終球方向の正報酬を時間基準で立ち上げる。
    """
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
            "body_ball_angle_at_kick_deg": 0.0,
            "body_ball_angle_sample_count": 0.0,
            "promoted_at": None,
            "direction_promoted_at": None,
            "inside_face_phase": "rough",
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
        state["body_ball_angle_sample_count"] = successful_kicks
        if form is not None and successful_kicks > 0.0:
            valid = form["form_valid_frozen"][env_ids]
            contact_cos = form["inside_contact_cos_frozen"][env_ids]
            threshold_cos = math.cos(math.radians(inside_contact_angle_deg))
            inside_success = valid & (kick_done > 0.5) & (contact_cos >= threshold_cos)
            inside_contact_rate = float(inside_success.float().sum()) / successful_kicks
            state["inside_contact_rate_ema"] += ema_alpha * (
                inside_contact_rate - state["inside_contact_rate_ema"]
            )

        kick_latch = getattr(env, "_kick_latch_state", None)
        if kick_latch is not None and successful_kicks > 0.0:
            kicked = kick_done > 0.5
            body_ball_angle = torch.rad2deg(
                torch.acos(
                    torch.clamp(kick_latch["body_ball_cos_frozen"][env_ids], -1.0, 1.0)
                )
            )
            state["body_ball_angle_at_kick_deg"] = float(body_ball_angle[kicked].mean())

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
        _set_reward_weight(env, direction_accuracy_term_name, 0.0)
        command_term.cfg.target_speed_range = start_range
        return {
            "stage": 1.0,
            "stage1_kick_rate_ema": state["kick_rate_ema"],
            "inside_contact_rate_ema": state["inside_contact_rate_ema"],
            "first_touch_rate": state["first_touch_rate"],
            "extra_touch_count": state["extra_touch_count"],
            "touch_to_kick_rate": state["touch_to_kick_rate"],
            "body_ball_angle_at_kick_deg": state["body_ball_angle_at_kick_deg"],
            "body_ball_angle_sample_count": state["body_ball_angle_sample_count"],
            "speed_min": start_range[0],
            "speed_max": start_range[1],
            "alpha": 0.0,
            "iterations_since_promotion": 0.0,
            "inside_face_phase": -1.0,
            "inside_face_multiplier": 0.0,
            "inside_face_sigma_deg": math.degrees(inside_face_maintain_sigma_angle),
            "inside_face_weight": 0.0,
            "direction_accuracy_alpha": 0.0,
            "direction_accuracy_weight": 0.0,
            "iterations_since_direction_promotion": 0.0,
        }

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
    direction_error_ema = speed["kick_dir_error_ema_deg"]
    if state["stage"] == 2 and direction_error_ema < stage3_promote_error_below_deg:
        state["stage"] = 3
        state["direction_promoted_at"] = now

    direction_promoted_at = state["direction_promoted_at"]
    if state["stage"] == 3:
        iterations_since_direction_promotion = max(0.0, now - direction_promoted_at)
        direction_accuracy_alpha = min(
            iterations_since_direction_promotion / stage3_direction_ramp_iterations,
            1.0,
        )
        direction_accuracy_weight = stage3_direction_accuracy_weight * direction_accuracy_alpha
    else:
        iterations_since_direction_promotion = 0.0
        direction_accuracy_alpha = 0.0
        direction_accuracy_weight = 0.0
    _set_reward_weight(env, direction_accuracy_term_name, direction_accuracy_weight)

    phase = state["inside_face_phase"]
    if phase == "rough":
        if direction_error_ema <= inside_face_precision_enter_below_deg:
            phase = "precision"
    elif phase == "precision":
        if direction_error_ema <= inside_face_maintain_enter_below_deg:
            phase = "maintain"
    elif phase != "maintain":
        raise ValueError(f"未知の inside_face_phase: {phase}")
    state["inside_face_phase"] = phase

    if phase == "rough":
        inside_face_multiplier = inside_face_rough_multiplier
        inside_face_sigma_angle = inside_face_rough_sigma_angle
        inside_face_phase_metric = 2.0
    elif phase == "precision":
        inside_face_multiplier = inside_face_precision_multiplier
        inside_face_sigma_angle = inside_face_precision_sigma_angle
        inside_face_phase_metric = 1.0
    else:
        inside_face_multiplier = inside_face_maintain_multiplier
        inside_face_sigma_angle = inside_face_maintain_sigma_angle
        inside_face_phase_metric = 0.0

    inside_face_weight = stage2_inside_face_weight * inside_face_multiplier
    _set_reward_weight(env, inside_face_term_name, inside_face_weight)
    _set_reward_param(
        env,
        inside_face_term_name,
        "sigma_angle",
        inside_face_sigma_angle,
    )
    return {
        "stage": float(state["stage"]),
        "stage1_kick_rate_ema": state["kick_rate_ema"],
        "inside_contact_rate_ema": state["inside_contact_rate_ema"],
        "first_touch_rate": state["first_touch_rate"],
        "extra_touch_count": state["extra_touch_count"],
        "touch_to_kick_rate": state["touch_to_kick_rate"],
        "body_ball_angle_at_kick_deg": state["body_ball_angle_at_kick_deg"],
        "body_ball_angle_sample_count": state["body_ball_angle_sample_count"],
        "iterations_since_promotion": now - promoted_at,
        "inside_face_phase": inside_face_phase_metric,
        "inside_face_multiplier": inside_face_multiplier,
        "inside_face_sigma_deg": math.degrees(inside_face_sigma_angle),
        "inside_face_weight": inside_face_weight,
        "direction_accuracy_alpha": direction_accuracy_alpha,
        "direction_accuracy_weight": direction_accuracy_weight,
        "iterations_since_direction_promotion": iterations_since_direction_promotion,
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
    return (
        reward
        * form["form_valid_frozen"].float()
        * shared["kick_done"].float()
        * _front_factor(shared, frozen=True)
    )


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
        * _front_factor(shared, frozen=True)
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
    return reward * state["kick_done"].float() * _front_factor(state, frozen=True)


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
    return reward * state["kick_done"].float() * _front_factor(state, frozen=True)
