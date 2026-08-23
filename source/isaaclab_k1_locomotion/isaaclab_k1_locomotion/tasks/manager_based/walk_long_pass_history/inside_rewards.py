# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_history 専用のインサイドキック報酬。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

from ..walk_kick.mdp.kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_FORM_STATE_ATTR = "_inside_kick_form_state"
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
            "inside_face_cos_last_touch": torch.zeros(env.num_envs, device=device),
            "swing_cos_last_touch": torch.zeros(env.num_envs, device=device),
            "swing_speed_last_touch": torch.zeros(env.num_envs, device=device),
            "form_valid_frozen": torch.zeros(env.num_envs, dtype=torch.bool, device=device),
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
        form["inside_face_cos_last_touch"][just_reset] = 0.0
        form["swing_cos_last_touch"][just_reset] = 0.0
        form["swing_speed_last_touch"][just_reset] = 0.0
        form["form_valid_frozen"][just_reset] = False
        form["inside_face_cos_frozen"][just_reset] = 0.0
        form["swing_cos_frozen"][just_reset] = 0.0
        form["swing_speed_frozen"][just_reset] = 0.0

    env_ids = torch.arange(env.num_envs, device=device)
    ball_pos = ball.data.root_pos_w[:, :3]
    nearest_foot = (foot_pos - ball_pos.unsqueeze(1)).norm(dim=-1).argmin(dim=1)
    selected_quat = foot_quat[env_ids, nearest_foot]
    selected_pre_vel = form["prev_foot_vel"][env_ids, nearest_foot]

    # URDF の中立姿勢では左足の内側が local -Y、右足の内側が local +Y。
    inside_local = torch.zeros(env.num_envs, 3, device=device, dtype=foot_pos.dtype)
    inside_local[:, 1] = torch.where(nearest_foot == 0, -1.0, 1.0)
    inside_world_xy = quat_apply(selected_quat, inside_local)[:, :2]
    inside_world_xy = inside_world_xy / inside_world_xy.norm(dim=-1, keepdim=True).clamp_min(
        _NORM_EPS
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
