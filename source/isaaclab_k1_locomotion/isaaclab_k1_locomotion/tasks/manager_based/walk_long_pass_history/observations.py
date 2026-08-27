# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``walk_long_pass_history`` 専用のボール・蹴り方向観測。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from ..walk_kick.mdp.kick_state import _ATTR as _KICK_LATCH_STATE_ATTR
from ..walk_kick.mdp.observations import (
    _OBS_DELAY_STATE_ATTR,
    _delayed_signal,
    delayed_ball_pos_b,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ``walk_long_pass_orbit`` の自己位置推定 DR と同じ値。
LOCALIZATION_DELAY_BASE_S = 0.20
LOCALIZATION_DELAY_JITTER_S = 0.10
LOCALIZATION_POSITION_ERROR_MAX_M = 0.20
LOCALIZATION_YAW_ERROR_MAX_RAD = 0.105  # 約 6 deg
MAP_TARGET_DISTANCE_RANGE_M = (5.0, 12.0)

# history task が従来の sole_pos / prev_ball_pos に載せていた一様ノイズ。
BALL_POSITION_NOISE_MAX_M = 0.02

_BALL_MEASUREMENT_STATE_ATTR = "_history_ball_measurement_state"
_LOCALIZATION_TARGET_STATE_ATTR = "_history_localization_target_state"
_RESET_PENDING_STATE_ATTR = "_history_observation_reset_pending"


def _reset_pending_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    state = getattr(env, _RESET_PENDING_STATE_ATTR, None)
    if state is None or state["ball"].shape[0] != env.num_envs:
        state = {
            "ball": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "localization": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        setattr(env, _RESET_PENDING_STATE_ATTR, state)
    return state


def _mark_reset_pending(
    env: ManagerBasedRLEnv,
    key: str,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> None:
    pending = _reset_pending_state(env)[key]
    if env_ids is None:
        pending[:] = True
    else:
        pending[env_ids] = True


def _consume_reset_pending(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    pending = _reset_pending_state(env)[key]
    result = pending.clone()
    pending[result] = False
    return result


class SharedDelayedBallPosition(ManagerTermBase):
    """ObservationManagerのreset通知を共有ボール計測へ伝えるterm。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        _mark_reset_pending(self._env, "ball", env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        noise_max: float = BALL_POSITION_NOISE_MAX_M,
        dim: int = 3,
        delay_steps: int = 1,
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        return shared_delayed_ball_pos_b(
            env,
            noise_max=noise_max,
            dim=dim,
            delay_steps=delay_steps,
            ball_cfg=ball_cfg,
        )


class FrozenMapTargetKickDirection(ManagerTermBase):
    """ObservationManagerのreset通知を固定目標・latch状態へ伝えるterm。"""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        _mark_reset_pending(self._env, "ball", env_ids)
        _mark_reset_pending(self._env, "localization", env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "kick_direction",
        delay_s: float = LOCALIZATION_DELAY_BASE_S,
        delay_jitter_s: float = LOCALIZATION_DELAY_JITTER_S,
        pos_err_max: float = LOCALIZATION_POSITION_ERROR_MAX_M,
        yaw_err_max: float = LOCALIZATION_YAW_ERROR_MAX_RAD,
        dist_range: tuple[float, float] = MAP_TARGET_DISTANCE_RANGE_M,
        ball_noise_max: float = BALL_POSITION_NOISE_MAX_M,
        group: str = "localization",
        ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
    ) -> torch.Tensor:
        return frozen_map_target_kick_dir_b(
            env,
            command_name=command_name,
            delay_s=delay_s,
            delay_jitter_s=delay_jitter_s,
            pos_err_max=pos_err_max,
            yaw_err_max=yaw_err_max,
            dist_range=dist_range,
            ball_noise_max=ball_noise_max,
            group=group,
            ball_cfg=ball_cfg,
        )


def _policy_corruption_enabled(env: ManagerBasedRLEnv) -> bool:
    """PLAY の ``enable_corruption=False`` を関数内ノイズにも反映する。"""
    cfg = getattr(env, "cfg", None)
    observations = getattr(cfg, "observations", None)
    policy = getattr(observations, "policy", None)
    return bool(getattr(policy, "enable_corruption", True))


def shared_delayed_ball_pos_b(
    env: ManagerBasedRLEnv,
    noise_max: float = BALL_POSITION_NOISE_MAX_M,
    dim: int = 3,
    delay_steps: int = 1,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """同一ステップ内で共有する遅延・ノイズ付きローカルボール位置。

    ``sole_pos``、``prev_ball_pos``、global→local の蹴り方向計算が、同じカメラ計測を
    見るための関数。ノイズを ObservationManager の外で一度だけ生成してキャッシュする。
    周辺分布は従来どおり一様 ``[-noise_max, noise_max]``、遅延も従来どおり1ステップ。
    """
    if dim not in (2, 3):
        raise ValueError(f"shared_delayed_ball_pos_b: dim は 2 または 3 (指定: {dim})")

    raw = delayed_ball_pos_b(env, ball_cfg=ball_cfg, delay_steps=delay_steps, dim=3)
    step = int(env.common_step_counter)
    state = getattr(env, _BALL_MEASUREMENT_STATE_ATTR, None)
    pending_reset = _consume_reset_pending(env, "ball")

    if state is None or state["value"].shape != raw.shape:
        state = {
            "value": raw.clone(),
            "step": -1,
            "noise_max": None,
        }
        setattr(env, _BALL_MEASUREMENT_STATE_ATTR, state)

    effective_noise = noise_max if _policy_corruption_enabled(env) else 0.0
    if state["step"] != step:
        if effective_noise > 0.0:
            noise = (torch.rand_like(raw) * 2.0 - 1.0) * effective_noise
            state["value"] = raw + noise
        else:
            state["value"] = raw.clone()
        state["step"] = step
        state["noise_max"] = effective_noise
        pending_reset[:] = False
    else:
        if state["noise_max"] != effective_noise:
            raise ValueError(
                "shared_delayed_ball_pos_b: 同一ステップの呼び出しで noise_max が一致しません: "
                f"{state['noise_max']} != {effective_noise}"
            )

    # ObservationManager.reset() は、初回resetやRecorderが有効な同一step内resetも通知する。
    # reset対象envだけ新しいボール位置・ノイズへ入れ替え、前エピソードを残さない。
    refresh = pending_reset
    if bool(refresh.any()):
        if effective_noise > 0.0:
            noise = (torch.rand_like(raw[refresh]) * 2.0 - 1.0) * effective_noise
            state["value"][refresh] = raw[refresh] + noise
        else:
            state["value"][refresh] = raw[refresh]

    return state["value"][:, :dim]


def frozen_map_target_kick_dir_b(
    env: ManagerBasedRLEnv,
    command_name: str = "kick_direction",
    delay_s: float = LOCALIZATION_DELAY_BASE_S,
    delay_jitter_s: float = LOCALIZATION_DELAY_JITTER_S,
    pos_err_max: float = LOCALIZATION_POSITION_ERROR_MAX_M,
    yaw_err_max: float = LOCALIZATION_YAW_ERROR_MAX_RAD,
    dist_range: tuple[float, float] = MAP_TARGET_DISTANCE_RANGE_M,
    ball_noise_max: float = BALL_POSITION_NOISE_MAX_M,
    group: str = "localization",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
) -> torch.Tensor:
    """固定global目標を、遅延・DR済み自己位置から見たactor用ローカル方向。

    目標点はリセット時の真のボール位置から、報酬が使う真のコマンド方向へ5-12 m先に
    一度だけ置く。自己位置 ``x/y/yaw`` は同じ200-300 msの遅延とエピソード固定biasを
    共有する。ボール位置は :func:`shared_delayed_ball_pos_b` と同じローカル計測を使う。

    ``kick_state`` の ``kick_done`` が初めて立ったステップで出力を凍結し、エピソードが
    終わるまで保持する。critic・報酬の観測やコマンドにはこの誤差を流さない。
    """
    robot = env.scene["robot"]
    ball = env.scene[ball_cfg.name]
    device = robot.data.root_pos_w.device
    num_envs = env.num_envs

    command = env.command_manager.get_command(command_name)  # [sin theta, cos theta, v]
    true_dir_w = torch.stack([command[:, 1], command[:, 0]], dim=-1)

    pending_reset = _consume_reset_pending(env, "localization")
    state = getattr(env, _LOCALIZATION_TARGET_STATE_ATTR, None)
    if state is None or state["target_w"].shape[0] != num_envs:
        state = {
            "bias_pos": torch.zeros(num_envs, 2, device=device),
            "bias_yaw": torch.zeros(num_envs, device=device),
            "target_w": torch.zeros(num_envs, 2, device=device),
            "frozen_dir_b": torch.zeros(num_envs, 2, device=device),
            "direction_frozen": torch.zeros(num_envs, dtype=torch.bool, device=device),
        }
        setattr(env, _LOCALIZATION_TARGET_STATE_ATTR, state)
        fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
    else:
        fresh = pending_reset

    if bool(fresh.any()):
        count = int(fresh.sum().item())

        bias_angle = (torch.rand(count, device=device) * 2.0 - 1.0) * math.pi
        bias_magnitude = torch.rand(count, device=device) * pos_err_max
        state["bias_pos"][fresh] = torch.stack(
            [bias_magnitude * torch.cos(bias_angle), bias_magnitude * torch.sin(bias_angle)],
            dim=-1,
        )
        state["bias_yaw"][fresh] = (
            torch.rand(count, device=device) * 2.0 - 1.0
        ) * yaw_err_max

        distance_min, distance_max = dist_range
        distance = torch.rand(count, device=device) * (distance_max - distance_min) + distance_min
        state["target_w"][fresh] = (
            ball.data.root_pos_w[fresh, :2] + distance.unsqueeze(-1) * true_dir_w[fresh]
        )
        state["frozen_dir_b"][fresh] = 0.0
        state["direction_frozen"][fresh] = False

    # Recorder有効時はreset前の観測で同じstepの遅延groupが先に更新される。reset後に
    # _delayed_signal のreset分岐へ入り直し、新エピソードの遅延量を引き直させる。
    if bool(fresh.any()):
        delay_root = getattr(env, _OBS_DELAY_STATE_ATTR, None)
        delay_gate = None if delay_root is None else delay_root["groups"].get(group)
        if delay_gate is not None and delay_gate["step"] == int(env.common_step_counter):
            delay_gate["step"] = -1

    robot_pos_w = robot.data.root_pos_w[:, :2]
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    # x/y/yaw は同じ自己位置推定器の出力なので、同じ遅延量を引く。
    # yaw は +-pi の境界を跨いでも補間できるよう cos/sin で履歴化する。
    pose = torch.stack(
        [robot_pos_w[:, 0], robot_pos_w[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    delayed_pose = _delayed_signal(env, "history_loc_pose", group, pose, delay_jitter_s, delay_s)
    delayed_pos_w = delayed_pose[:, :2]
    delayed_yaw = torch.atan2(delayed_pose[:, 3], delayed_pose[:, 2])

    estimated_pos_w = delayed_pos_w + state["bias_pos"]
    estimated_yaw = delayed_yaw + state["bias_yaw"]

    # actorへ渡すボール位置スロットと同じ、1ステップ遅延・同一ノイズ標本を使う。
    measured_ball_b = shared_delayed_ball_pos_b(
        env,
        noise_max=ball_noise_max,
        dim=2,
        delay_steps=1,
        ball_cfg=ball_cfg,
    )
    cos_yaw, sin_yaw = torch.cos(estimated_yaw), torch.sin(estimated_yaw)
    measured_ball_w = torch.stack(
        [
            cos_yaw * measured_ball_b[:, 0] - sin_yaw * measured_ball_b[:, 1],
            sin_yaw * measured_ball_b[:, 0] + cos_yaw * measured_ball_b[:, 1],
        ],
        dim=-1,
    )
    estimated_ball_w = estimated_pos_w + measured_ball_w

    target_delta_w = state["target_w"] - estimated_ball_w
    candidate_dir_b = torch.stack(
        [
            cos_yaw * target_delta_w[:, 0] + sin_yaw * target_delta_w[:, 1],
            -sin_yaw * target_delta_w[:, 0] + cos_yaw * target_delta_w[:, 1],
        ],
        dim=-1,
    )
    candidate_dir_b = candidate_dir_b / candidate_dir_b.norm(dim=-1, keepdim=True).clamp(
        min=1e-6
    )

    # TerminationManager がObservationManagerより先に kick_state を更新するため、
    # latchした同じステップの候補方向をここで保存できる。リセット直後は前エピソードの
    # kick_done が一時的に残り得るので、episode_length_buf==0 のenvは必ず除外する。
    kick_latch_state = getattr(env, _KICK_LATCH_STATE_ATTR, None)
    if kick_latch_state is not None:
        kick_done = kick_latch_state["kick_done"] & (env.episode_length_buf != 0)
        newly_frozen = kick_done & (~state["direction_frozen"])
        if bool(newly_frozen.any()):
            state["frozen_dir_b"][newly_frozen] = candidate_dir_b[newly_frozen]
            state["direction_frozen"][newly_frozen] = True

    return torch.where(
        state["direction_frozen"].unsqueeze(-1),
        state["frozen_dir_b"],
        candidate_dir_b,
    )
