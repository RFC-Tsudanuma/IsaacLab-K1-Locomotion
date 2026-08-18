# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def kick_finished(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    delay_steps: int = 30,
    track_ball: bool = False,
    v_thresh_target_frac: float = 0.0,
    v_thresh_floor: float = 0.0,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """キック成立 (kick_done) から delay_steps 後にエピソードを終了する。shape: (N,) bool

    1 エピソード = 1 キック。飛翔後しばらくは項1-3 を凍結値で dense に払い続けたいので、
    latch 直後ではなく delay_steps だけ待ってから終了させる。

    NOTE: この項は weight とは無関係に毎ステップ評価されるため、:func:`kick_state` の
          更新を保証する役割も担っている。RewardManager は weight==0 の項をスキップするので、
          カリキュラムで 0 から立ち上げる報酬項に状態更新を任せることはできない。
          TerminationManager は RewardManager より先に走るので、報酬項が読む時点で最新になる。

    ``track_ball`` (転がるボール用に P_kick を latch まで追従させる) と
    ``v_thresh_target_*`` (latch 閾値を指令速度に追従させる) は :func:`kick_state` に
    そのまま渡す。**この項が毎ステップ最初に kick_state を呼ぶ**ので、ここに渡せば
    その step の状態全体に効く。

    ``r_max`` / ``orbit_beta`` (回り込み型の G) と ``overshoot_margin``、
    ``lateral_band`` (終端の構えの横のあそび) も同じく :func:`kick_state` にそのまま
    渡すが、こちらは **報酬項側にも同じ値を配ること**。既定
    (``r_max=None`` / ``overshoot_margin=0.0`` / ``lateral_band=None``) では
    従来の挙動と完全に一致する。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        track_ball=track_ball,
        v_thresh_target_frac=v_thresh_target_frac,
        v_thresh_floor=v_thresh_floor,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
    )

    if not hasattr(env, "_kick_done_counter"):
        env._kick_done_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    just_reset = env.episode_length_buf == 1
    env._kick_done_counter[just_reset] = 0

    counting = state["kick_done"]
    env._kick_done_counter[counting] += 1

    return env._kick_done_counter >= delay_steps
