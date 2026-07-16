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
) -> torch.Tensor:
    """キック成立 (kick_done) から delay_steps 後にエピソードを終了する。shape: (N,) bool

    1 エピソード = 1 キック。飛翔後しばらくは項1-3 を凍結値で dense に払い続けたいので、
    latch 直後ではなく delay_steps だけ待ってから終了させる。

    NOTE: この項は weight とは無関係に毎ステップ評価されるため、:func:`kick_state` の
          更新を保証する役割も担っている。RewardManager は weight==0 の項をスキップするので、
          カリキュラムで 0 から立ち上げる報酬項に状態更新を任せることはできない。
          TerminationManager は RewardManager より先に走るので、報酬項が読む時点で最新になる。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    if not hasattr(env, "_kick_done_counter"):
        env._kick_done_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    just_reset = env.episode_length_buf == 1
    env._kick_done_counter[just_reset] = 0

    counting = state["kick_done"]
    env._kick_done_counter[counting] += 1

    return env._kick_done_counter >= delay_steps
