# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用のカリキュラム関数。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_kick_angle_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    start_deg: float,
    end_deg: float,
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """キック方向コマンドの角度範囲を ``start_step`` → ``end_step`` で ±start_deg → ±end_deg に線形に広げる。

    学習初期はキック方向をほぼ正面 (±start_deg) に限定して「回り込みがほぼ不要」な
    易しい状況から始め、徐々に範囲を広げて真後ろ (±180°) までの深い回り込みを要求する。
    真後ろを最初から混ぜると立ち上がりが遅いので、難易度カリキュラムで学習を安定させる。

    ``KickDirectionCommand`` は reset 時の再サンプル (``_resample_command``) で
    ``cfg.angle_range`` を読むため、cfg を書き換えるだけで次エピソードから反映される。

    Note:
        ``start_step`` / ``end_step`` は ``env.common_step_counter`` と比較する
        (locomotion の ``modify_reward_weight_linear`` と同じ規約)。
    """
    s = env.common_step_counter
    if s <= start_step:
        deg = start_deg
    elif s >= end_step:
        deg = end_deg
    else:
        alpha = (s - start_step) / float(end_step - start_step)
        deg = start_deg + alpha * (end_deg - start_deg)
    rad = math.radians(deg)
    term = env.command_manager.get_term(command_name)
    term.cfg.angle_range = (-rad, rad)
    return {"kick_angle_deg": float(deg)}
