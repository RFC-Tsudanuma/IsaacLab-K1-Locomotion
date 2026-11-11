# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""win_the_ball固有のカリキュラム。"""

from collections import deque
import math

import torch
from isaaclab.envs import ManagerBasedRLEnv


def kick_success_direction_expansion(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "kick_direction",
    metric_window_episodes: int = 1000,
    hold_iterations: float = 1000.0,
    steps_per_iteration: int = 48,
    qualification_rate: float = 0.9,
    max_direction_error_deg: float = 15.0,
    initial_stage: int = 0,
    front_halfwidth: float = math.pi / 4,
    side_halfwidth: float = 3.0 * math.pi / 4,
    rear_halfwidth: float = math.pi,
) -> dict[str, float]:
    """前方、側方、後方を成功率と方向精度で順番に解禁する。

    stage 0は前方 ``±front_halfwidth``、stage 1は側方を含む
    ``±side_halfwidth``、stage 2は後方を含む ``±rear_halfwidth``。各stageで直近
    ``metric_window_episodes`` 終了エピソードの成立率と方向誤差が条件を満たした状態を
    ``hold_iterations`` PPO iteration連続で維持した場合だけ次へ進む。

    4096並列環境では「1000終了エピソード」は1回のreset batchより少なく、維持期間に
    ならない。そのため維持時間は並列数に依存しないPPO iterationで測る。条件を外した
    reset batchが来たら連続維持は0から数え直す。
    """
    if metric_window_episodes <= 0 or hold_iterations <= 0.0 or steps_per_iteration <= 0:
        raise ValueError("評価窓、維持iteration、steps_per_iterationは正にしてください。")
    if not 0.0 <= qualification_rate <= 1.0:
        raise ValueError("qualification_rateは0から1の範囲にしてください。")
    if initial_stage not in (0, 1, 2):
        raise ValueError("initial_stageは0、1、2のいずれかにしてください。")
    if not 0.0 < front_halfwidth < side_halfwidth < rear_halfwidth <= math.pi:
        raise ValueError("方向範囲は 0 < front < side < rear <= pi としてください。")

    state = getattr(env, "_win_ball_direction_curriculum", None)
    if state is None:
        state = {
            "history": deque(maxlen=metric_window_episodes),
            "stage": initial_stage,
            "hold_start_iteration": None,
        }
        env._win_ball_direction_curriculum = state

    command = env.command_manager.get_term(command_name)
    kick_rate = command.metrics.get("kick_rate")
    direction_error = command.metrics.get("kick_dir_error_deg")
    valid_env_ids = env_ids
    if env_ids is not None and len(env_ids) > 0:
        # 環境生成直後の初期resetは終了エピソードではないので、統計にも維持量にも含めない。
        valid_env_ids = env_ids[env.episode_length_buf[env_ids] > 0]

    if (
        kick_rate is not None
        and direction_error is not None
        and valid_env_ids is not None
        and len(valid_env_ids) > 0
    ):
        rates = (kick_rate[valid_env_ids] > 0.5).detach().to("cpu").tolist()
        errors = direction_error[valid_env_ids].abs().detach().to("cpu").tolist()
        state["history"].extend(zip(rates, errors))

    history = state["history"]
    rolling_rate = sum(float(kicked) for kicked, _ in history) / len(history) if history else 0.0
    successful_errors = [error for kicked, error in history if kicked]
    rolling_direction_error = (
        sum(successful_errors) / len(successful_errors) if successful_errors else float("inf")
    )
    direction_ok = rolling_direction_error <= max_direction_error_deg
    conditions_met = (
        len(history) == metric_window_episodes
        and rolling_rate >= qualification_rate
        and direction_ok
    )

    now_iteration = env.common_step_counter / steps_per_iteration
    if conditions_met:
        if state["hold_start_iteration"] is None:
            state["hold_start_iteration"] = now_iteration
    else:
        state["hold_start_iteration"] = None

    held_iterations = (
        now_iteration - state["hold_start_iteration"]
        if state["hold_start_iteration"] is not None
        else 0.0
    )

    if held_iterations >= hold_iterations and state["stage"] < 2:
        state["stage"] += 1
        history.clear()
        state["hold_start_iteration"] = None
        held_iterations = 0.0

    halfwidth = (front_halfwidth, side_halfwidth, rear_halfwidth)[state["stage"]]
    command.cfg.ranges.heading = (-halfwidth, halfwidth)

    return {
        "stage": float(state["stage"]),
        "metric_window_episodes": float(len(history)),
        "held_iterations": held_iterations,
        "conditions_met": float(conditions_met),
        "rolling_kick_rate": rolling_rate,
        "rolling_direction_error_deg": rolling_direction_error,
        "heading_halfwidth_deg": math.degrees(halfwidth),
    }