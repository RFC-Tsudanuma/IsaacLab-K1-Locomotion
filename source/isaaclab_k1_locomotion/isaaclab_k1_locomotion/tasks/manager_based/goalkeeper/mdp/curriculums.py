# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスクのカリキュラム。

ステージ3: セーブ成功率の指数移動平均 (EMA) に応じてボール初速の上限を
連続的に引き上げる適応カリキュラム。既存タスクの common_step_counter 線形
スケジュールと違い「学習の進み具合」そのものに追従する。

しきい値・増分・上限は EventTerm の params ではなく GoalkeeperParamsCfg
(env cfg の ``goalkeeper`` フィールド) から読むので、``--override_json`` で
設定ファイルから制御できる。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .events import _gk_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def adaptive_ball_speed(
    env: "ManagerBasedRLEnv",
    env_ids,
) -> float:
    """セーブ成功率 EMA に応じてボール初速上限 ``_gk_speed_hi`` を調整する。

    CurriculumManager からリセット対象の env_ids で呼ばれる。エピソードの成否は
    termination manager の発火状況から判定する:

        * 失点 (goal_conceded) が発火          → 失敗 (0)
        * time_out / save_success (成功系)     → 成功 (1)
        * それ以外 (転倒・場外などの失敗終了)  → 失敗 (0)

    調整則 (パラメータは GoalkeeperParamsCfg):
        * EMA > adaptive_success_threshold → 上限 += adaptive_speed_delta
          (ball_speed_cap でクランプ)
        * EMA < adaptive_fail_threshold   → 上限 -= adaptive_speed_delta
          (ball_speed_max = 初期上限でクランプ)
        * 調整のたびに EMA を中立値へ戻し、次の調整には新しい証拠を要求する
          (連続リセットバッチでの階段的暴走を防ぐ)

    戻り値は現在の初速上限 [m/s] (Curriculum/ ログに出る)。
    """
    p = _gk_params(env)

    hi = getattr(env, "_gk_speed_hi", None)
    if hi is None:
        env._gk_speed_hi = torch.tensor(float(p.ball_speed_max), device=env.device)
        env._gk_success_ema = torch.tensor(0.5, device=env.device)
        env._gk_episode_count = 0
        hi = env._gk_speed_hi

    if env_ids is None or len(env_ids) == 0:
        return float(env._gk_speed_hi.item())

    tm = env.termination_manager
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ("time_out", "save_success"):
        if name in tm.active_terms:
            success |= tm.get_term(name)
    if "goal_conceded" in tm.active_terms:
        success &= ~tm.get_term("goal_conceded")

    batch = success[env_ids].float()
    n = batch.numel()
    env._gk_episode_count += n

    alpha = min(1.0, float(p.adaptive_ema_alpha) * n)
    env._gk_success_ema = (1.0 - alpha) * env._gk_success_ema + alpha * batch.mean()

    # ウォームアップ中は調整しない (EMA が立ち上がるまで待つ)
    if env._gk_episode_count >= int(p.adaptive_warmup_episodes):
        neutral = 0.5 * (float(p.adaptive_success_threshold) + float(p.adaptive_fail_threshold))
        if env._gk_success_ema.item() > float(p.adaptive_success_threshold):
            env._gk_speed_hi = (env._gk_speed_hi + float(p.adaptive_speed_delta)).clamp(
                max=float(p.ball_speed_cap)
            )
            env._gk_success_ema.fill_(neutral)
        elif env._gk_success_ema.item() < float(p.adaptive_fail_threshold):
            env._gk_speed_hi = (env._gk_speed_hi - float(p.adaptive_speed_delta)).clamp(
                min=float(p.ball_speed_max)
            )
            env._gk_success_ema.fill_(neutral)

    return float(env._gk_speed_hi.item())
