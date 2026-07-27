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
    conceded = (
        tm.get_term("goal_conceded")
        if "goal_conceded" in tm.active_terms
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )

    # --- 成功率の測り方はエピソードの構成で切り替える ---
    # ★ 2026-07-24: 直接制御版にエピソード継続モード (relaunch_ball_after_save) を
    #   入れたため、1 エピソードに複数球が入るようになった。エピソード単位の
    #   終了フラグで測ると「全球セーブできたか」になり、1 球あたりのセーブ率 p に
    #   対して成功率 ≈ p^(球数) と極端に厳しくなる (p=0.96 でも 4 球なら 0.85)。
    #   閾値 0.85 では初速がまず上がらないので、継続モードでは 1 球あたりで測る。
    #
    #   モード判定は ``save_success`` が終了条件に登録されているかで行う
    #   (階層版と直接制御版の旧設定はこれを DoneTerm に持つ = 1 球 1 エピソード)。
    if "save_success" in tm.active_terms:
        # 従来モード: 1 球 = 1 エピソード。終了フラグがそのまま成否。
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in ("time_out", "save_success"):
            if name in tm.active_terms:
                success |= tm.get_term(name)
        success &= ~conceded
        batch = success[env_ids].float()
        n = batch.numel()
    else:
        # 継続モード: セーブ実績カウントと失点フラグから 1 球あたりの率を出す。
        from .observations import gk_buffers

        bufs = gk_buffers(env)
        saved = bufs["save_count"][env_ids].float()   # このエピソードでセーブした球数
        lost = conceded[env_ids].float()              # 失点は 1 エピソードにつき最大 1
        faced = saved + lost                          # 対峙した球数
        # 1 球も対峙していない env (転倒・場外で即終了など) は証拠にならないので除外。
        valid = faced > 0
        n = int(valid.sum().item())
        if n == 0:
            return float(env._gk_speed_hi.item())
        batch = saved[valid] / faced[valid]

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
