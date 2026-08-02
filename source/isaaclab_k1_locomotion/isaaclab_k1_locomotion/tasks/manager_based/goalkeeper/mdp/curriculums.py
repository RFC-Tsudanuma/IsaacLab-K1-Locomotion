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


def _update_success_ema(env: "ManagerBasedRLEnv", env_ids, p) -> int:
    """リセットされた env の成否から成功率 EMA を更新し、採用した件数 n を返す。

    :func:`adaptive_ball_speed` の判定部分をそのまま切り出したもの。
    n == 0 のときは証拠が無かったということなので、呼び出し側は調整をスキップする。
    """
    tm = env.termination_manager
    conceded = (
        tm.get_term("goal_conceded")
        if "goal_conceded" in tm.active_terms
        else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )

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
        saved = bufs["save_count"][env_ids].float()
        lost = conceded[env_ids].float()
        faced = saved + lost
        valid = faced > 0
        n = int(valid.sum().item())
        if n == 0:
            return 0
        batch = saved[valid] / faced[valid]

    env._gk_episode_count += n
    alpha = min(1.0, float(p.adaptive_ema_alpha) * n)
    env._gk_success_ema = (1.0 - alpha) * env._gk_success_ema + alpha * batch.mean()
    return n


def adaptive_difficulty(
    env: "ManagerBasedRLEnv",
    env_ids,
) -> dict:
    """セーブ成功率 EMA に応じて **難易度を 2 軸で** 上げ下げする適応カリキュラム。

    :func:`adaptive_ball_speed` (初速のみ) の後継。判定ロジック (1 球あたりの成功率、
    ウォームアップ、中立リセット) は共通で、動かす対象を増やしたもの。

    昇順の並び (易 → 難):
        1. ``aim_y_range`` を ``aim_y_stages`` に沿って段階的に広げる
        2. 広げ切ったら ``ball_speed_hi`` を ``adaptive_speed_delta`` ずつ上げる
    降順はその逆 (直近に上げた軸から戻す)。

    ★ なぜ ``aim_y_range`` を先に動かすのか:
      本タスクの設計メモにある実測 —「セーブ可否はほぼ **必要横移動量** だけで決まり、
      0.7m で成功率が半減する」— の通り、難易度の主因は初速ではなく横移動量である。
      ところが従来のカリキュラムは初速しか動かしておらず、``aim_y_range`` は
      最初から最大値 (±1.1) 固定だった。その分布では **37% の球が初期状態で
      「成功率半減」領域** に入っており、成功率が 62% 前後で頭打ちになる。
      引き上げ閾値 0.85 / 引き下げ閾値 0.55 の不感帯にはまり、35000 iter 回しても
      初速が 1.0 → 1.2 しか動かない (実質休眠) 状態だった。
      まず「届く範囲」を狭くして確実に止められるようにし、そこから広げる。

    Returns:
        ログ用の dict (Curriculum/<term名>/<key> として TensorBoard に出る)。
    """
    p = _gk_params(env)
    stages = [float(s) for s in p.aim_y_stages]

    if getattr(env, "_gk_speed_hi", None) is None:
        env._gk_speed_hi = torch.tensor(float(p.ball_speed_max), device=env.device)
        env._gk_success_ema = torch.tensor(0.5, device=env.device)
        env._gk_episode_count = 0
        env._gk_aim_stage = 0
        env._gk_aim_y = torch.tensor(stages[0], device=env.device)
        env._gk_cooldown = 0

    def _log() -> dict:
        return {
            "aim_stage": float(env._gk_aim_stage),
            "aim_y_range": float(env._gk_aim_y.item()),
            "ball_speed_hi": float(env._gk_speed_hi.item()),
            "success_ema": float(env._gk_success_ema.item()),
            "cooldown_left": float(max(0, env._gk_cooldown - env._gk_episode_count)),
        }

    if env_ids is None or len(env_ids) == 0:
        return _log()
    if _update_success_ema(env, env_ids, p) == 0:
        return _log()
    if env._gk_episode_count < int(p.adaptive_warmup_episodes):
        return _log()

    # --- 難易度を変えた直後は、新しい難易度での実績が溜まるまで判定を止める ---
    if env._gk_episode_count < env._gk_cooldown:
        return _log()

    ema = env._gk_success_ema.item()
    neutral = 0.5 * (float(p.adaptive_success_threshold) + float(p.adaptive_fail_threshold))
    top_stage = len(stages) - 1

    if ema > float(p.adaptive_success_threshold):
        # 易 → 難: まず狙い先を広げ、広げ切ってから初速を上げる
        if env._gk_aim_stage < top_stage:
            env._gk_aim_stage += 1
            env._gk_aim_y.fill_(stages[env._gk_aim_stage])
        else:
            env._gk_speed_hi = (env._gk_speed_hi + float(p.adaptive_speed_delta)).clamp(
                max=float(p.ball_speed_cap)
            )
        env._gk_success_ema.fill_(neutral)
        env._gk_cooldown = env._gk_episode_count + int(p.adaptive_cooldown_episodes)
    elif ema < float(p.adaptive_fail_threshold):
        # 難 → 易: 直近に上げた軸 (初速) から戻す
        if env._gk_speed_hi.item() > float(p.ball_speed_max) + 1e-6:
            env._gk_speed_hi = (env._gk_speed_hi - float(p.adaptive_speed_delta)).clamp(
                min=float(p.ball_speed_max)
            )
        elif env._gk_aim_stage > 0:
            env._gk_aim_stage -= 1
            env._gk_aim_y.fill_(stages[env._gk_aim_stage])
        env._gk_success_ema.fill_(neutral)
        env._gk_cooldown = env._gk_episode_count + int(p.adaptive_cooldown_episodes)

    return _log()
