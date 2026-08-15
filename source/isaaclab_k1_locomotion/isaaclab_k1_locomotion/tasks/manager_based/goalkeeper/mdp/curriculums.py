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

import json
import os
import torch
from typing import TYPE_CHECKING

from .events import _gk_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# カリキュラム進捗の永続化
#
# ★ 2026-08-15 追加。rsl_rl の save() が保存するのはモデル・オプティマイザ・iter だけで、
#   カリキュラムの到達点 (_gk_speed_hi / _gk_aim_stage) は **resume のたびに初期値へ
#   巻き戻る**。直接制御版のログを調べたところ、12 本のランすべてが speed 1.00 から
#   始まっており、20000 iter かけて 2.3〜4.1 まで上げては次のランで捨てる、を
#   繰り返していた。これを止めるために自前で永続化する。
#
#   ckpt 本体に入れず別ファイルにしてあるのは、rsl_rl に手を入れずに済ませるため。
#   保存先は「学習ログのランディレクトリ」で、--resume 時は渡された ckpt のあるラン
#   ディレクトリから読む (train_goalkeeper.py が env に _gk_curriculum_paths を設定)。
# ---------------------------------------------------------------------------

_STATE_FILENAME = "curriculum_state.json"


def _curriculum_state_path(env: "ManagerBasedRLEnv", key: str) -> str | None:
    """保存/読み込み先のパスを返す。未設定なら None (永続化を行わない)。"""
    paths = getattr(env, "_gk_curriculum_paths", None)
    if not paths:
        return None
    d = paths.get(key)
    return os.path.join(d, _STATE_FILENAME) if d else None


def _load_curriculum_state(env: "ManagerBasedRLEnv") -> dict:
    """resume 元のランディレクトリから進捗を読む。無ければ空 dict。"""
    path = _curriculum_state_path(env, "load")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:  # 壊れていても学習は続ける (最易段から始まるだけ)
        print(f"[goalkeeper] カリキュラム進捗の読み込みに失敗しました ({e})。最易段から開始します。")
        return {}


def save_curriculum_state(env: "ManagerBasedRLEnv") -> None:
    """現在の進捗を保存する。``adaptive_difficulty`` が難易度を変えるたびに呼ぶ。"""
    path = _curriculum_state_path(env, "save")
    if not path:
        return
    state = {
        "aim_stage": int(getattr(env, "_gk_aim_stage", 0)),
        "ball_speed_hi": float(getattr(env, "_gk_speed_hi", torch.tensor(0.0)).item()),
        "success_ema": float(getattr(env, "_gk_success_ema", torch.tensor(0.5)).item()),
        "episode_count": int(getattr(env, "_gk_episode_count", 0)),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, path)          # 書き込み中の中断で壊れないよう原子的に置換
    except Exception as e:
        print(f"[goalkeeper] カリキュラム進捗の保存に失敗しました ({e})。学習は続行します。")


def _neutral_ema(p) -> float:
    """難易度を変えた直後に EMA を戻す中立値。

    既定 (``adaptive_neutral_ema = None``) は success と fail の中点で、これは従来の
    挙動そのもの。明示すると 2 つの閾値から独立に決められる。

    分離が要る理由: 降格を減らそうと ``adaptive_fail_threshold`` を下げると、中点も
    一緒に下がって「中立値 → success_threshold」の距離が伸び、**昇格が遅くなる**。
    往復を止めることと昇格を速く保つことが両立できなくなる。
    """
    v = getattr(p, "adaptive_neutral_ema", None)
    if v is not None:
        return float(v)
    return 0.5 * (float(p.adaptive_success_threshold) + float(p.adaptive_fail_threshold))


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
        neutral = _neutral_ema(p)
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
        # ★ 2026-08-11: 「到達不能球」(hard_ball_prob で混ぜたもの) で失点したエピソードは
        #   丸ごと集計から除外する。取れない球はほぼ確実に失点するので、含めると
        #   success_ema が恒常的に下がり、適応カリキュラムが「難しすぎる」と誤判定して
        #   難易度を下げ続けてしまう (混入率 15% なら成功率が 15pt 目減りする)。
        #   除外すれば、到達不能球はカリキュラムに対して中立になる。
        # ★ 2026-08-14: 幾何判定で「そのキーパー位置から物理的に取れない」と分かった球
        #   での失点も除外する (:func:`~.events._mark_unreachable`)。
        #   ユーザー指示: 昇格閾値 (0.85) を下げるのではなく分母から外す。キーパーは
        #   可能な限り全部止めるべきなので基準は下げず、「右端にいるときに左端へ速い球」
        #   のような物理的に不可能な球でカリキュラムが止まるのだけを防ぐ。
        #   失点ペナルティ (-500) は除外しないので、ポリシーは諦めずに向かう。
        skip = bufs["hard_ball"][env_ids] | bufs["unreachable"][env_ids]
        hard_lost = conceded[env_ids] & skip
        valid = (faced > 0) & (~hard_lost)
        n = int(valid.sum().item())
        if n == 0:
            return 0
        # ★ 2026-08-09: 「エピソードごとの比率の平均」から「球で重み付けした率」に変更。
        #   エピソードは初失点で終わるので、前者は 1 球あたりの成功率より系統的に低く出る
        #   (失点したエピソードは必ず分母が小さい状態で打ち切られるため)。
        #   実測: 学習ログ EMA 0.755 に対し、同条件の 1 球あたり実測は 0.851 (差 10pt)。
        #   この 10pt のぶんだけ閾値 adaptive_success_threshold=0.85 が実質 0.915 相当に
        #   なっており、カリキュラムが不感帯で停止する主因だった (34600 iter 回して
        #   ball_speed_hi が 1.0→1.55 までしか動かなかった)。
        batch = saved[valid].sum() / faced[valid].sum()

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
        # ★ 2026-08-15: 保存済みの進捗があれば復元する。無ければ最易段から。
        #   rsl_rl の save() はモデル・オプティマイザ・iter しか保存しないので、
        #   カリキュラムの到達点は自前で永続化しないと **resume のたびに 1.0 / 0.4 へ
        #   巻き戻る**。直接制御版が 12 本のランを回して毎回 speed 1.00 から
        #   やり直していたのはこれが原因 (ログで確認済み)。
        st = _load_curriculum_state(env)
        env._gk_speed_hi = torch.tensor(
            st.get("ball_speed_hi", float(p.ball_speed_max)), device=env.device
        )
        env._gk_success_ema = torch.tensor(st.get("success_ema", 0.5), device=env.device)
        env._gk_episode_count = int(st.get("episode_count", 0))
        env._gk_aim_stage = int(st.get("aim_stage", 0))
        env._gk_aim_y = torch.tensor(
            stages[min(env._gk_aim_stage, len(stages) - 1)], device=env.device
        )
        env._gk_cooldown = 0
        if st:
            print(
                f"[goalkeeper] カリキュラム進捗を復元: aim_stage={env._gk_aim_stage}"
                f" / ball_speed_hi={env._gk_speed_hi.item():.2f}"
                f" / episodes={env._gk_episode_count}"
            )

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
    neutral = _neutral_ema(p)
    top_stage = len(stages) - 1

    if ema > float(p.adaptive_success_threshold):
        # 易 → 難: まず狙い先を広げ、広げ切ってから初速を上げる
        if env._gk_aim_stage < top_stage:
            env._gk_aim_stage += 1
            env._gk_aim_y.fill_(stages[env._gk_aim_stage])
        else:
            # ★ 2026-08-15: 加算から **乗算** に変更。
            #   加算 0.05 は上限が 3.0 だった頃の設定で、cap 6.0 だと 1.0 → 6.0 に
            #   100 回の昇格が必要になる。実測 1 昇格 ≈ 3400 iter なので 340,000 iter
            #   (約 8 日) かかる計算で、実用にならなかった。
            #   乗算 ×1.2 なら 10 回で到達する。速い球ほど 0.05 m/s の差は相対的に
            #   小さいので、比率で上げる方が難易度の刻みとしても素直。
            #   ratio <= 1.0 を指定した場合は従来どおり adaptive_speed_delta の加算。
            ratio = float(getattr(p, "adaptive_speed_ratio", 1.0))
            if ratio > 1.0:
                env._gk_speed_hi = (env._gk_speed_hi * ratio).clamp(max=float(p.ball_speed_cap))
            else:
                env._gk_speed_hi = (env._gk_speed_hi + float(p.adaptive_speed_delta)).clamp(
                    max=float(p.ball_speed_cap)
                )
        env._gk_success_ema.fill_(neutral)
        env._gk_cooldown = env._gk_episode_count + int(p.adaptive_cooldown_episodes)
        save_curriculum_state(env)
    elif ema < float(p.adaptive_fail_threshold):
        # 難 → 易: 直近に上げた軸 (初速) から戻す
        #
        # ★ 2026-08-14: **実際に難易度を下げられたときだけ** EMA のリセットと
        #   クールダウンを行うよう修正した。以前は最易段 (aim_stage=0 かつ
        #   hi=ball_speed_max) で下げる余地が無くても、EMA を neutral (=0.70) に
        #   戻して 3000 エピソードの判定停止を張っていた。難易度は 1 ミリも動いて
        #   いないのに実績を捨てるので、
        #     * EMA が実態 (実測 0.6) より高い 0.70 に繰り返し引き戻される
        #     * その間の判定が止まる
        #   となり、最易段で詰まったときに「詰まっている」という事実がログ上でも
        #   薄まっていた (実際に 12500 iter 停滞したランで cooldown_left が
        #   常時非ゼロだった)。下げられないなら何もせず、EMA を素直に育てる。
        lowered = False
        if env._gk_speed_hi.item() > float(p.ball_speed_max) + 1e-6:
            ratio = float(getattr(p, "adaptive_speed_ratio", 1.0))
            if ratio > 1.0:
                env._gk_speed_hi = (env._gk_speed_hi / ratio).clamp(min=float(p.ball_speed_max))
            else:
                env._gk_speed_hi = (env._gk_speed_hi - float(p.adaptive_speed_delta)).clamp(
                    min=float(p.ball_speed_max)
                )
            lowered = True
        elif env._gk_aim_stage > 0:
            env._gk_aim_stage -= 1
            env._gk_aim_y.fill_(stages[env._gk_aim_stage])
            lowered = True
        if lowered:
            env._gk_success_ema.fill_(neutral)
            env._gk_cooldown = env._gk_episode_count + int(p.adaptive_cooldown_episodes)
            save_curriculum_state(env)

    return _log()


def adaptive_hard_ball(
    env: "ManagerBasedRLEnv",
    env_ids,
) -> dict:
    """難易度カリキュラムが上限/頭打ちに達したら「到達不能球」を自動で混ぜ始める。

    **必ず :func:`adaptive_difficulty` より後に登録すること。** 本関数は成功率 EMA を
    更新せず、``adaptive_difficulty`` が書いた ``_gk_speed_hi`` / ``_gk_episode_count``
    を読むだけなので、先に走らせないと 1 ステップ古い状態を見ることになる
    (CurriculumManager は cfg の定義順に実行する)。

    なぜ自動化するか:
        到達不能球の初速は ``ball_speed_cap`` ではなく **その時点の初速上限 hi の
        ``hard_ball_speed_mult`` 倍** から引かれる。学習初期 (hi=1.0) に有効化しても
        1.0〜1.6 m/s にしかならず「不能」にならないので、有効化はカリキュラムが
        伸び切ってからでなければならない。しかしそれを手動運用にすると、忘れたときに
        **取れない球を一度も経験していないポリシーがそのまま実機へ行く**。実測では
        そのポリシーに到達不能球を与えると転倒が 1 回 → 56 回に増える。リセット条件が
        転倒である以上、これは最悪の失敗モードなので構造的に防ぐ。

    有効化のトリガ (いずれか):
        1. ``ball_speed_hi >= ball_speed_cap`` — これ以上難しくならない
        2. ``ball_speed_hi`` が ``hard_ball_plateau_episodes`` の間まったく動かない
        3. 総エピソード数が ``hard_ball_force_episodes`` を超えた (保険)

    有効化後は ``hard_ball_ramp_episodes`` ごとに ``hard_ball_step`` ずつ増やし、
    ``hard_ball_prob_max`` で頭打ちにする。一度上げた混入率は下げない
    (カリキュラムが再び動き出しても、取れない球の経験は保持したいため)。

    ``hard_ball_prob`` は ``GoalkeeperParamsCfg`` の値を直接書き換える。
    :func:`~.events.reset_ball_shot` は毎回 cfg から読み直すのでそのまま反映される。

    Returns:
        ログ用 dict (Curriculum/<term名>/<key> として TensorBoard に出る)。
    """
    p = _gk_params(env)

    if getattr(env, "_gk_hard_state", None) is None:
        # last_hi: 直近に観測した初速上限 / since: それが変わってからのエピソード数の基点
        env._gk_hard_state = {"last_hi": None, "since": 0, "started": False, "next_step_at": 0}

    st = env._gk_hard_state
    hi_buf = getattr(env, "_gk_speed_hi", None)
    episodes = int(getattr(env, "_gk_episode_count", 0))

    def _log() -> dict:
        return {
            "hard_ball_prob": float(p.hard_ball_prob),
            "started": float(st["started"]),
            "plateau_episodes": float(episodes - st["since"]),
        }

    # 自動化が無効、または難易度カリキュラムがまだ初期化されていない
    if not bool(getattr(p, "hard_ball_auto", False)) or hi_buf is None:
        return _log()

    hi = float(hi_buf.item())
    if st["last_hi"] is None or abs(hi - st["last_hi"]) > 1e-6:
        # 初速上限が動いた → 頭打ち判定の基点をリセット
        st["last_hi"] = hi
        st["since"] = episodes

    if not st["started"]:
        # ★ 2026-08-14: 「難易度が十分上がっていること」を前提条件に追加した。
        #   以前は「ball_speed_hi が動かない」だけを頭打ち判定にしていたが、これは
        #   **「上限で頭打ち」と「一度も上がっていない」を区別できない**。実際、
        #   最易段 (aim_stage=0 / hi=1.0) から一度も動かないランで判定が成立し、
        #   到達不能球が最易段で有効化された。到達不能球の初速は cap ではなく
        #   **その時点の hi の hard_ball_speed_mult 倍** なので、hi=1.0 では
        #   1.0〜1.6 m/s にしかならず「不能」にならない = 立ち上がりを遅くするだけ。
        #   狙い先が最終段まで広がっていることを必須条件にする。
        top_stage = len(p.aim_y_stages) - 1
        advanced = int(getattr(env, "_gk_aim_stage", 0)) >= top_stage
        at_cap = advanced and hi >= float(p.ball_speed_cap) - 1e-6
        plateaued = advanced and (episodes - st["since"]) >= int(p.hard_ball_plateau_episodes)
        # 保険だけは難易度に関係なく発火させる (学習が終わるまでに一度も経験しない
        # 事態を防ぐのが目的なので)。ただし閾値は実測ペースに合わせること:
        # 4096 env / 25s エピソードで **1 iter あたり約 63 エピソード** 進む
        # (実測: 12500 iter で 786,000 エピソード)。
        forced = episodes >= int(p.hard_ball_force_episodes)
        if at_cap or plateaued or forced:
            st["started"] = True
            st["next_step_at"] = episodes
            reason = "cap" if at_cap else ("plateau" if plateaued else "force")
            print(
                f"[goalkeeper] 到達不能球の混入を開始します (trigger={reason},"
                f" hi={hi:.2f}, episodes={episodes})"
            )

    if st["started"] and episodes >= st["next_step_at"]:
        target = float(p.hard_ball_prob_max)
        if float(p.hard_ball_prob) < target - 1e-9:
            p.hard_ball_prob = min(target, float(p.hard_ball_prob) + float(p.hard_ball_step))
        st["next_step_at"] = episodes + int(p.hard_ball_ramp_episodes)

    return _log()
