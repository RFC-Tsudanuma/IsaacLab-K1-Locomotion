# mdp/curriculums.py
from __future__ import annotations
import math
import os
import torch
from typing import TYPE_CHECKING

# K1_GATE_DEBUG=1 で kick_rate_gated_speed_range の内部状態を定期的に print する。
_GATE_DEBUG = os.environ.get("K1_GATE_DEBUG", "") not in ("", "0")

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def scale_feet_landing_penalty(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    threshold: float = 1.4,
    scale: float = 2.0,
) -> torch.Tensor | None:
    episode_extras = env.extras.get("log", {})

    lin_rew = episode_extras.get("Episode_Reward/track_lin_vel_xy_exp", None)
    ang_rew = episode_extras.get("Episode_Reward/track_ang_vel_z_exp", None)

    if lin_rew is None or ang_rew is None:
        return None

    # 初回だけ元のweightを記憶
    if not hasattr(env, "_feet_landing_base_weight"):
        env._feet_landing_base_weight = env.reward_manager.get_term_cfg("feet_landing_velocity").weight

    # スカラーでも配列でも対応
    def to_scalar(x):
        if isinstance(x, torch.Tensor):
            return x.mean().item()
        return float(x)

    combined = to_scalar(lin_rew) + to_scalar(ang_rew)

    base_weight = env._feet_landing_base_weight
    target_weight = base_weight * scale if combined > threshold else base_weight

    term = env.reward_manager.get_term_cfg("feet_landing_velocity")
    if abs(term.weight - target_weight) > 1e-6:
        term.weight = target_weight

    return torch.tensor(combined)


def window_reward_weight(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    weight: float,
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
) -> None:
    """start_step < step <= end_step の期間だけ weight を適用し、それ以外は 0 にする。"""
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter

    new_weight = weight if start_step < step <= end_step else 0.0

    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - new_weight) > 1e-8:
        term.weight = new_weight


def linear_reward_weight(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
) -> None:
    """ステップ数（またはiteration数）に応じて報酬重みを線形補間するカリキュラム。

    steps_per_iteration > 0 の場合、start_step/end_step をiteration単位として解釈する。
    steps_per_iteration = 0（デフォルト）の場合、common_step_counter（物理ステップ数）を使う。
    """
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter
    if step <= start_step:
        new_weight = start_weight
    elif step >= end_step:
        new_weight = end_weight
    else:
        alpha = (step - start_step) / (end_step - start_step)
        new_weight = start_weight + (end_weight - start_weight) * alpha

    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - new_weight) > 1e-8:
        term.weight = new_weight


def piecewise_reward_weight(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    knots: list[tuple[float, float]],
    steps_per_iteration: int = 0,
) -> dict[str, float]:
    """折れ線で報酬重みを動かすカリキュラム。``linear_reward_weight`` の多段版。

    ``knots`` は ``[(step, weight), ...]`` を step の昇順で与える。区間内は線形補間、
    最初の knot より前は最初の weight、最後の knot より後は最後の weight で固定する。
    ``steps_per_iteration > 0`` なら step を iteration 単位として解釈するのは
    :func:`linear_reward_weight` と同じ。

    **なぜ 1 本の線形ランプでは足りないか** (walk_weak_kick の ``kick_velocity_strong``):
    この項は「立ち上げてから落とす」必要がある。0 から学習し直すタスクでは、まず
    ``kick_velocity_strong`` を効かせて **「ボールを強く蹴る」という行動そのものを
    獲得させ** (これが無いとキックが発見されない)、獲得できてから 0 へ落として
    「指令どおりの強さで蹴る」へ移す。上げっぱなしだと常に全力キックが最適のままで、
    下げっぱなしだとそもそもキックを発見できない。

    NOTE: 返り値の dict は ``Curriculum/<term_name>/weight`` として TensorBoard に出る。
          いま折れ線のどこにいるかを kick_rate と並べて読むこと。
    """
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter

    if step <= knots[0][0]:
        new_weight = knots[0][1]
    elif step >= knots[-1][0]:
        new_weight = knots[-1][1]
    else:
        new_weight = knots[-1][1]
        for (s0, w0), (s1, w1) in zip(knots[:-1], knots[1:]):
            if s0 <= step <= s1:
                alpha = 0.0 if s1 == s0 else (step - s0) / (s1 - s0)
                new_weight = w0 + (w1 - w0) * alpha
                break

    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - new_weight) > 1e-8:
        term.weight = new_weight

    return {"weight": new_weight}


def linear_reward_param(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    param_name: str,
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
) -> dict[str, float]:
    """報酬項の **params の 1 つ** を線形にアニールするカリキュラム。

    :func:`linear_reward_weight` が weight を動かすのに対し、こちらはシェイピング係数
    (``sigma_velocity`` など) を動かす。weight と違って「効かせる強さ」ではなく
    「採点の厳しさ」を変えたいときに使う。

    **なぜ必要か** (walk_weak_kick の ``sigma_velocity``): 指令帯 (0.25, 2.0) の幅 1.75 に
    対して既定の σ=1.0 は太すぎ、指令 0.5 に対して v=1.5 を出しても
    exp(−((1.5−0.5)/1.0)²) = 0.37 とそこそこの点が付く。σ を絞れば指令間の差が付くが、
    最初から絞ると学習初期の下手なキックが軒並み 0 点になって勾配が死ぬ。
    そこで太い σ で始めて徐々に絞る。

    NOTE: ``RewardManager.get_term_cfg`` が返す cfg の ``params`` を書き換える。報酬関数は
          毎ステップ params を読み直すので次のステップから反映される。
    NOTE: 返り値は ``Curriculum/<term_name>/<param_name>`` として TensorBoard に出る。
    """
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter

    if step <= start_step:
        new_value = start_value
    elif step >= end_step:
        new_value = end_value
    else:
        alpha = (step - start_step) / (end_step - start_step)
        new_value = start_value + (end_value - start_value) * alpha

    term = env.reward_manager.get_term_cfg(term_name)
    term.params[param_name] = new_value

    return {param_name: new_value}


def piecewise_reward_param(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    param_name: str,
    knots: list[tuple[float, float]],
    steps_per_iteration: int = 0,
) -> dict[str, float]:
    """折れ線で報酬項の **params の 1 つ** を動かすカリキュラム。

    :func:`piecewise_reward_weight` の params 版であり、:func:`linear_reward_param` の
    多段版。``knots`` の意味・補間・端の飽和・``steps_per_iteration`` の扱いは
    :func:`piecewise_reward_weight` とまったく同じ (``[(step, value), ...]`` を step の
    昇順、区間内は線形補間、最初の knot より前は最初の値、最後の knot より後は
    最後の値で固定)。

    **なぜ linear_reward_param を 2 本並べてはいけないか**
    -----------------------------------------------------
    「1500 → 3000 で 0.45 → 0.25、そのあと 3000 → 4000 でさらに 0.15 へ」のような
    多段のアニールを、:func:`linear_reward_param` 2 本で書きたくなる。**これは壊れる。**
    あちらは ``step <= start_step`` のとき ``start_value`` を **書き込む** ので、
    2 本目 (start_step = 3000) は 3000 までずっと 0.25 を書き続け、1 本目は 3000 以降
    ずっと 0.25 を書き続ける。同じ param に 2 つの書き手がいる状態になり、
    最終的な値は **CurriculumManager がどちらの項を後に走らせたか** で決まる
    (項の順序は cfg の属性順という実装依存の要素)。1 本の折れ線にすれば書き手が
    1 つになり、この曖昧さが構造的に消える。

    NOTE: :func:`linear_reward_param` と同じく ``RewardManager.get_term_cfg`` が返す
          cfg の ``params`` を書き換える。報酬関数は毎ステップ params を読み直すので
          次のステップから反映される。
    NOTE: 返り値は ``Curriculum/<term_name>/<param_name>`` として TensorBoard に出る。
    """
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter

    if step <= knots[0][0]:
        new_value = knots[0][1]
    elif step >= knots[-1][0]:
        new_value = knots[-1][1]
    else:
        new_value = knots[-1][1]
        for (s0, v0), (s1, v1) in zip(knots[:-1], knots[1:]):
            if s0 <= step <= s1:
                alpha = 0.0 if s1 == s0 else (step - s0) / (s1 - s0)
                new_value = v0 + (v1 - v0) * alpha
                break

    term = env.reward_manager.get_term_cfg(term_name)
    term.params[param_name] = new_value

    return {param_name: new_value}


def linear_command_speed_range(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    command_name: str,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
) -> dict[str, float]:
    """KickDirectionCommand の ``target_speed_range`` を線形に動かすカリキュラム。

    下限・上限それぞれを ``start_range`` → ``end_range`` へ線形補間する。
    :func:`linear_reward_weight` と同じく、``steps_per_iteration > 0`` なら
    start_step/end_step を iteration 単位として解釈する。

    **なぜ必要か（walk_long_pass の失敗記録）**

    キック帯を一段で上げると、fine-tune の出発点となるポリシーの実蹴り速度が新しい帯の
    下限を大きく下回り、``kick_velocity_scaled`` の Gaussian と ``kick_direction`` の
    片側速度ゲートが揃って ~0 を返すようになる。一方 ``kick_finished`` は latch の
    2 秒後にエピソードを終わらせるので、**キックは常に「残りの歩行報酬を捨てる」コストを
    払っている**。キック報酬が消えるとこの収支が逆転し、ポリシーは「蹴らずに time_out
    まで歩き回る」に収束する（実測: 帯 (2.0,3.0) → (3.2,5.0) を一段で飛ばしたところ
    kick_rate 0.997 → 0.037、time_out 0.944、mean_reward はむしろ 12.1 → 21.5 に増加）。
    しかも探索 std が潰れるため、iteration を増やしても抜けられない局所最適になる。

    帯を滑らかに動かせば、各時点で「今のポリシーがぎりぎり届く速度」が指令され続けるので
    キック報酬が払われ、収支が逆転しない。``v_gate_frac`` は v_target への相対値なので
    帯に自動追従する（ゲート側を別途ランプする必要はない）。

    NOTE: ``KickDirectionCommand._resample_command`` は毎回 ``self.cfg.target_speed_range``
          を読み直すので、cfg を書き換えるだけで次のエピソードから反映される。
    NOTE: 返り値の dict は ``Curriculum/<term_name>/speed_min`` などとして TensorBoard に
          出る。帯が今どこまで動いたかを kick_rate と並べて読むこと。ランプ途中で
          kick_rate が落ち始めたら、その帯の値がそのスイングの実質的な速度上限。
    """
    if steps_per_iteration > 0:
        step = env.common_step_counter // steps_per_iteration
    else:
        step = env.common_step_counter

    if step <= start_step:
        alpha = 0.0
    elif step >= end_step:
        alpha = 1.0
    else:
        alpha = (step - start_step) / (end_step - start_step)

    speed_min = start_range[0] + (end_range[0] - start_range[0]) * alpha
    speed_max = start_range[1] + (end_range[1] - start_range[1]) * alpha

    term = env.command_manager.get_term(command_name)
    term.cfg.target_speed_range = (speed_min, speed_max)

    return {"speed_min": speed_min, "speed_max": speed_max}


# --------------------------------------------------------------------------- #
# 以下 2 つは fewa/walk_kick_dual_encoder_tune (walk_long_pass の dual encoder 化) からの
# 移植。walk_kick / walk_weak_kick / walk_middle_kick の既存タスクはどれも
# ``target_speed_range`` を動かすカリキュラムを持たない (weak は基底の帯のまま、
# middle は最初から終点固定) ので、**現時点でこの 2 つを使っているタスクは無い**。
# 帯を動かす新しい段を dual 系に足すときの受け皿としてここに置いてある。
#
# 使うときの注意: ``steps_per_iteration`` は PPO の ``num_steps_per_env`` と
# 一致させること。locomotion/agents/rsl_rl_ppo_cfg.py は 48 なので 48 が正しい
# (walk_kick 系の既存ランプは 24 で書かれており、実際には書いてある iteration の
#  半分で終わっている。既存の収束済み挙動を変えないため、そちらは触っていない)。
# --------------------------------------------------------------------------- #


def linear_reward_weight_after_speed_gate(
    env: ManagerBasedRLEnv,
    _env_ids: torch.Tensor,
    term_name: str,
    start_weight: float,
    end_weight: float,
    command_name: str,
    ramp_iterations: int,
    steps_per_iteration: int = 0,
    gate_alpha_min: float = 1.0,
) -> dict[str, float]:
    """:func:`kick_rate_gated_speed_range` の帯が目標に届いてから重みをランプする。

    :func:`linear_reward_weight` の壁時計版と違い、開始時刻を **帯カリキュラムの
    到達** に紐付ける。ゲート付きの帯は「蹴れている間しか進まない」ので目標に届く
    iteration が事前に決まらず、壁時計で後段の項を立ち上げると、帯がまだ途中の
    ポリシーに追加の圧力を掛けてしまう。

    ``command_name`` のゲートの α が ``gate_alpha_min`` に達した時点を 0 として、
    ``ramp_iterations`` かけて ``start_weight`` → ``end_weight`` へ動かす。
    帯がそこまで進まなければ重みは ``start_weight`` のまま (= 立ち上がらない)。

    Returns:
        ``Curriculum/<term_name>/`` 以下に出る現在の重みと経過 iteration。
    """
    if steps_per_iteration > 0:
        now = env.common_step_counter / steps_per_iteration
    else:
        now = float(env.common_step_counter)

    gate = getattr(env, "_kick_speed_gate_state", {}).get(command_name, None)
    state = getattr(env, "_gated_weight_ramp_state", None)
    if state is None:
        state = {}
        env._gated_weight_ramp_state = state

    # 帯が到達した瞬間の iteration を一度だけ記録する (以降は戻っても取り消さない。
    # 一度立ち上げた罰を帯の揺り戻しで出し入れすると、報酬の定義が振動する)。
    if term_name not in state and gate is not None and gate["alpha"] >= gate_alpha_min:
        state[term_name] = now

    started_at = state.get(term_name, None)
    if started_at is None:
        new_weight = start_weight
        elapsed = 0.0
    else:
        elapsed = max(0.0, now - started_at)
        alpha = 1.0 if ramp_iterations <= 0 else min(elapsed / ramp_iterations, 1.0)
        new_weight = start_weight + (end_weight - start_weight) * alpha

    term = env.reward_manager.get_term_cfg(term_name)
    if abs(term.weight - new_weight) > 1e-8:
        term.weight = new_weight

    return {"weight": new_weight, "iterations_since_gate": elapsed}


def kick_rate_gated_speed_range(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
    advance_above: float = 0.80,
    retreat_below: float = 0.50,
    ema_alpha: float = 0.01,
    retreat_scale: float = 2.0,
) -> dict[str, float]:
    """キック成立率で開閉するゲート付きの ``target_speed_range`` カリキュラム。

    :func:`linear_command_speed_range` の壁時計版に対して、**帯を進めるかどうかを
    ポリシーの実績で決める**。進捗 ``progress`` は iteration と同じ単位の内部時計で、
    直近のキック成立率 (EMA) が

    * ``advance_above`` 以上 … 経過ぶんだけ進む (= 壁時計版と同じ速さ)
    * ``retreat_below`` 未満 … ``retreat_scale`` 倍の速さで戻る
    * その間             … その場で止まる

    と動く。α は ``progress`` を ``start_step`` → ``end_step`` に写した値。

    **なぜ壁時計ではだめか**

    壁時計の線形ランプは「ポリシーがこの速さで付いてくる」という賭けで、外れたときに
    自己回復しない。帯がスイングの実力を追い越すと ``kick_velocity_scaled`` の Gaussian と
    ``kick_direction`` の片側速度ゲートが揃って ~0 を返し、一方で ``kick_finished`` は
    latch の 2 秒後にエピソードを終わらせるので、キックは「残りの歩行報酬を捨てる」
    コストだけが残る。収支が逆転してポリシーは「蹴らずに time_out まで歩く」へ落ち、
    探索 std も潰れるので iteration を足しても戻ってこない
    (:func:`linear_command_speed_range` の docstring の失敗記録)。壁時計はそのまま
    帯を上げ続けるため、この局所最適から抜ける道が塞がれる。

    ゲート付きなら、キックが崩れた時点で帯が止まり、崩れ続ければ**キックが成立していた
    帯まで戻る**。報酬信号が復活するので、ポリシーは自力で戻れる。

    **kick_rate の読み方**

    ``env.command_manager`` の ``kick_rate`` メトリクスは「そのエピソードで値 latch が
    起きたか」の 0/1 を env ごとに持ち、``CommandManager.reset`` が平均を取ってから
    ゼロ化する。``ManagerBasedRLEnv._reset_idx`` は **curriculum → command reset** の順に
    呼ぶので、ここで ``env_ids`` を見れば、これから
    ``Metrics/kick_direction/kick_rate`` として記録されるのと同じ値
    (= 今終わったエピソード群の成立率) をゼロ化前に読める。

    EMA の初期値は 1.0。``start_step`` までは帯が動かないので、そこまでに実測値へ
    十分収束する (4096 env なら 1 iteration あたり数百エピソードが終わる)。

    Returns:
        TensorBoard に ``Curriculum/<term_name>/`` 以下で出る値。``alpha`` が止まって
        いるのに ``kick_rate_ema`` が低いままなら、その帯がスイングの実質的な上限。
    """
    if end_step <= start_step:
        raise ValueError(f"end_step ({end_step}) は start_step ({start_step}) より大きくすること。")

    if steps_per_iteration > 0:
        now = env.common_step_counter / steps_per_iteration
    else:
        now = float(env.common_step_counter)

    state = getattr(env, "_kick_speed_gate_state", None)
    if state is None:
        state = {}
        env._kick_speed_gate_state = state
    if command_name not in state:
        state[command_name] = {"alpha": 0.0, "kick_rate_ema": 1.0, "last_now": now}
    gate = state[command_name]

    # -- 直近に終わったエピソード群のキック成立率を EMA に入れる
    command_term = env.command_manager.get_term(command_name)
    metric = command_term.metrics.get("kick_rate", None)
    if metric is not None and env_ids is not None and len(env_ids) > 0:
        kick_rate = float(metric[env_ids].mean())
        gate["kick_rate_ema"] += ema_alpha * (kick_rate - gate["kick_rate_ema"])

    # -- ゲートの開閉に応じて α を進める / 戻す
    #
    # start_step までは触らない (fine-tune 直後の、履歴入力と std リセットに慣れるまでの
    # 落ち着き期間)。その後の公称速度は壁時計版と同じ 1/(end_step - start_step) / iteration。
    elapsed = max(0.0, now - gate["last_now"])
    gate["last_now"] = now
    ema = gate["kick_rate_ema"]
    if now >= start_step:
        delta = elapsed / (end_step - start_step)
        if ema >= advance_above:
            gate["alpha"] += delta
        elif ema < retreat_below:
            gate["alpha"] -= delta * retreat_scale
        # advance_above > ema >= retreat_below は据え置き (ヒステリシス)。
        gate["alpha"] = min(max(gate["alpha"], 0.0), 1.0)

    if _GATE_DEBUG:
        gate["calls"] = gate.get("calls", 0) + 1
        if gate["calls"] <= 5 or gate["calls"] % 25 == 0:
            print(
                f"[gate] calls={gate['calls']} now={now:.3f} elapsed={elapsed:.5f}"
                f" alpha={gate['alpha']:.6f} ema={ema:.4f} id(env)={id(env)}"
            )

    alpha = gate["alpha"]
    speed_min = start_range[0] + (end_range[0] - start_range[0]) * alpha
    speed_max = start_range[1] + (end_range[1] - start_range[1]) * alpha

    command_term.cfg.target_speed_range = (speed_min, speed_max)

    return {
        "speed_min": speed_min,
        "speed_max": speed_max,
        "alpha": alpha,
        "kick_rate_ema": ema,
    }


def kick_rate_gated_expansion(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start_step: int,
    end_step: int,
    steps_per_iteration: int = 0,
    advance_above: float = 0.80,
    retreat_below: float = 0.50,
    ema_alpha: float = 0.01,
    retreat_scale: float = 2.0,
    apex_metric_name: str | None = None,
    apex_advance_above: float = 0.40,
    apex_retreat_below: float = 0.25,
    ball_event_name: str = "reset_ball",
    half_angle_range: tuple[float, float] = (1.047, math.pi),
    dist_range_start: tuple[float, float] = (0.5, 0.8),
    dist_range_end: tuple[float, float] = (0.5, 1.5),
    heading_halfwidth_range: tuple[float, float] = (math.pi / 4, math.pi),
    approach_term_name: str = "approach_penalty",
    approach_end_weight: float = -3.0,
    approach_fade_iterations: int = 500,
    avoidance_term_name: str = "ball_avoidance",
    avoidance_end_weight: float = -3.0,
) -> dict[str, float]:
    """キック成立率で開閉するゲートで、**限定レンジ → 全方位** を一本の α で進める。

    :func:`kick_rate_gated_speed_range` とゲートの作りは同じ (EMA、``advance_above`` /
    ``retreat_below`` のヒステリシス、戻りは ``retreat_scale`` 倍速)。違うのは α が
    動かす先だけで、こちらは次の 5 つを **同時に** 線形補間する:

    1. ``events.<ball_event_name>.params["half_angle"]``  … ボール出現の方位範囲
    2. ``events.<ball_event_name>.params["dist_range"]``  … ボール出現の距離
    3. ``commands.<command_name>.ranges.heading``         … 蹴り方向の範囲 (±半幅)
    4. ``rewards.<approach_term_name>.weight``            … 接近圧 (下の式)
    5. ``rewards.<avoidance_term_name>.weight``           … 回り込み圧 (``end × α``)

    approach_penalty と ball_avoidance のクロスフェード
    --------------------------------------------------
    2 つは向きが逆の項で、限定レンジでは接近圧 (approach_penalty) が要り、全方位では
    「構えができるまで寄るな」の抑止 (ball_avoidance) が要る
    (:func:`..rewards.ball_avoidance` の docstring)。段を分けずに 1 本の run で
    やりたいので、α で入れ替える::

        w_a  = approach_end_weight × clamp(now / approach_fade_iterations, 0, 1)
        approach_penalty.weight  = w_a × (1 − α)
        ball_avoidance.weight    = avoidance_end_weight × α

    ``w_a`` の側 (壁時計の 0 → 500 iteration フェードイン) は基底 walk_kick が
    ``linear_reward_weight`` で入れているものと同じ立ち上がりを、**この関数の中で
    再現している**。基底のカリキュラム項をそのまま残すと同じ weight を 2 つの
    curriculum 項が書き合い、どちらが最後に走るかで値が決まってしまうため。
    このゲートを使うタスクでは基底の ``approach_penalty_weight`` を ``None`` にして、
    書き手をこの関数 1 つに絞ること。

    積 (``w_a × (1 − α)``) にしてあるのは、フェードインの途中でゲートが開き始めても
    「立ち上がりきっていない接近圧をさらに薄める」という素直な向きに合成されるため。
    ``min`` だと α が進むほど ``w_a`` の立ち上がりの形が階段状に切り取られる。

    なぜ壁時計で広げないか
    ----------------------
    ボールの出現範囲と蹴り方向の範囲は **キックの難しさそのもの** を決める軸なので、
    ポリシーの実力より先に広げると回り込みが間に合わずキックが成立しなくなる。
    キック報酬が消えると ``kick_finished`` の「残りの歩行報酬を捨てるコスト」だけが
    残って収支が逆転し、「蹴らずに time_out まで歩く」へ落ちる
    (:func:`linear_command_speed_range` の失敗記録と同じ機構)。壁時計はそこから
    自己回復しない。ゲート付きなら崩れた時点で止まり、崩れ続ければ蹴れていた範囲まで
    戻る。

    第 2 の指標でゲートを絞る (``apex_metric_name``)
    ------------------------------------------------
    既定 ``None`` = **キック成立率だけ**を見る従来の挙動 (inside / weak / middle の
    呼び出し元は 1 ビットも変わらない)。

    ロブ系ではこれだけでは足りない。``kick_rate`` は「蹴れたか」しか見ないので、
    **浮かせるのをやめてトーキックで転がしても 1.0 のまま**になり、apex が
    立ち上がらないままゲートだけが全方位まで開き切る。そこで
    ``apex_metric_name="kick_apex_height"`` を渡すと、同じ command term の
    メトリクスをもう 1 本 EMA で追い、次の複合条件で α を動かす::

        前進: kick_rate_ema >= advance_above  かつ  apex_ema >= apex_advance_above
        後退: kick_rate_ema <  retreat_below  または apex_ema <  apex_retreat_below
        それ以外は据え置き (ヒステリシス)

    「前進は AND / 後退は OR」なので、**どちらか一方が崩れた時点で拡大が止まる**。
    第 2 指標の EMA は ``apex_advance_above`` から始める (``kick_rate_ema`` の初期値
    1.0 と同じ「まだ実測が無いうちは前進を妨げない」向き)。

    実装の重複について
    ------------------
    ゲートの計算そのものは :func:`kick_rate_gated_speed_range` とほぼ同じコードだが、
    **共通化のためにあちらへ手を入れることは意図的に避けている**。あちらは
    walk_long_pass 系の収束済みの run が依存している関数で、リファクタで挙動が
    1 ビットでも変われば既存 run の再現性が失われる。ロジックの重複はその対価として
    受け入れる。ゲート状態も別の属性 (``_kick_expansion_gate_state``) に持ち、
    あちらの ``_kick_speed_gate_state`` とは共有しない。

    Returns:
        TensorBoard に ``Curriculum/<term_name>/`` 以下で出る値。``alpha`` が止まって
        いるのに ``kick_rate_ema`` が低いままなら、その範囲が今のポリシーの実質的な上限。
    """
    if end_step <= start_step:
        raise ValueError(f"end_step ({end_step}) は start_step ({start_step}) より大きくすること。")

    if steps_per_iteration > 0:
        now = env.common_step_counter / steps_per_iteration
    else:
        now = float(env.common_step_counter)

    state = getattr(env, "_kick_expansion_gate_state", None)
    if state is None:
        state = {}
        env._kick_expansion_gate_state = state
    if command_name not in state:
        # apex_ema の初期値を「前進を妨げない値」に置くのは kick_rate_ema = 1.0 と同じ趣旨。
        # ema_alpha = 0.01 では最初の 100 iteration 程度は初期値が支配するので、
        # ここを 0 にすると実測が溜まる前に後退側へ倒れてしまう。
        state[command_name] = {
            "alpha": 0.0,
            "kick_rate_ema": 1.0,
            "apex_ema": apex_advance_above,
            "last_now": now,
        }
    gate = state[command_name]

    # -- 直近に終わったエピソード群のキック成立率を EMA に入れる
    #
    # ManagerBasedRLEnv._reset_idx は curriculum → command reset の順に呼ぶので、
    # ここで env_ids を見ると Metrics/<command>/kick_rate として記録されるのと同じ値を
    # ゼロ化前に読める (kick_rate_gated_speed_range と同じ読み方)。
    command_term = env.command_manager.get_term(command_name)
    have_episodes = env_ids is not None and len(env_ids) > 0
    metric = command_term.metrics.get("kick_rate", None)
    if metric is not None and have_episodes:
        kick_rate = float(metric[env_ids].mean())
        gate["kick_rate_ema"] += ema_alpha * (kick_rate - gate["kick_rate_ema"])

    # -- 第 2 の指標 (ロブなら kick_apex_height) も同じ読み方で EMA に入れる
    if apex_metric_name is not None:
        apex_metric = command_term.metrics.get(apex_metric_name, None)
        if apex_metric is None:
            raise KeyError(
                f"command '{command_name}' に metrics['{apex_metric_name}'] がありません。"
                " apex_metric_name の綴りか、その指標を出す設定 (KickDirectionCommandCfg) を"
                " 確認してください。"
            )
        if have_episodes:
            apex = float(apex_metric[env_ids].mean())
            gate["apex_ema"] += ema_alpha * (apex - gate["apex_ema"])

    # -- ゲートの開閉に応じて α を進める / 戻す
    elapsed = max(0.0, now - gate["last_now"])
    gate["last_now"] = now
    ema = gate["kick_rate_ema"]
    apex_ema = gate["apex_ema"]
    # apex_metric_name = None のときは第 2 条件を恒真 / 恒偽にして従来の挙動に落とす。
    apex_ok = True if apex_metric_name is None else apex_ema >= apex_advance_above
    apex_bad = False if apex_metric_name is None else apex_ema < apex_retreat_below
    if now >= start_step:
        delta = elapsed / (end_step - start_step)
        if ema >= advance_above and apex_ok:
            gate["alpha"] += delta
        elif ema < retreat_below or apex_bad:
            gate["alpha"] -= delta * retreat_scale
        # 前進条件を満たさず後退条件にも掛からない範囲は据え置き (ヒステリシス)。
        gate["alpha"] = min(max(gate["alpha"], 0.0), 1.0)

    if _GATE_DEBUG:
        gate["calls"] = gate.get("calls", 0) + 1
        if gate["calls"] <= 5 or gate["calls"] % 25 == 0:
            print(
                f"[expansion] calls={gate['calls']} now={now:.3f} elapsed={elapsed:.5f}"
                f" alpha={gate['alpha']:.6f} ema={ema:.4f} id(env)={id(env)}"
            )

    alpha = gate["alpha"]

    def _lerp(a: float, b: float) -> float:
        return a + (b - a) * alpha

    # -- 1+2. ボール出現の方位と距離 (EventManager は params を毎回読み直す)
    half_angle = _lerp(half_angle_range[0], half_angle_range[1])
    dist_min = _lerp(dist_range_start[0], dist_range_end[0])
    dist_max = _lerp(dist_range_start[1], dist_range_end[1])
    ball_event = env.event_manager.get_term_cfg(ball_event_name)
    ball_event.params["half_angle"] = half_angle
    ball_event.params["dist_range"] = (dist_min, dist_max)

    # -- 3. 蹴り方向の範囲 (KickDirectionCommand は resample のたびに cfg を読み直す)
    heading_half = _lerp(heading_halfwidth_range[0], heading_halfwidth_range[1])
    command_term.cfg.ranges.heading = (-heading_half, heading_half)

    # -- 4+5. approach_penalty / ball_avoidance のクロスフェード
    fade = 1.0 if approach_fade_iterations <= 0 else min(now / approach_fade_iterations, 1.0)
    approach_weight = approach_end_weight * fade * (1.0 - alpha)
    avoidance_weight = avoidance_end_weight * alpha

    approach_term = env.reward_manager.get_term_cfg(approach_term_name)
    if abs(approach_term.weight - approach_weight) > 1e-8:
        approach_term.weight = approach_weight
    avoidance_term = env.reward_manager.get_term_cfg(avoidance_term_name)
    if abs(avoidance_term.weight - avoidance_weight) > 1e-8:
        avoidance_term.weight = avoidance_weight

    logged = {
        "alpha": alpha,
        "kick_rate_ema": ema,
        "half_angle": half_angle,
        "dist_max": dist_max,
        "heading_half": heading_half,
        "approach_weight": approach_weight,
        "avoidance_weight": avoidance_weight,
    }
    # 第 2 指標を使う呼び出し元でだけ TB タグを 1 本増やす (使わない側のタグ集合は不変)。
    if apex_metric_name is not None:
        logged["apex_ema"] = apex_ema
    return logged
