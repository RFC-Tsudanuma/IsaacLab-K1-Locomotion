# mdp/curriculums.py
from __future__ import annotations
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