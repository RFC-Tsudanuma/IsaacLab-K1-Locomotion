# mdp/curriculums.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

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