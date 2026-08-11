# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_flag 用の観測関数。

2 種類ある:

* :func:`prev_action_of` … 行動空間が 13 次元になったので、``last_action`` (行動全体)
  を項ごとのスライスに分ける。既存の 55 次元の並びを 1 bit も動かさないための措置。
* :func:`kick_latch_state` / :func:`kick_frozen_values` … critic 専用の特権情報。
  post-latch の報酬が「latch 時の凍結値 × 残り窓ステップ数」で決まりきっているのに、
  critic はそのどちらも見えていなかった。埋めるのがこの 2 つ。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from ...walk_kick.mdp.kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------- #
# 行動スライス
# --------------------------------------------------------------------------- #
_SLICE_ATTR = "_action_term_slices"


def _term_slice(env: ManagerBasedRLEnv, action_name: str) -> slice:
    """行動ベクトル中でその項が占める範囲。一度だけ解決してキャッシュする。"""
    cache = getattr(env, _SLICE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, _SLICE_ATTR, cache)
    s = cache.get(action_name)
    if s is None:
        names = env.action_manager.active_terms
        dims = env.action_manager.action_term_dim
        idx = names.index(action_name)
        start = sum(dims[:idx])
        s = slice(start, start + dims[idx])
        cache[action_name] = s
    return s


def prev_action_of(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """前ステップの行動のうち、指定した項の分だけ。shape: (N, dim(action_name))

    IsaacLab の ``mdp.last_action(env, action_name=...)`` を使わない理由:
    あちらは ``ActionTerm.raw_actions`` を返すが、``ActionManager.reset()`` は自身の
    ``_action`` バッファしかゼロにせず各項の ``raw_actions`` は触らない。
    ``JointPositionAction`` は ``reset()`` を実装していないので、``action_name`` 指定に
    切り替えるとリセット直後の 1 ステップだけ **前エピソードの行動が観測に漏れる**
    (引数なしの ``last_action`` = ``action_manager.action`` にはこの漏れが無い)。

    行動空間を 13 次元に増やしても既存 55 次元の中身を 1 bit も変えないことが
    このタスクの前提 (checkpoint をパディングで引き継ぐため) なので、
    ゼロクリアされる ``action_manager.action`` を項ごとに切り出す方式にしている。
    """
    return env.action_manager.action[:, _term_slice(env, action_name)]


# --------------------------------------------------------------------------- #
# critic 専用の特権情報
#
# リセット直後の扱いについて:
#   ManagerBasedRLEnv.step() の順序は Termination → Reward → _reset_idx → Observation。
#   kick_state は episode_length_buf == 1 で遅延リセットする設計なので、_reset_idx 直後
#   (episode_length_buf == 0) に観測を取ると **前エピソードの latch 状態が残っている**。
#   同一ステップ内はキャッシュが効いて再計算も走らない。そこで新エピソード 1 本目の
#   観測だけ明示的にゼロで潰す (walk_kick.mdp.observations.prev_ball_pos_b と同じ対処)。
# --------------------------------------------------------------------------- #


def _fresh_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """リセット直後 (episode_length_buf == 0) の env を 0 にするマスク。shape: (N, 1)"""
    return (env.episode_length_buf > 0).float().unsqueeze(-1)


def kick_latch_state(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    delay_steps: int,
) -> torch.Tensor:
    """latch 状態。shape: (N, 2)

    * ``[:, 0]`` = ``kick_done`` (0/1)。値 latch が発火したか。
    * ``[:, 1]`` = 猶予窓の進捗 (0→1)。latch から ``delay_steps`` 経つと 1 になり、
      そこで ``kick_finished`` がエピソードを終わらせる。

    この 2 つが必要な理由は :func:`kick_frozen_values` の docstring 参照。
    ``kick_done`` は「凍結値がまだ全部 0 なだけ」と「本当に τ=0 で蹴れた」を
    critic が区別するためのゲートでもある。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    kick_done = state["kick_done"].float()

    # _kick_done_counter は terminations.kick_finished が作る。その項が無い構成
    # (walk phase など) では存在しないのでゼロ扱いにする。
    counter = getattr(env, "_kick_done_counter", None)
    if counter is None:
        progress = torch.zeros_like(kick_done)
    else:
        progress = (counter.float() / max(int(delay_steps), 1)).clamp(max=1.0)

    return torch.stack([kick_done, progress], dim=-1) * _fresh_mask(env)


def kick_frozen_values(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    v_scale: float,
) -> torch.Tensor:
    """latch 時にスナップショットされた値。shape: (N, 5)

    並びは ``[τ_direction, v_ball, v_ball_3d, φ, p_style]``。

    なぜ critic に要るか
    --------------------
    post-latch の報酬は latch 時の凍結値だけの関数で、窓が閉じるまで 1 ステップも
    変化しない (walk_kick.mdp.rewards 参照)::

        r_dir = clamp((exp(-τ²/2σ²) - 0.5) * 2 * p_style, min=0)
        項1 = w₁·r_dir
        項2 = w₂·r_dir·exp(-((v_3d - v_target)/σ_v)²)
        項7 = w₇·r_dir·f(φ)

    つまり残り窓が n ステップなら、post-latch の残リターンは
    ``(項1 + 項2 + 項7) × dt × n`` という決定論的な定数。critic にこれらを渡せば
    その部分を誤差ゼロで当てられる。逆に渡さないと post-latch の 100 ステップで
    大きな TD 誤差が出続け、本当に学習させたい転倒罰 (-100) がその予測誤差ノイズに
    埋もれる。猶予窓 2 秒を「蹴った直後に転ぶのを罰する」ために取っている設計
    (walk_kick_env_cfg の _KICK_WINDOW_S) が効くようになるのがここ。

    ``v_target`` は既に policy 観測 (``target_kick_velocity``) に入っているので不要。

    スケールについて
    ----------------
    EmpiricalNormalization は ``rate = batch / count`` で更新するが、引き継ぐ
    checkpoint の count は既に 1.5e9 に達しており、パディングで足した次元の統計は
    実質更新されない (mean=0 / var=1 のまま固定)。そのため **観測側で O(1) に
    正規化してから渡す**。

    Args:
        v_scale: ボール速度の正規化に使う代表速度 [m/s]。**目標ボール速度の帯の上限を
            渡すこと** (env cfg 側が ``target_speed_range[1]`` から取る)。帯を張り替えた
            のにここが古い値のままだと、速度の観測レンジだけがずれて critic が
            読み違える。τ を割る π と違って物理的な定数ではないので、値を直接書かない。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    out = torch.stack(
        [
            state["tau_direction_frozen"] / math.pi,  # 0..1 (τ は定義上 0..π)
            state["v_ball_frozen"] / v_scale,         # 0..1 程度
            state["v_ball_3d_frozen"] / v_scale,
            state["phi_frozen"],                      # rad, 概ね -0.5..1.0 なので素通し
            state["p_style_frozen"],                  # 0..1
        ],
        dim=-1,
    )
    return out * _fresh_mask(env)
