# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""キック検出フラグの報酬。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ...walk_kick.mdp.kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def kick_flag(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    action_name: str = "kick_flag",
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """フラグ出力と正解ラベル ``kick_done`` の二乗誤差 (負値)。shape: (N,)

    ``r = -(sigmoid(a) - kick_done)²``。完全に当てて 0、完全に外して -1。

    なぜ BCE ではなく二乗誤差か
    ---------------------------
    BCE は自信満々に外すと -∞ に発散する。報酬として払うと value target が壊れて
    PPO が不安定になるので、有界な二乗誤差 (Brier score) を使う。
    rsl_rl 側に補助損失として足すなら BCE の方が良いが、それはネットワーク改造が要る。

    ``pos_weight`` (クラス不均衡の補正)
    ------------------------------------
    正例 (post-latch) は猶予窓の 100 ステップだけ、負例 (pre-latch) はそれ以前の全部。
    補正無しだと無情報なポリシーの最適解が 0.5 からずれ、片側に偏った出発点になる。
    正例側の誤差だけ ``pos_weight`` 倍して ``n_neg / n_pos`` に釣り合わせると、
    無情報時の最適出力がちょうど 0.5 (= sigmoid の勾配が最大の点) に来る。

    既定は 1.0 (無補正)。**実際の値は env cfg 側が実測エピソード長から決める**
    (``_FLAG_POS_WEIGHT``)。エピソード長はタスクごとに違うので、ここに置くと必ず
    古くなる。

    メトリクス ``flag_pred_final`` が ``kick_rate`` を大きく下回ったまま張り付いたら
    この値を上げる。逆に ``flag_pre_latch_pred`` (誤検出) が上がるようなら下げる。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    # RewardManager は行動適用後・_reset_idx 前に走るので、raw_actions は今ステップの値。
    p = torch.sigmoid(env.action_manager.get_term(action_name).raw_actions[:, 0])
    y = state["kick_done"].float()

    w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
    return -((p - y) ** 2) * w
