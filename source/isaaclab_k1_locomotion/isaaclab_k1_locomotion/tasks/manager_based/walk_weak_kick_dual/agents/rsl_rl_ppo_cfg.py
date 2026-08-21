# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_weak_kick_dual の RunnerCfg。

policy の差し替えは :func:`~..walk_kick_dual.agents.rsl_rl_ppo_cfg._use_history_cnn_policy`
に集約されている (切り出し方 K = 5 / CNN の形も含めてあちらで 1 箇所管理)。
**全 stage で呼ぶこと**。1 段でも忘れると、そこで checkpoint の連鎖が切れる。

左右対称性の mirror loss
(:func:`~..walk_kick_both_feet.agents.rsl_rl_ppo_cfg._use_mirror_loss`) も
**全 stage で呼ぶこと**。こちらは形が変わらないぶん忘れても起動ログに何も出ず、
その段だけ対称化の圧が消えて蹴り足が片方へ戻る (``kick_foot_right_frac`` が
0/1 に張り付いてから気づくことになる)。

experiment_name は既存の 1 フレーム版 (``k1_walk_kick_weak`` /
``k1_walk_kick_360_weak`` / ``k1_walk_kick_360_weak_noisy_ball``) と別にしてあり、
ログが混ざらない。
"""

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...walk_kick_both_feet.agents.rsl_rl_ppo_cfg import _use_mirror_loss
from ...walk_kick_dual.agents.rsl_rl_ppo_cfg import _use_history_cnn_policy
from ...walk_weak_kick.agents.rsl_rl_ppo_cfg import (
    K1WalkKick360WeakPPORunnerCfg,
    K1WalkKickWeakPPORunnerCfg,
)


@configclass
class K1WalkKickDualWeakPPORunnerCfg(K1WalkKickWeakPPORunnerCfg):
    """Stage 2 (weak, dual)。

    引き継ぎ元は ``k1_walk_kick_dual_walk_phase`` (履歴入力版の walk phase)。
    同梱の 1 フレーム walk phase checkpoint から始めるなら
    ``--warm_start_from_single_frame`` を付けること。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual_weak"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)


@configclass
class K1WalkKickDual360WeakPPORunnerCfg(K1WalkKick360WeakPPORunnerCfg):
    """Stage 3 (weak, dual, クリーン 360)。``k1_walk_kick_dual_weak`` から始める前提。

    最終段ではない (次が DR 段)。σ_direction のアニールがこの段で 1500 → 3000
    iteration を掛けて走るので、``--max_iterations`` は 3000 以上にすること。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual_360_weak"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)


@configclass
class K1WalkKickDual360WeakDRPPORunnerCfg(K1WalkKick360WeakPPORunnerCfg):
    """Stage 4 (weak, dual, 観測 DR・最終)。``k1_walk_kick_dual_360_weak`` から始める前提。

    環境の差は観測 DR とランプの定数化だけ (観測の次元・並びは Stage 3 と同一) なので、
    ネットワークと PPO ハイパラは Stage 3 と完全に同じ。experiment_name だけ分ける。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual_360_weak_dr"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)
