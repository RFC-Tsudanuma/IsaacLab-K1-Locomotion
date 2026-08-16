# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_middle_kick_dual の RunnerCfg。

policy の差し替えは :func:`~..walk_kick_dual.agents.rsl_rl_ppo_cfg._use_history_cnn_policy`
に集約されている (切り出し方 K = 5 / CNN の形も含めてあちらで 1 箇所管理)。
**全 stage で呼ぶこと**。1 段でも忘れると、そこで checkpoint の連鎖が切れる。

experiment_name は既存の 1 フレーム版 (``k1_walk_middle_kick`` /
``k1_walk_middle_kick_360`` / ``k1_walk_middle_kick_360_noisy_ball``) と別にしてあり、
ログが混ざらない。
"""

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...walk_kick_dual.agents.rsl_rl_ppo_cfg import _use_history_cnn_policy
from ...walk_middle_kick.agents.rsl_rl_ppo_cfg import (
    K1WalkMiddleKick360PPORunnerCfg,
    K1WalkMiddleKickPPORunnerCfg,
)


@configclass
class K1WalkMiddleKickDualPPORunnerCfg(K1WalkMiddleKickPPORunnerCfg):
    """Stage 2 (middle, dual)。

    引き継ぎ元は ``k1_walk_kick_dual_walk_phase`` (履歴入力版の walk phase)。
    同梱の 1 フレーム walk phase checkpoint から始めるなら
    ``--warm_start_from_single_frame`` を付けること。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_middle_kick_dual"
        _use_history_cnn_policy(self)


@configclass
class K1WalkMiddleKickDual360PPORunnerCfg(K1WalkMiddleKick360PPORunnerCfg):
    """Stage 3 (middle, dual, クリーン 360)。``k1_walk_middle_kick_dual`` から始める前提。

    最終段ではない (次が DR 段)。σ_direction のアニールがこの段で 1500 → 3000
    iteration を掛けて走るので、``--max_iterations`` は 3000 以上にすること。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_middle_kick_dual_360"
        _use_history_cnn_policy(self)


@configclass
class K1WalkMiddleKickDual360DRPPORunnerCfg(K1WalkMiddleKick360PPORunnerCfg):
    """Stage 4 (middle, dual, 観測 DR・最終)。``k1_walk_middle_kick_dual_360`` から始める前提。

    環境の差は観測 DR とランプの定数化だけ (観測の次元・並びは Stage 3 と同一) なので、
    ネットワークと PPO ハイパラは Stage 3 と完全に同じ。experiment_name だけ分ける。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_middle_kick_dual_360_dr"
        _use_history_cnn_policy(self)
