# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKick360PPORunnerCfg, K1WalkKickPPORunnerCfg


@configclass
class K1WalkKickWeakPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 2 (weak)。walk phase の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは Walk-Kick 系と同一に保つ (観測 55 次元も同じ)。
    experiment_name だけ分けて、既存の walk_kick の run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_weak"


@configclass
class K1WalkKick360WeakPPORunnerCfg(K1WalkKick360PPORunnerCfg):
    """Stage 3 (weak)。k1_walk_kick_weak の checkpoint から始める前提。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_360_weak"


@configclass
class K1WalkKick360WeakNoisyBallPPORunnerCfg(K1WalkKick360WeakPPORunnerCfg):
    """Stage 4 (weak, 知覚ノイズ)。k1_walk_kick_360_weak の checkpoint から始める前提。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_360_weak_noisy_ball"


@configclass
class K1WalkKick360WeakNoisyBallWalkInitPPORunnerCfg(K1WalkKick360WeakNoisyBallPPORunnerCfg):
    """Stage 5 (weak, 知覚ノイズ, 歩行状態からの reset)。

    experiment_name を分けて、立ち姿勢リセットの run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_360_weak_noisy_ball_walk_init"
