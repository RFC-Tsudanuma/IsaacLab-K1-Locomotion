# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKick360PPORunnerCfg, K1WalkKickPPORunnerCfg


@configclass
class K1WalkMiddleKickPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 2 (middle)。walk phase の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは Walk-Kick 系と同一に保つ (観測 55 次元も同じ)。
    experiment_name だけ分けて、既存の walk_kick / walk_kick_weak の run と
    ログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_middle_kick"


@configclass
class K1WalkMiddleKick360PPORunnerCfg(K1WalkKick360PPORunnerCfg):
    """Stage 3 (middle)。k1_walk_middle_kick の checkpoint から始める前提。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_middle_kick_360"


@configclass
class K1WalkMiddleKick360NoisyBallPPORunnerCfg(K1WalkMiddleKick360PPORunnerCfg):
    """Stage 4 (middle, 知覚ノイズ)。k1_walk_middle_kick_360 の checkpoint から始める前提。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_middle_kick_360_noisy_ball"
