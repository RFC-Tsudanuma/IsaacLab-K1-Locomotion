# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""win_the_ball 第1段階の RSL-RL runner 設定。"""

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg


@configclass
class K1WinTheBallKickPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """前方習得後に側方・後方を解禁する統合タスクをfine-tuneする。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_win_the_ball_kick"
        self.max_iterations = 20000


@configclass
class K1WinTheBallDirectionalPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """側方通常キックと、正面姿勢を保つ低速ヒールキックを学ぶ第2段階。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_win_the_ball_directional"
        self.max_iterations = 5000