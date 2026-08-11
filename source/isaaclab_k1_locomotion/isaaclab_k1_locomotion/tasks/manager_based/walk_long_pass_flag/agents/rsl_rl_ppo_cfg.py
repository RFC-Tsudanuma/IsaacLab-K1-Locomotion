# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_long_pass.agents.rsl_rl_ppo_cfg import K1WalkLongPassPPORunnerCfg


@configclass
class K1WalkLongPassFlagPPORunnerCfg(K1WalkLongPassPPORunnerCfg):
    """ロングパス + キック検出フラグ用。

    ネットワーク・PPO ハイパラは long_pass と同一に保つ。次元だけが 55/12 → 56/13
    (critic は 61 → 69) に増える。

    ITER は控えめでよい。蹴り方の報酬もコマンド分布も変えていないので、増えた仕事は
    「latch を 1 bit で報告する」だけ。1000-2000 iteration で
    ``flag_pred_final`` が ``kick_rate`` に追いつくはず。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_flag"
        self.max_iterations = 5000
