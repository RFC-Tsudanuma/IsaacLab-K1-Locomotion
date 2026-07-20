# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg


@configclass
class K1WalkPassPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """パス（弱いキック）用。ネットワーク・PPO ハイパラは Walk-Kick と同一に保つ。

    観測 55 次元も同じなので、walk phase の checkpoint をそのまま引き継げる。
    experiment_name だけ分けて、強キックの run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_pass"
