# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_loop_pass.agents.rsl_rl_ppo_cfg import K1WalkLoopPass360PPORunnerCfg


@configclass
class K1WalkLongPassPPORunnerCfg(K1WalkLoopPass360PPORunnerCfg):
    """ロングパス用。loop_pass_360 の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは Walk-Kick 系と同一に保つ (観測 55 次元も同じ)。
    experiment_name だけ分けて、他のキック run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass"
