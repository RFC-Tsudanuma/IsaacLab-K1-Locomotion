# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg


@configclass
class K1WalkLoopPassPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """ループシュート用。ネットワーク・PPO ハイパラは Walk-Kick と同一に保つ。

    観測 55 次元も同じなので、walk phase の checkpoint をそのまま引き継げる
    (walk_pass と同じ理由。詳細は walk_loop_pass_env_cfg のモジュール docstring)。
    experiment_name だけ分けて、他のキック run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_loop_pass"


@configclass
class K1WalkLoopPass360PPORunnerCfg(K1WalkLoopPassPPORunnerCfg):
    """全方位版。loop_pass の checkpoint から --load_pretrained で始める前提。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_loop_pass_360"
