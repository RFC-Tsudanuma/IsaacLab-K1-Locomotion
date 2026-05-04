# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...locomotion.agents.rsl_rl_ppo_cfg import K1FlatPPORunnerCfg


@configclass
class K1WalkKickPPORunnerCfg(K1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick"
