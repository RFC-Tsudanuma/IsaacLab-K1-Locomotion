# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import K1FlatPPORunnerCfg


@configclass
class K1KickPPORunnerCfg(K1FlatPPORunnerCfg):
    resume_experiment_name = "k1_flat"

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_kick"
        self.max_iterations = 3000
        self.save_interval = 100
        self.algorithm.entropy_coef = 0.005
