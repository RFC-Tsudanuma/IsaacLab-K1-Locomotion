# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg


@configclass
class K1WalkInsideKickPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """右足インサイドキック。walk phase の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは Walk-Kick 系と同一に保つ (観測 55 次元・履歴なしも同じ)。
    experiment_name だけ分けて、既存の walk_kick / walk_middle_kick 系の run と
    ログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_inside_kick"


@configclass
class K1WalkInsideKickCleanPPORunnerCfg(K1WalkInsideKickPPORunnerCfg):
    """フォールバック (ボール観測ノイズ無し) 用。**通常は使わない。**

    本命と混ざると比較にならないので experiment_name を分ける。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_inside_kick_clean"
