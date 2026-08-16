# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg


@configclass
class K1WalkKickBothFeetPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 2 (両足キック)。stage 1 の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは walk_kick 系と同一に保つ (観測 55 次元も同じ)。
    experiment_name だけ分けて、既存の walk_kick の run とログが混ざらないようにする。

    NOTE: ``symmetry_cfg`` は基底クラスのまま None。観測から左右非対称なスロット
          (左足裏) は消えたので mirror loss は原理的には定義できるようになったが、
          ``locomotion.mdp.symmetry._mirror_policy_obs`` は歩行タスクの 49 次元専用
          なので、有効化には 55 次元用の鏡像規約を書き下ろす必要がある。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_both_feet"


@configclass
class K1WalkKickBothFeetWalkPhasePPORunnerCfg(K1WalkKickBothFeetPPORunnerCfg):
    """Stage 1 (両足キック版の歩行のみ)。観測・ネットワークは stage 2 と同一に保つ。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_both_feet_walk_phase"
