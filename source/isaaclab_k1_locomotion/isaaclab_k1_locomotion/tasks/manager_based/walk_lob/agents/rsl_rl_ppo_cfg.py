# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_loop_shoot.agents.rsl_rl_ppo_cfg import K1WalkLoopShootPPORunnerCfg


@configclass
class K1WalkLobPPORunnerCfg(K1WalkLoopShootPPORunnerCfg):
    """ロブキック用。ネットワーク・PPO ハイパラは Walk-Kick / Walk-Loop-Shoot と同一に保つ。

    観測 55 次元も同じなので、walk phase の checkpoint をそのまま引き継げる。
    Walk-Loop-Shoot の checkpoint から始めることもできる（そちらは既に「浮かせる」
    挙動を持っているぶん有利だが、アクチュエータの T-N カーブ導入後は旧 checkpoint の
    歩容が前提と合わないので、原則 walk phase から通しで回すこと）。
    experiment_name だけ分けて、Walk-Loop-Shoot の run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob"


@configclass
class K1WalkLobWalkPhasePPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 1 (歩行のみ) 用。観測・ネットワークは stage 2 と同一に保つ。

    walk_kick 系の walk phase とは **アクチュエータが違う** (T-N カーブ付き) ので、
    checkpoint も別管理にする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_walk_phase"
