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
    アクチュエータは walk_kick / walk_loop_* と共通の素の DelayedPDActuator に戻した
    (T-N カーブは学習が進まなくなったため撤回)。物理が同一なので Walk-Loop-Shoot の
    checkpoint から始めても問題ない (そちらは既に「浮かせる」挙動を持っているぶん有利)。
    experiment_name だけ分けて、Walk-Loop-Shoot の run とログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob"


@configclass
class K1WalkLobWalkPhasePPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 1 (歩行のみ) 用。観測・ネットワークは stage 2 と同一に保つ。

    アクチュエータは walk_kick 系の walk phase と同一 (T-N カーブは撤回済み)。
    experiment_name だけ分けてログを混ぜないようにしているだけなので、
    ``k1_walk_kick_walk_phase`` の checkpoint を代わりに使っても構わない。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_walk_phase"
