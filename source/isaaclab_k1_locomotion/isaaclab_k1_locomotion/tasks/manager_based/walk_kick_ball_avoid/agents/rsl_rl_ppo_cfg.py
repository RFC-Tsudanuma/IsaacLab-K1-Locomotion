# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick_ball_avoid の RunnerCfg。

ネットワーク・PPO ハイパラは walk_kick 系と完全に同一に保つ (観測も 55 次元 /
critic 61 次元のまま)。``experiment_name`` だけ分けて、既存の walk_kick の run と
ログが混ざらないようにする。

mirror loss は **入れない** (継承元 :class:`~...walk_kick.agents.rsl_rl_ppo_cfg.K1WalkKickPPORunnerCfg`
が ``symmetry_cfg = None`` にしているのをそのまま使う)。観測から左足裏スロットが
消えたので 55 次元用の鏡像規約 (:func:`~...walk_kick_both_feet.symmetry.compute_symmetric_states`)
は原理的には適用できるが、このタスクの主題は Ball Avoidance の解釈の検証なので、
差分を 2 点に絞るために入れていない。
"""

from isaaclab.utils import configclass

from ...walk_kick.agents.rsl_rl_ppo_cfg import (
    K1WalkKickPPORunnerCfg,
    K1WalkKickWalkPhasePPORunnerCfg,
)


@configclass
class K1WalkKickBallAvoidPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 2 (キック)。stage 1 の checkpoint から --load_pretrained で始める前提。

    .. note::
        既存の ``k1_walk_kick`` の checkpoint は流用しないこと。次元は同じでも
        観測スロット 3 の意味が違う (左足裏 → ボール 3D 位置)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_ball_avoid"
        self.max_iterations = 20000


@configclass
class K1WalkKickBallAvoidWalkPhasePPORunnerCfg(K1WalkKickWalkPhasePPORunnerCfg):
    """Stage 1 (歩行のみ)。観測・ネットワークは stage 2 と同一に保つ。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_ball_avoid_walk_phase"
        self.max_iterations = 20000
