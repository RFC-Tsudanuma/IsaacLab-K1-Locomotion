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
        self.max_iterations = 20000

        # 左右対称性 (mirror loss) は walk_kick では使えないので無効化する。
        # locomotion.mdp.symmetry の _mirror_policy_obs は歩行タスクの 49 次元観測
        # 専用で、走らせると次元不一致で ValueError になる。加えて walk_kick の観測は
        # 左足裏の位置だけを含む (対応する右足裏が観測に無い) ため、矢状面での左右反転が
        # そもそも定義できない。キック足を固定する論文の設定とも整合する。
        self.algorithm.symmetry_cfg = None


@configclass
class K1WalkKickWalkPhasePPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 1 (歩行のみ) 用。観測・ネットワークは stage 2 と同一に保つ。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_walk_phase"
        self.max_iterations = 20000


@configclass
class K1WalkKick360PPORunnerCfg(K1WalkKickPPORunnerCfg):
    """全方位版。walk_kick の checkpoint から --load_pretrained で始める前提。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_360"
