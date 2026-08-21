# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ...walk_long_pass.agents.rsl_rl_ppo_cfg import K1WalkLongPassPPORunnerCfg
from ..symmetry import compute_symmetric_states

_MIRROR_LOSS_COEFF = 0.5


@configclass
class K1WalkLongPassHistoryPPORunnerCfg(K1WalkLongPassPPORunnerCfg):
    """ロングパス + 短期 I/O 履歴用。

    **ネットワーク構造は long_pass と同一に保つ**。隠れ層・活性化・
    PPO 本体のハイパラは変えない。変更点は actor 入力幅 55 → 223 と、
    係数 0.5 の mirror loss だけ。data augmentation は使わない。

    experiment_name だけ分けて、他のキック run とログが混ざらないようにする。

    観測スロットの意味が左足裏からボール位置へ変わるため、旧 checkpoint を
    拡張しても親との挙動一致は保証されない。ゲート付き球速カリキュラムの停止・後退と、
    最終帯到達後の仕上げを見込んで既定 iteration は 5000。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_history"
        self.max_iterations = 5000
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=_MIRROR_LOSS_COEFF,
        )
