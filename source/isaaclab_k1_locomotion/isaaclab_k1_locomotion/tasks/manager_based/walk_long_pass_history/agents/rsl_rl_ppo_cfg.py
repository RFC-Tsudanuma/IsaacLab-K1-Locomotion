# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_long_pass.agents.rsl_rl_ppo_cfg import K1WalkLongPassPPORunnerCfg


@configclass
class K1WalkLongPassHistoryPPORunnerCfg(K1WalkLongPassPPORunnerCfg):
    """ロングパス + 短期 I/O 履歴用。

    **ネットワーク構造と PPO 設定は long_pass と同一に保つ**。隠れ層・活性化・
    PPO 本体のハイパラは変えず、actor 入力幅だけを 55 → 223 に広げる。

    experiment_name だけ分けて、他のキック run とログが混ざらないようにする。

    観測スロットの意味が左足裏からボール位置へ変わるため、旧 checkpoint を
    拡張しても親との挙動一致は保証されない。ゲート付き球速カリキュラムの停止・後退と、
    最終帯到達後の仕上げを見込んで既定 iteration は 5000。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_history"
        self.max_iterations = 5000
