# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...walk_long_pass.agents.rsl_rl_ppo_cfg import K1WalkLongPassPPORunnerCfg


@configclass
class K1WalkLongPassHistoryPPORunnerCfg(K1WalkLongPassPPORunnerCfg):
    """ロングパス + 短期 I/O 履歴用。

    **ネットワーク構造は long_pass と同一に保つ** (これがこのタスクの前提)。隠れ層も
    活性化も PPO ハイパラも触らない。変わるのは actor 入力層の幅 55 → 223 だけで、
    それは cfg ではなく観測側が決める。

    experiment_name だけ分けて、他のキック run とログが混ざらないようにする。

    ITER は控えめでよい。報酬もコマンド分布も行動空間も変えていないので、増えた仕事は
    「増えた入力を使うかどうか」だけ。拡張直後は過去フレームの重みが 0 なので親と
    完全に同じ挙動から始まり、履歴が効くなら 1000-2000 iteration で
    kick_vel_ratio / kick_dir_error が動き出すはず。既定は余裕を見て 3000。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_history"
        self.max_iterations = 3000
