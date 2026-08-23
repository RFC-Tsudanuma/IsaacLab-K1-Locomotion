# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass 系列の Stage 1-3 (履歴入力版)。

``scripts/rsl_rl/train_walk_long_pass.sh`` は歩行から通しで 4 段学習する::

    Stage 1 walk phase → Stage 2 loop_pass → Stage 3 loop_pass_360 → Stage 4 long_pass

Stage 4 (:class:`~.walk_long_pass_env_cfg.K1WalkLongPassEnvCfg`) の actor は
:data:`~.walk_long_pass_env_cfg._OBS_HISTORY_LENGTH` フレームの観測履歴を見るので、
**前段も同じ観測形でないと checkpoint が繋がらない**
(actor の入力次元と重みの名前が変わり、train.py が黙って捨てる)。

一方で Stage 1-3 の元タスク

* ``Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0``
* ``Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0``
* ``Isaac-Velocity-Flat-K1-Walk-Loop-Pass-360-v0``

は walk_kick / walk_kick_360 / walk_pass / walk_lob / walk_mid_kick / loop_shoot と
**共用**で、そこに履歴を足すと全ファミリーのパイプラインが道連れになり、コミット済みの
基準 checkpoint も actor が使えなくなる。そこで long_pass 系列専用のタスクを別 ID で
登録し、共用タスクは 1 フレーム観測のまま残す。

中身の差分は :func:`~.walk_long_pass_env_cfg.enable_obs_history` (観測を履歴にする) と
:func:`~.walk_long_pass_env_cfg.disable_landing_shaping` (着地 shaping 3 項を消す) の
2 箇所だけ。コマンド・シーン・カリキュラムは元タスクと完全に同一。
着地 shaping を消すのは Stage 4 と揃えるため — 段によって有無が変わると、前段が
獲得した歩容が次の段で罰せられることになる。experiment_name (= logs の出力先) は
``k1_walk_long_pass_*`` で分けてあり、既存の run と混ざらない。
"""

from isaaclab.utils import configclass

from ..walk_kick.walk_kick_env_cfg import K1WalkKickWalkPhaseEnvCfg, K1WalkKickWalkPhaseEnvCfg_PLAY
from ..walk_loop_pass.walk_loop_pass_env_cfg import (
    K1WalkLoopPass360EnvCfg,
    K1WalkLoopPass360EnvCfg_PLAY,
    K1WalkLoopPassEnvCfg,
    K1WalkLoopPassEnvCfg_PLAY,
)
from .walk_long_pass_env_cfg import (
    disable_landing_shaping,
    enable_obs_history,
    rebalance_gait_vs_kick,
)

# NOTE: replace_sole_pos_with_ball_pos / use_fixed_kick_foot は実機評価で不採用に
#       なったので呼んでいない (理由は walk_long_pass_env_cfg の __post_init__ の
#       コメント参照)。戻すときは Stage 4 と **必ず揃えて** 有効にすること。


@configclass
class K1WalkLongPassWalkPhaseEnvCfg(K1WalkKickWalkPhaseEnvCfg):
    """Stage 1 (歩行のみ) の履歴入力版。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)


@configclass
class K1WalkLongPassWalkPhaseEnvCfg_PLAY(K1WalkKickWalkPhaseEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)


@configclass
class K1WalkLongPassLoopPassEnvCfg(K1WalkLoopPassEnvCfg):
    """Stage 2 (限定レンジのループパス) の履歴入力版。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


@configclass
class K1WalkLongPassLoopPassEnvCfg_PLAY(K1WalkLoopPassEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


@configclass
class K1WalkLongPassLoop360EnvCfg(K1WalkLoopPass360EnvCfg):
    """Stage 3 (全方位ループパス) の履歴入力版。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


@configclass
class K1WalkLongPassLoop360EnvCfg_PLAY(K1WalkLoopPass360EnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)
