# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_lob_plant の 3 段ぶんの RunnerCfg。

3 段とも **観測履歴 (100 フレーム) の actor** を使う
(:func:`~...walk_inside_kick.agents.rsl_rl_ppo_cfg._use_history_cnn_policy`)。
段ごとに入れたり外したりしないので、stage 1 → 2 → 3 の checkpoint は履歴 → 履歴で
そのまま繋がる。stage 1 だけ引き継ぎ元 (共用の歩行 checkpoint) が 1 フレーム観測なので
``--warm_start_from_single_frame`` が要る (通しスクリプトが自動で付ける)。

**mirror loss は入れない。** ``_use_history_cnn_policy`` を :mod:`..walk_inside_kick`
側から import しているのはそのため — :mod:`...walk_kick_dual.agents.rsl_rl_ppo_cfg` の
同名関数は ``PPOSparseMirror`` + ``_use_mirror_loss`` を掛ける版で、あちらの目的は
「両足で蹴れるようにする」こと。この系列は右足専用 (walk_lob 以来 ``kick_foot_right_frac``
は 1.0 付近に張り付く想定) なので、鏡像対称性を promote する損失は獲得済みの蹴り方を
壊す方向にしか働かない (:mod:`...walk_lob_rough.agents.rsl_rl_ppo_cfg` と同じ判断)。

ネットワーク幅・PPO ハイパラは Walk-Lob = Walk-Loop-Shoot = Walk-Kick と同一に保つ。
観測も 55 / 61 次元のままなので、インサイド系や walk_kick 系の checkpoint とも
(履歴の有無さえ合っていれば) 相互に載る。
"""

from isaaclab.utils import configclass

from ...walk_inside_kick.agents.rsl_rl_ppo_cfg import _use_history_cnn_policy
from ...walk_lob.agents.rsl_rl_ppo_cfg import K1WalkLobPPORunnerCfg

# --------------------------------------------------------------------------- #
# 1 iteration あたりの env step 数。
#
# 基底 :class:`~...locomotion.agents.rsl_rl_ppo_cfg.K1FlatPPORunnerCfg` の値そのもの
# (48) だが、**明示的に代入する**。env cfg 側の
# :data:`~..walk_lob_plant_env_cfg._SPI` はカリキュラムの時間換算にこの値を使うので、
# 基底が動くと「書いてある end_step が iteration にならない」という walk_kick 系の
# 既知バグ (_SPI = 24 なのに num_steps_per_env = 48) を作り直すことになる。
# ここで固定しておけば、基底が動いてもこの系列だけは食い違わない。
#
# **この値と walk_lob_plant_env_cfg._SPI は必ず一致させること。** 片方だけ変えると
# カリキュラムの窓が黙って倍/半分になる (通しスクリプトの ITER も一緒に見直すこと)。
# --------------------------------------------------------------------------- #
_NUM_STEPS_PER_ENV = 48


@configclass
class K1WalkLobPlantWalkPhasePPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 1 (歩行のみ・履歴 actor)。

    引き継ぎ元は共用の歩行 checkpoint
    ``logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt``。
    あちらは 1 フレーム観測なので ``--warm_start_from_single_frame`` が要る
    (付けないと actor が 1 本も引き継がれず train.py が止まる)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_plant_walk_phase"
        self.num_steps_per_env = _NUM_STEPS_PER_ENV
        _use_history_cnn_policy(self)


@configclass
class K1WalkLobPlantPPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 2 (平坦・ロブ本体)。``k1_walk_lob_plant_walk_phase`` から始める前提。

    履歴 → 履歴なので ``--warm_start_from_single_frame`` は不要。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_plant"
        self.num_steps_per_env = _NUM_STEPS_PER_ENV
        _use_history_cnn_policy(self)


@configclass
class K1WalkLobPlantRoughPPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 3 (凹凸 + ボール DR)。``k1_walk_lob_plant`` から始める前提。

    環境の差は地形と DR と「カリキュラムが固定済みであること」だけで、観測の次元・
    並びは stage 2 と同一。ネットワークと PPO ハイパラも stage 2 と完全に同じで、
    experiment_name だけ分けてログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_plant_rough"
        self.num_steps_per_env = _NUM_STEPS_PER_ENV
        _use_history_cnn_policy(self)


@configclass
class K1WalkLobPlant360PPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 2b (平坦・全方位)。``k1_walk_lob_plant`` から始める前提。

    環境の差は「拡大ゲートが 1 本生きていること」と 15 秒エピソードだけで、観測の
    次元・並びは stage 2 と同一。ネットワークと PPO ハイパラも stage 2 と完全に
    同じで、experiment_name だけ分けてログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_plant_360"
        self.num_steps_per_env = _NUM_STEPS_PER_ENV
        _use_history_cnn_policy(self)


@configclass
class K1WalkLobPlant360RoughPPORunnerCfg(K1WalkLobPPORunnerCfg):
    """Stage 3b (凹凸 + ボール DR + fewa 方式の観測ノイズ)。

    ``k1_walk_lob_plant_360`` から始める前提。観測は **次元・並びとも stage 2b と
    同一** (差し替えたのは各項の func / params / noise だけ) なので、checkpoint は
    そのまま載る。ネットワークと PPO ハイパラも同じ。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_lob_plant_360_rough"
        self.num_steps_per_env = _NUM_STEPS_PER_ENV
        _use_history_cnn_policy(self)
