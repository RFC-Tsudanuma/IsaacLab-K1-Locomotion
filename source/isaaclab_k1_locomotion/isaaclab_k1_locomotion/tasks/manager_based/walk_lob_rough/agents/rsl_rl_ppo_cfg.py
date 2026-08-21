# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_lob_rough の RunnerCfg (actor を ActorCriticHistoryCNN に差し替える)。

**6 つの RunnerCfg は experiment_name 以外すべて同一。** ネットワークも PPO ハイパラも
共通なので、``k1_walk_lob_hist_walk_phase`` → ``k1_walk_lob_hist_kick`` →
``k1_walk_lob_hist`` → (凹凸へ) と ``--load_pretrained`` でそのまま繋がる。
継承元を段ごとに変えていないのはそのため (walk_kick / loop_shoot の RunnerCfg も
experiment_name しか違わないので、どれを継承しても同じ値になる)。

dual 系 (:mod:`...walk_kick_dual.agents.rsl_rl_ppo_cfg`) との違いは 2 点:

1. **mirror loss を入れない。** この系列は両足キック化をしないので、観測に
   左右非対称な項が残る前提。``locomotion.mdp.symmetry`` の鏡像規約はこの観測
   構成に対して定義できない。
2. **``PPOSparseMirror`` に差し替えない。** 1 の帰結。mirror を間引く実装なので、
   mirror を使わないこちらでは素の PPO のままでよい。

``num_mini_batches`` も継承元の 4 のまま。dual が 8 に割っているのは履歴観測 ×
mirror loss で観測バッチを 2 倍 clone するのが原因で、mirror が無いこちらでは
ピークがおよそ半分になる。**それでも OOM する場合は
``self.algorithm.num_mini_batches = 8`` を足すこと** (総バッチ量は変わらないので
学習への影響がいちばん小さい OOM 対策)。

CNN の切り出し方 (直近フレーム数・カーネル/フィルタ/ストライド) は dual と同じ値を
import して使う。履歴長 H = 100 は環境側
(:data:`...walk_kick_dual.walk_kick_dual_env_cfg._OBS_HISTORY_LENGTH`) が決め、
ネットワークは観測の形 (N, H, D) から H を読む。
"""

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...locomotion.networks import RslRlPpoActorCriticHistoryCnnCfg
from ...walk_kick_dual.agents.rsl_rl_ppo_cfg import (
    _CNN_FILTERS,
    _CNN_KERNEL_SIZES,
    _CNN_STRIDES,
    _NUM_RECENT_FRAMES,
)
from ...walk_lob.agents.rsl_rl_ppo_cfg import K1WalkLobPPORunnerCfg


def _use_history_cnn_policy(cfg) -> None:
    """policy を :class:`~...locomotion.networks.ActorCriticHistoryCNN` に差し替える。

    PPO ハイパラ・MLP 幅・正規化の有無は継承元の値をそのまま引き継ぐ。
    dual 版と違い ``num_mini_batches`` と ``class_name`` は触らない
    (モジュール docstring 参照)。

    **全段で必ず呼ぶこと。** 1 段でも呼び忘れると actor だけ形が違うので
    checkpoint の連鎖がそこで切れる。
    """
    base = cfg.policy
    cfg.policy = RslRlPpoActorCriticHistoryCnnCfg(
        init_noise_std=base.init_noise_std,
        noise_std_type=base.noise_std_type,
        actor_obs_normalization=base.actor_obs_normalization,
        critic_obs_normalization=base.critic_obs_normalization,
        actor_hidden_dims=base.actor_hidden_dims,
        critic_hidden_dims=base.critic_hidden_dims,
        activation=base.activation,
        num_recent_frames=_NUM_RECENT_FRAMES,
        cnn_kernel_sizes=_CNN_KERNEL_SIZES,
        cnn_filters=_CNN_FILTERS,
        cnn_strides=_CNN_STRIDES,
    )


@configclass
class _K1WalkLobHistPPORunnerCfgBase(K1WalkLobPPORunnerCfg):
    """6 段共通の土台。派生は ``experiment_name`` を設定するだけ。"""

    def __post_init__(self):
        super().__post_init__()
        _use_history_cnn_policy(self)


# --------------------------------------------------------------------------- #
# 平坦 3 段 (まずこちらを通す)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobHistWalkPhasePPORunnerCfg(_K1WalkLobHistPPORunnerCfgBase):
    """Stage 1 (平坦・歩行のみ)。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_walk_lob_hist_walk_phase"


@configclass
class K1WalkLobHistKickPPORunnerCfg(_K1WalkLobHistPPORunnerCfgBase):
    """Stage 2 (平坦・キック)。引き継ぎ元は ``k1_walk_lob_hist_walk_phase``。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_walk_lob_hist_kick"


@configclass
class K1WalkLobHistPPORunnerCfg(_K1WalkLobHistPPORunnerCfgBase):
    """Stage 3 (平坦・ロブ)。引き継ぎ元は ``k1_walk_lob_hist_kick``。

    .. warning::
       **Stage 1 から直接は繋がない。** ロブの報酬集合は歩行ポリシーから
       ブートストラップできない (env cfg のモジュール docstring 参照)。
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_walk_lob_hist"


# --------------------------------------------------------------------------- #
# 凹凸 3 段
#
# ``k1_walk_lob_rough_walk_phase`` の名前は 2026-08-17 の学習済み run
# (8000 iteration、健全) を再利用できるよう据え置いてある。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobRoughWalkPhasePPORunnerCfg(_K1WalkLobHistPPORunnerCfgBase):
    """Stage 1 (凹凸・歩行のみ)。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_walk_lob_rough_walk_phase"


@configclass
class K1WalkLobRoughKickPPORunnerCfg(_K1WalkLobHistPPORunnerCfgBase):
    """Stage 2 (凹凸・キック)。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_walk_lob_rough_kick"


@configclass
class K1WalkLobRoughPPORunnerCfg(_K1WalkLobHistPPORunnerCfgBase):
    """Stage 3 (凹凸・ロブ)。平坦 3 段の checkpoint から fine-tune する想定。"""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_walk_lob_rough"
