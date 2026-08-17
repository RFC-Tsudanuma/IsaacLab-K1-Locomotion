# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick_dual の RunnerCfg (actor を ActorCriticHistoryCNN に差し替える)。

``fewa/walk_kick_dual_encoder_tune`` の walk_long_pass/agents/rsl_rl_ppo_cfg.py からの移植。
:func:`_use_history_cnn_policy` は :mod:`..walk_weak_kick_dual` と
:mod:`..walk_middle_kick_dual` からも import されるので、切り出し方の定数は
ここ 1 箇所で管理する。

mirror loss (:func:`~...walk_kick_both_feet.agents.rsl_rl_ppo_cfg._use_mirror_loss`) も
**全 stage で呼ぶ**。both_feet 側で 1 箇所管理しており、dual 3 ファミリーはそれを
そのまま使う (観測は both_feet と同じ 55/58 次元で、履歴になっても最終軸は変わらない)。
"""

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...locomotion.networks import RslRlPpoActorCriticHistoryCnnCfg
from ...walk_kick.agents.rsl_rl_ppo_cfg import (
    K1WalkKick360PPORunnerCfg,
    K1WalkKickPPORunnerCfg,
    K1WalkKickWalkPhasePPORunnerCfg,
)
from ...walk_kick_both_feet.agents.rsl_rl_ppo_cfg import _use_mirror_loss

# --------------------------------------------------------------------------- #
# Actor に「そのまま」入れる直近フレーム数 K
#
# 履歴長 H = 100 は環境側 (walk_kick_dual_env_cfg._OBS_HISTORY_LENGTH) が決め、
# ネットワークは観測の形 (N, H, D) から H を読む。ここで持つのは切り出し方だけ。
# --------------------------------------------------------------------------- #
_NUM_RECENT_FRAMES = 5

# --------------------------------------------------------------------------- #
# 履歴符号化 CNN: 1 次元・隠れ層 2 つ
#
# [kernel size, filter size, stride size] = [6, 32, 3] と [4, 16, 2]。
# 系列長は 100 → 32 → 15 と縮み、潜在は 16 * 15 = 240 次元。
# actor MLP の入力は 5*55 + 240 = 515 次元になる。
# --------------------------------------------------------------------------- #
_CNN_KERNEL_SIZES = [6, 4]
_CNN_FILTERS = [32, 16]
_CNN_STRIDES = [3, 2]


def _use_history_cnn_policy(cfg) -> None:
    """policy を :class:`~...locomotion.networks.ActorCriticHistoryCNN` に差し替える。

    PPO ハイパラ・MLP 幅・正規化の有無は継承元の値をそのまま引き継ぐ。

    dual 系列の **全 stage** がこれを呼ぶ (walk_kick_dual / walk_weak_kick_dual /
    walk_middle_kick_dual)。**1 段でも呼び忘れると、そこで checkpoint の連鎖が
    切れる** (actor だけ形が違うので train.py に黙って捨てられ、起動ログを読まない
    限り気づけない。現在は actor が 0 本なら止まるようにしてある)。
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
class K1WalkKickDualWalkPhasePPORunnerCfg(K1WalkKickWalkPhasePPORunnerCfg):
    """Stage 1 (歩行のみ) の dual encoder 版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual_walk_phase"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)


@configclass
class K1WalkKickDualPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 2 (限定レンジのキック) の dual encoder 版。

    .. note::
        引き継ぎ元は **履歴入力版の Stage 1** (``k1_walk_kick_dual_walk_phase``)。
        共用タスクの 1 フレーム checkpoint から始めるなら
        ``--warm_start_from_single_frame`` が必要 (付けないと actor が 1 本も
        引き継がれず、train.py が止まる)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)


@configclass
class K1WalkKickDual360PPORunnerCfg(K1WalkKick360PPORunnerCfg):
    """Stage 3 (全方位・クリーン)。``k1_walk_kick_dual`` の checkpoint から始める前提。

    最終段ではない (次が DR 段)。σ_direction のアニールがこの段で 1500 → 3000
    iteration を掛けて走るので、``--max_iterations`` は 3000 以上にすること。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual_360"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)


@configclass
class K1WalkKickDual360DRPPORunnerCfg(K1WalkKick360PPORunnerCfg):
    """Stage 4 (全方位 + 観測 DR・最終)。``k1_walk_kick_dual_360`` から始める前提。

    環境の差は観測 DR とランプの定数化だけ (観測の次元・並びは Stage 3 と同一) なので、
    ネットワークと PPO ハイパラは Stage 3 と完全に同じ。experiment_name だけ分けて
    ログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_dual_360_dr"
        _use_history_cnn_policy(self)
        _use_mirror_loss(self)
