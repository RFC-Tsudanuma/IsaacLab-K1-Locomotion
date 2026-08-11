# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...locomotion.networks import RslRlPpoActorCriticHistoryCnnCfg
from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickWalkPhasePPORunnerCfg
from ...walk_loop_pass.agents.rsl_rl_ppo_cfg import K1WalkLoopPass360PPORunnerCfg, K1WalkLoopPassPPORunnerCfg

# --------------------------------------------------------------------------- #
# Actor に「そのまま」入れる直近フレーム数 K
#
# 履歴長 H = 50 は環境側 (walk_long_pass_env_cfg._OBS_HISTORY_LENGTH) が決め、
# ネットワークは観測の形 (N, H, D) から H を読む。ここで持つのは切り出し方だけ。
# --------------------------------------------------------------------------- #
_NUM_RECENT_FRAMES = 5

# --------------------------------------------------------------------------- #
# 履歴符号化 CNN: 1 次元・隠れ層 2 つ
#
# [kernel size, filter size, stride size] = [6, 32, 3] と [4, 16, 2]。
# 系列長は 50 → 15 → 6 と縮み、潜在は 16 * 6 = 96 次元。
# actor MLP の入力は 5*55 + 96 = 371 次元になる。
# --------------------------------------------------------------------------- #
_CNN_KERNEL_SIZES = [6, 4]
_CNN_FILTERS = [32, 16]
_CNN_STRIDES = [3, 2]


def _use_history_cnn_policy(cfg) -> None:
    """policy を :class:`~...locomotion.networks.ActorCriticHistoryCNN` に差し替える。

    PPO ハイパラ・MLP 幅・正規化の有無は継承元の値をそのまま引き継ぐ。

    long_pass 系列の 4 段すべてがこれを呼ぶ。**1 段でも呼び忘れると、そこで
    checkpoint の連鎖が切れる** (actor だけ形が違うので train.py に黙って捨てられ、
    起動ログを読まない限り気づけない)。
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
class K1WalkLongPassWalkPhasePPORunnerCfg(K1WalkKickWalkPhasePPORunnerCfg):
    """Stage 1 (歩行のみ) の履歴入力版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_walk_phase"
        _use_history_cnn_policy(self)


@configclass
class K1WalkLongPassLoopPassPPORunnerCfg(K1WalkLoopPassPPORunnerCfg):
    """Stage 2 (限定レンジのループパス) の履歴入力版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_loop_pass"
        _use_history_cnn_policy(self)


@configclass
class K1WalkLongPassLoop360PPORunnerCfg(K1WalkLoopPass360PPORunnerCfg):
    """Stage 3 (全方位ループパス) の履歴入力版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_loop_360"
        _use_history_cnn_policy(self)


@configclass
class K1WalkLongPassPPORunnerCfg(K1WalkLoopPass360PPORunnerCfg):
    """Stage 4 (ロングパス本体)。actor は 50 フレームの観測履歴を見る。

    観測の中身 (55 次元) と行動空間は Walk-Kick 系と同じだが、actor の入力だけが
    「直近 5 フレームそのまま + 50 フレームの CNN 潜在」に変わっている。PPO の
    ハイパラは継承元のまま。experiment_name だけ分けて、他のキック run とログが
    混ざらないようにする。

    .. note::
        引き継ぎ元は **共用の** loop_pass_360 ではなく、履歴入力版の Stage 3
        (:class:`K1WalkLongPassLoop360PPORunnerCfg`, experiment_name
        ``k1_walk_long_pass_loop_360``)。共用タスクの checkpoint から
        ``--load_pretrained`` すると actor の重みが 1 つも引き継がれない
        (critic・観測正規化の統計・action noise std だけが残り、起動ログに
        "Skipped 8 tensors" が出る)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass"
        _use_history_cnn_policy(self)
