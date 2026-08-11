# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...locomotion.networks import RslRlPpoActorCriticHistoryCnnCfg
from ...walk_loop_pass.agents.rsl_rl_ppo_cfg import K1WalkLoopPass360PPORunnerCfg

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


@configclass
class K1WalkLongPassPPORunnerCfg(K1WalkLoopPass360PPORunnerCfg):
    """ロングパス用。actor は 50 フレームの観測履歴を見る。

    観測の中身 (55 次元) と行動空間は Walk-Kick 系と同じだが、actor の入力だけが
    「直近 5 フレームそのまま + 50 フレームの CNN 潜在」に変わっている。PPO の
    ハイパラは継承元のまま。experiment_name だけ分けて、他のキック run とログが
    混ざらないようにする。

    .. warning::
        actor の形が変わったので、loop_pass_360 の checkpoint から
        ``--load_pretrained`` しても **actor の重みは 1 つも引き継がれない**
        (critic・観測正規化の統計・action noise std は形が同じなので引き継がれる)。
        train.py は形の合わないテンソルを黙って捨てるため、起動ログの
        "Skipped 8 tensors" で確認すること。歩行・キックの挙動は actor の学習し直し
        になるので、帯カリキュラムの前提が変わる点に注意
        (walk_long_pass_env_cfg のモジュール docstring 参照)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass"

        # -- actor を履歴 + CNN エンコーダ付きに差し替える
        #    (PPO ハイパラ・MLP 幅・正規化の有無は継承元の値をそのまま引き継ぐ)
        base = self.policy
        self.policy = RslRlPpoActorCriticHistoryCnnCfg(
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
