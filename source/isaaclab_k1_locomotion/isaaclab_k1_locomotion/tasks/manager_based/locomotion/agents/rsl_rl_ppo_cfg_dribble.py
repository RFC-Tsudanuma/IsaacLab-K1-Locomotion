# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,  # noqa: F401  (利用可: 凍結方策が recurrent の場合に下記で使用)
    RslRlPpoAlgorithmCfg,
)


@configclass
class K1DribblePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "k1_dribble"
    # 上位 (ドリブル) 方策のネットワーク構造。
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    # 凍結する下位 (歩行) 方策のネットワーク構造。
    # ``_build_frozen_policy`` がこの設定でネットワークを構築してから checkpoint の
    # 重みを読み込む。上位方策 (``policy``) とは別物なので、読み込む歩行 checkpoint の
    # 構造に合わせてここで指定すること (例: flat 方策なら [512, 256, 128] / elu)。
    # 凍結方策が recurrent の場合は ``RslRlPpoActorCriticRecurrentCfg`` に差し替える。
    # critic_* は推論では使わないので actor 側に揃えておけばよい。
    low_level_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
