# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlSymmetryCfg,
)

from ..mdp.symmetry import compute_symmetric_states
from ..rough_env_cfg import _USE_RECURRENT_POLICY


@configclass
class K1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 50
    experiment_name = "k1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7207805082202461,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=0.7207805082202461,
    #     actor_obs_normalization=True,
    #     critic_obs_normalization=True,
    #     actor_hidden_dims=[128],
    #     critic_hidden_dims=[128],
    #     activation="elu",
    #     rnn_type="gru",
    #     rnn_hidden_dim=256,
    #     rnn_num_layers=2,
    # )
    # actor = RslRlRNNModelCfg(
    #     init_noise_std=0.7207805082202461,
    #     obs_normalization=True,
    #     rnn_type="lstm",
    #     rnn_hidden_dim=[128,128,128],
    #     rnn_num_layers=2,
    #     stocastic=True,
    # )
    # critic = RslRlRNNModelCfg(
    #     init_noise_std=0.7207805082202461,
    #     obs_normalization=True,
    #     rnn_type="lstm",
    #     rnn_hidden_dim=[256,128,128],
    #     rnn_num_layers=2,
    #     stocastic=True,
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005399484409787433,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.00012551115172973836,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class K1FlatPPORunnerCfg(K1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "k1_flat"
        # self.policy.actor_hidden_dims = [256, 128, 128]
        # self.policy.critic_hidden_dims = [256, 256, 128]
        self.save_interval = 100

        # 左右対称性を mirror loss として学習に加える (data augmentation は使わない)。
        # policy(mirror(obs)) ≈ mirror(policy(obs)) を促す MSE 損失が PPO 損失に加算される。
        # mirror_loss_coeff は損失の重み (要調整)。
        # NOTE: rsl_rl の symmetry はミニバッチを batch_size[0] に沿って 2 倍にする前提で
        #       実装されており、recurrent (LSTM/GRU) 方策ではミニバッチが [時間T, 軌跡数N] の
        #       2 次元 + act_inference が単一ステップ扱いになるため構造的に動作しない。
        #       そのため MLP のときだけ mirror loss を使い、recurrent のときは env 側の
        #       joint_mirror_symmetry 報酬で対称性を担保する (_USE_RECURRENT_POLICY で排他切替)。
        if not _USE_RECURRENT_POLICY:
            self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
                use_data_augmentation=False,
                use_mirror_loss=True,
                data_augmentation_func=compute_symmetric_states,
                mirror_loss_coeff=0.5,
            )
        if _USE_RECURRENT_POLICY and self.policy.__class__ is not RslRlPpoActorCriticRecurrentCfg:
            raise ValueError(
                "When using recurrent policy, please use RslRlPpoActorCriticRecurrentCfg for policy configuration."
            )


@configclass
class K1GetupPPORunnerCfg(K1RoughPPORunnerCfg):
    """起き上がり (get-up) 用の PPO 設定。

    K1RoughPPORunnerCfg を継承 (symmetry_cfg 無し)。全身 22 自由度なので、脚 12 自由度前提の
    mirror loss/対称拡張 (K1FlatPPORunnerCfg 側) は使わない。左右対称性は env の body_symmetry
    報酬で担保する。ログは experiment_name "k1_getup" に分離する。
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000
        self.experiment_name = "k1_getup"
        self.save_interval = 200
