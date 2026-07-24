# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mdp.symmetry import compute_symmetric_states


@configclass
class K1GoalkeeperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """ゴールキーパー上位ポリシーの PPO 設定 (around_ball 系と同じハイパラから開始)。

    ステージ1→2→3 で同じネットワーク形状を使う (checkpoint を --resume で受け渡す
    ため変更しないこと)。experiment_name はステージ別サブクラスで分ける。
    """

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "k1_goalkeeper_stage2"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
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


@configclass
class K1GoalkeeperStage1PPORunnerCfg(K1GoalkeeperPPORunnerCfg):
    """ステージ1 (ボールなし・目標到達と停止)。"""

    max_iterations = 4000
    experiment_name = "k1_goalkeeper_stage1"


@configclass
class K1GoalkeeperStage3PPORunnerCfg(K1GoalkeeperPPORunnerCfg):
    """ステージ3 (適応初速カリキュラム)。Stage2 ckpt から --resume で継続する。"""

    max_iterations = 8000
    experiment_name = "k1_goalkeeper_stage3"


# ---------------------------------------------------------------------------
# 直接制御版 (goalkeeper_direct_env_cfg.py)
# ---------------------------------------------------------------------------

@configclass
class K1GKDirectPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """直接制御版ゴールキーパーの PPO 設定 (12 関節を直接出力)。

    ``actor_hidden_dims`` は歩行 ckpt (0524_walk.pt = [256,128,128]) からの
    warmstart 互換のため **変更しないこと**。ステージ間も同一形状で ``--resume``
    により重みを引き継ぐので、途中で変えると学習をやり直すことになる。
    """

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "k1_gk_direct_stage1"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # 左右対称性を mirror loss として学習に加える (locomotion の K1FlatPPORunnerCfg と同設定)。
        # policy(mirror(obs)) ≈ mirror(policy(obs)) を促す MSE 損失が PPO 損失に加算される。
        #
        # 初回学習 (mirror loss なし) では「右への横移動は良好だが、左へ動くときだけ
        # 左膝を曲げる」非対称な歩容に収束した。左右を別々に学習した結果、片側だけ
        # 良い解を見つけて反対側が劣った解のまま固定される典型例。mirror loss を入れると
        # できている側 (右) の歩容に反対側が引き寄せられる。
        #
        # data augmentation は使わない (locomotion と揃える)。actor は MLP なので
        # recurrent 方策で mirror loss が動かない問題 (locomotion のコメント参照) は該当しない。
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.5,
        )


@configclass
class K1GKDirectStage2PPORunnerCfg(K1GKDirectPPORunnerCfg):
    """Stage 2 (ゴール + ボールでセーブ)。Stage 1 ckpt から --resume。"""

    max_iterations = 12000
    experiment_name = "k1_gk_direct_stage2"


@configclass
class K1GKDirectStage3PPORunnerCfg(K1GKDirectPPORunnerCfg):
    """Stage 3 (適応初速カリキュラム)。Stage 2 ckpt から --resume。"""

    max_iterations = 8000
    experiment_name = "k1_gk_direct_stage3"
