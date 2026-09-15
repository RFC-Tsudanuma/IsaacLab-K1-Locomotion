# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

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
from .history_actor_critic import RslRlHistoryActorCriticCfg
from .multi_expert_ppo import RslRlMultiExpertPpoAlgorithmCfg


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

        # 観測は「最新コマンド (command) + コマンドを除いた観測の履歴 (policy/critic)」の
        # 3 グループ構成 (flat_env_cfg.K1FlatObservationsCfg)。HistoryActorCritic が
        # 履歴グループから直近 mlp_history_steps ステップ分のみを MLP に入力する。
        self.policy = RslRlHistoryActorCriticCfg(
            init_noise_std=self.policy.init_noise_std,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.obs_groups = {
            "policy": ["command", "policy"],
            "critic": ["command", "critic"],
        }

        # 履歴観測 (policy 4600 + critic 5100 次元) により rollout storage が ~7.6GB
        # (4096 env) となり、20GB GPU では num_mini_batches=4 のミニバッチ + mirror loss
        # の拡張バッチが載らず OOM する。ミニバッチを半分にして更新時のピークを抑える
        # (データ総量・イテレーション数は不変、KL 適応 LR が勾配ステップ数の変化を吸収)。
        self.algorithm.num_mini_batches = 8

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
class K1FlatPosturePPORunnerCfg(K1FlatPPORunnerCfg):
    """上体の傾き・上下動抑制の仕上げ学習 (Isaac-Velocity-Flat-Posture) 用。

    K1FlatPPORunnerCfg との違いはイテレーション数のみ。仕上げ resume は
    m04 レシピ (+2500it) を既定とし、引数なしで 2 万イテレーション回って
    しまう事故を防ぐ。必要なら --max_iterations で上書きできる。
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2500


@configclass
class K1FlatGoalkeeperPPORunnerCfg(K1FlatPPORunnerCfg):
    """横移動特化 (Isaac-Velocity-Flat-Goalkeeper) 用。

    学習済み Flat ポリシーからの warm-start を前提に既定 10000 イテレーション
    (y カリキュラム拡張 + 5000 iter からの extreme 導入分)。resume の checkpoint
    解決を簡単にするため experiment_name は "k1_flat" を共有する (Posture と同様)。
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000


@configclass
class K1FlatFastPPORunnerCfg(K1FlatPPORunnerCfg):
    """通常歩行の高速実験版 (Isaac-Velocity-Flat-Fast) 用。

    学習済み Flat ポリシーからの warm-start を前提に既定 10000 イテレーション。
    resume の checkpoint 解決のため experiment_name は "k1_flat" を共有する。
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000


@configclass
class K1TurnPPORunnerCfg(K1FlatPPORunnerCfg):
    """高速その場回転タスク (Isaac-Velocity-Flat-Turn) 用。

    max_iterations は 8000 (K1FlatPPORunnerCfg の 20000 から削減)。歩行タスクの 20000 は
    3 次元コマンド空間 (vx ±2.0 / vy ±0.9 / ωz ±1.0) + extreme corner サンプリング +
    3 段速度カリキュラムを学ぶために必要だったが、本タスクのコマンドは Δψ の 1 次元で
    探索すべき空間が大幅に狭い。スクラッチで二足バランス自体は学ぶ必要があるため、
    その分の余裕を見て 8000 とする。

    NOTE: ``schedule="adaptive"`` (KL 適応 LR) なので max_iterations は純粋な停止条件で、
    LR スケジュールには一切影響しない。``save_interval=100`` で checkpoint が残るため、
    収束したら途中で止めてもその checkpoint がそのまま成果物になる。
    逆に足りなければ --max_iterations で伸ばすか --resume で継続すればよい。

    `init_noise_std` は既定のまま (スクラッチ学習なので finetune 用に下げない)。
    experiment_name だけ "k1_turn" に分けて、歩行タスクのログ・checkpoint と混ざらないようにする。

    観測レイアウトは歩行タスクと完全に同一 (1 ステップ 49 次元) なので、
    obs_groups / mirror loss / num_mini_batches は継承したままで正しく動く。
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "k1_turn"


def _multi_expert_algorithm(base: RslRlPpoAlgorithmCfg, learner_mode: str) -> RslRlMultiExpertPpoAlgorithmCfg:
    """既存の PPO 設定値を引き継いだ `RslRlMultiExpertPpoAlgorithmCfg` を作る (浅いコピー)。"""
    values = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name != "class_name"}
    return RslRlMultiExpertPpoAlgorithmCfg(**values, learner_mode=learner_mode)


@configclass
class K1TransitionWalkPPORunnerCfg(K1FlatPPORunnerCfg):
    """歩行 ⇄ 回転の遷移学習 (Isaac-Velocity-Flat-Transition-Walk): 歩行 expert を学習、回転 expert を凍結。

    PPO を `MultiExpertPPO` に差し替える以外は K1FlatPPORunnerCfg (HistoryActorCritic /
    mirror loss / num_mini_batches=8) と同一。凍結 expert の checkpoint は
    ``--frozen_ckpt turn=/path/model.pt`` で与える (cfg 既定は空)。

    experiment_name は "k1_transition" に分ける。学習元の歩行 checkpoint は
    ``--resume --checkpoint /abs/path/model.pt`` (絶対パス) で読み込むこと。
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "k1_transition"
        self.algorithm = _multi_expert_algorithm(self.algorithm, learner_mode="walk")


@configclass
class K1TransitionTurnPPORunnerCfg(K1TransitionWalkPPORunnerCfg):
    """遷移学習 (Isaac-Velocity-Flat-Transition-Turn): 回転 expert を学習、歩行 expert を凍結
    (``--frozen_ckpt walk=/path/model.pt``)。"""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learner_mode = "turn"
