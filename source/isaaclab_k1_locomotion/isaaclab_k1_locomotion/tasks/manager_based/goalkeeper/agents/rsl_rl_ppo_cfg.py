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

from ..mdp.symmetry import compute_symmetric_states, compute_symmetric_states_high_level


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
        # ★ 2026-08-09: data augmentation を有効化し、mirror loss 係数を 0.5 → 2.0 に上げた。
        #   34600→54599 iter の run で「右への横移動だけ足が上がらず転びやすい」非対称が
        #   目視で出た。URDF の関節軸・可動域・質量・関節の並び順はすべて正しくミラーで、
        #   Loss/symmetry も 0.009 で安定していたので、構造バグではなく学習の残差。
        #   (mirror loss 導入前は逆に左が悪かった、という記録が上のコメントにある通り、
        #    どちらが劣るかは run ごとに入れ替わる = 残差の典型的な症状。)
        #
        #   mirror loss は「出力」を縛るだけで、critic の価値推定・advantage・観測正規化の
        #   統計は左右非対称なまま残る。data augmentation はバッチ自体を左右均等にするので
        #   そちらも揃う。学習時間は collection 1.1s に対し learning 0.1s なので、
        #   バッチが 2 倍になっても全体では +10% 程度。
        #
        #   有効化の前提として compute_symmetric_states に critic 観測の反転を実装した
        #   (それが無いと critic が食い違った組を学習する)。goalkeeper/mdp/symmetry.py 参照。
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=2.0,
        )


@configclass
class K1GKDirectStage2PPORunnerCfg(K1GKDirectPPORunnerCfg):
    """Stage 2: ゴール + ボールでセーブを学習する本編。**Stage 1 ckpt から --resume**。

    ★ 2026-07-24: 旧「Stage2 (初速固定レンジ 12000) → Stage3 (適応 8000)」の 2 段構成を
      廃止し 1 段に統合した (train_gk_direct_stage2.sh 参照)。適応カリキュラムは
      難易度の初期値が最も易しい側から始まるので、固定レンジ専用の段は不要だった。
      統合したぶん反復数は旧 2 段の合計 (12000 + 8000) を引き継ぐ。
    ★ 2026-07-31: 旧 Stage3 用のクラスをこちらに一本化し、experiment_name も
      ``k1_gk_direct_stage2`` に統一した (ステージ番号とログ出力先を一致させるため)。
      統合直後の 3 run 分のログは ``k1_gk_direct_stage3/`` に残っている。
    """

    max_iterations = 20000
    experiment_name = "k1_gk_direct_stage2"


# ---------------------------------------------------------------------------
# 階層版 v2 (goalkeeper_hier_env_cfg.py)
#   凍結下位 = k1_gk_direct_stage1/2026-07-28 (実機デプロイ実績あり、横 1.28 m/s)
#   上位 action = 歩行コマンド (vx, vy, wz) の 3 次元
# ---------------------------------------------------------------------------

@configclass
class K1GKHierPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """階層版ゴールキーパーの上位ポリシー PPO 設定。

    Stage1 → Stage2 で同じネットワーク形状を使う (``--resume`` で ckpt を渡すため
    途中で変更しないこと)。experiment_name はステージ別サブクラスで分ける。
    """

    num_steps_per_env = 24
    save_interval = 100
    max_iterations = 10000
    experiment_name = "k1_gk_hier_stage2"

    # 上位 (ゴールキーパー戦術) 方策のネットワーク構造。
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    # 凍結する下位 (07-28) のネットワーク構造。
    # ★ 実際には exported/policy.pt (TorchScript) を渡すのでこの設定は使われない
    #   (TorchScript は正規化器ごと焼き込まれている)。生の model_*.pt を渡す運用に
    #   切り替えるときのために、07-28 の学習時設定と一致させてある。
    #   その場合 actor_obs_normalization=True が必須: 07-28 は観測正規化ありで
    #   学習されており、ここが False だと _build_frozen_policy が
    #   actor_obs_normalizer.* を strict=False で黙って捨て、正規化なしで走る。
    low_level_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
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
        # ★ 2026-08-13: 0.004 → 0.008。1 回目の Stage1 学習で mean_noise_std が
        #   初期 0.7 から 0.045 まで潰れ、iter 1000 以降は方策がほぼ決定論的になって
        #   同じ状態しか訪れなくなっていた (全指標が横ばい)。物理 DR を reset に
        #   変えて状態分布を広げる以上、探索も残っていないと活かせない。
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # 上位にも左右対称性を掛ける。観測レイアウトは直接制御版と同一で、行動だけが
        # 12 関節 → 歩行コマンド 3 次元に変わるので専用の変換関数を使う。
        # 凍結下位 (07-28) は data augmentation 無し・係数 0.5 の世代で、横移動時に
        # 約 10°/s の yaw ドリフト (左右非対称歩容の典型症状) が残っている。上位まで
        # 左右で別戦略に収束すると片側のセーブだけ弱いキーパーになるので入れておく。
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states_high_level,
            mirror_loss_coeff=2.0,
        )


@configclass
class K1GKHierStage1PPORunnerCfg(K1GKHierPPORunnerCfg):
    """Stage 1: ボールなし。ランダム目標 y への到達・停止と、姿勢/前後位置の維持。

    上位が学ぶのは実質「3 つの数字の出し方」だけなので、報酬の最適化自体は
    1000 iter 程度で頭打ちになる (1 回目の学習で確認済み)。

    ★ 2026-08-13: 3000 → 5000。ただし「報酬をさらに伸ばす」ためではなく、
      **物理 DR を reset モードに変えた分の経験を稼ぐため**。startup のままでは
      物理パラメータが 4096 通りで固定なので反復を増やしても何も広がらなかったが、
      reset にした今は 1 iter ごとに新しい物理条件のエピソードが積み上がる。
      実機転移を見据えて分布を広く踏ませる、という目的の反復。
    """

    max_iterations = 5000
    experiment_name = "k1_gk_hier_stage1"


@configclass
class K1GKHierStage2PPORunnerCfg(K1GKHierPPORunnerCfg):
    """Stage 2: ゴール + ボール + 適応カリキュラム。Stage1 ckpt から --resume。"""

    max_iterations = 20000
    experiment_name = "k1_gk_hier_stage2"
