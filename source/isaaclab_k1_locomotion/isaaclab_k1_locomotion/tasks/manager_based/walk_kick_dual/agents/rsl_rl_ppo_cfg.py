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

# --------------------------------------------------------------------------- #
# PPO の update 時のミニバッチ分割数。継承元 (locomotion/agents/rsl_rl_ppo_cfg.py の
# K1RoughPPORunnerCfg.algorithm) は 4。**dual 系列だけ 8 に上げる。**
#
# 理由: 履歴観測 (N, 100, 55) と mirror loss の組み合わせで GPU メモリが足りない。
# rsl_rl の PPO は mirror loss を計算する直前に、symmetry で 2 倍にした観測を
# ``obs_batch.detach().clone()`` する (ppo.py:328)。実測 (2026-08-17、16 GB GPU):
#
#   4096 env × 48 steps ÷ 4 mini batch = 49,152 サンプル
#   49,152 × (100 × 55) × 4 B ≈ 1.01 GiB          ← ミニバッチの観測
#   symmetry でバッチ 2 倍 → clone が 2.02 GiB    ← ここで OOM
#
# 8 に割ると 1 回あたりの経路のピークが半分になり、~2.5 GiB 下がる。
# **総バッチ量 (1 iteration で舐めるサンプル数) は変わらない**。分割数が増えるぶん
# 勾配のノイズは増えるが、49,152 → 24,576 サンプルはまだ十分大きい。
#
# NOTE: 1 フレーム観測の both_feet 系は観測が 1/100 なので無関係。この上書きを
#       :func:`_use_history_cnn_policy` に置いてあるのは、**履歴を使う RunnerCfg が
#       必ずここを通る**ため (dual 3 ファミリーの 10 stage 全部)。both_feet はこの
#       関数を呼ばないので影響しない。
# NOTE: num_envs を減らす / 履歴長 H を縮める方向でも同じ効果は出るが、どちらも
#       学習結果そのものを変える。ミニバッチ分割は「同じ更新を何回に分けるか」
#       だけなので、OOM 対策として副作用がいちばん小さい。
# --------------------------------------------------------------------------- #
_NUM_MINI_BATCHES = 8

# --------------------------------------------------------------------------- #
# PPO の実装を :class:`~...locomotion.networks.PPOSparseMirror` に差し替える。
#
# 素の rsl_rl は mirror loss を **全ミニバッチ** (5 epoch × 8 = 40 回/update) に
# 掛ける。1 回あたり「観測を 2 倍にして actor をもう一度 forward + backward」
# なので、履歴 CNN の actor では learning が実測 0.78 → 2.54 s/iteration (×3.3)。
# PPOSparseMirror は 5 ミニバッチに 1 回 (40 回中 8 回) だけ掛けるので、
# learning は ~1.1 s に戻る見込み。詳細は ppo_sparse_mirror.py の docstring。
#
# NOTE: **1 フレーム観測の both_feet 系はこの差し替えをしない** (素の PPO のまま)。
#       あちらは観測が 1/100 で mirror loss のコストが誤差なので、間引く理由が無い。
#       この非対称は意図的で、「dual だけ mirror の実効係数が 1/5」という違いに
#       なる。both_feet と dual を並べて比較するときはこの点に注意すること。
# NOTE: 間引いたぶん平均的な対称化の圧も 1/5 になるが、係数
#       (both_feet 側の ``_MIRROR_LOSS_COEFF`` = 0.5) は据え置きにしてある。
#       上げ下げの判断材料は ppo_sparse_mirror.py の NOTE を参照。
# --------------------------------------------------------------------------- #
_ALGORITHM_CLASS_NAME = "PPOSparseMirror"


def _use_history_cnn_policy(cfg) -> None:
    """policy を :class:`~...locomotion.networks.ActorCriticHistoryCNN` に差し替える。

    PPO ハイパラ・MLP 幅・正規化の有無は継承元の値をそのまま引き継ぐ。**ただし
    ``num_mini_batches`` と ``class_name`` だけは上書きする**
    (:data:`_NUM_MINI_BATCHES` / :data:`_ALGORITHM_CLASS_NAME`)。関数名と
    役割 (ネットワークの差し替え) からは外れるが、履歴観測を使う RunnerCfg が必ず
    通る唯一の場所なので、漏れが起きないようここに置いてある。理由はどちらも
    履歴観測 × mirror loss のコスト (GPU メモリと learning 時間) で、詳細は
    それぞれの定数のコメント参照。

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
    # 履歴観測 × mirror loss の OOM 対策。継承元の 4 を 8 に割る (総バッチ量は不変)。
    cfg.algorithm.num_mini_batches = _NUM_MINI_BATCHES
    # mirror loss を 5 ミニバッチに 1 回だけ掛ける PPO に差し替える (learning の高速化)。
    cfg.algorithm.class_name = _ALGORITHM_CLASS_NAME


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
