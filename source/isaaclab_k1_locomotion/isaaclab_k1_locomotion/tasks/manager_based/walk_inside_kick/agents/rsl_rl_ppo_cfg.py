# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...locomotion.networks import RslRlPpoActorCriticHistoryCnnCfg
from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg
from ...walk_kick_dual.agents.rsl_rl_ppo_cfg import (
    _CNN_FILTERS,
    _CNN_KERNEL_SIZES,
    _CNN_STRIDES,
    _NUM_MINI_BATCHES,
    _NUM_RECENT_FRAMES,
)


@configclass
class K1WalkInsideKickPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """右足インサイドキック。walk phase の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは Walk-Kick 系と同一に保つ (観測 55 次元・履歴なしも同じ)。
    experiment_name だけ分けて、既存の walk_kick / walk_middle_kick 系の run と
    ログが混ざらないようにする。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_inside_kick"


@configclass
class K1WalkInsideKickCleanPPORunnerCfg(K1WalkInsideKickPPORunnerCfg):
    """フォールバック (ボール観測ノイズ無し) 用。**通常は使わない。**

    本命と混ざると比較にならないので experiment_name を分ける。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_inside_kick_clean"


def _use_history_cnn_policy(cfg) -> None:
    """policy を :class:`~...locomotion.networks.ActorCriticHistoryCNN` に差し替える。

    stage 2/3 (:class:`K1WalkInsideKickDualPPORunnerCfg` /
    :class:`K1WalkInsideKickDualRoughPPORunnerCfg`) の両方が必ず呼ぶこと。
    **1 段でも呼び忘れると、そこで checkpoint の連鎖が切れる** — actor だけ形が違うので
    train.py に黙って捨てられ、起動ログの "Skipped N tensors" を読まない限り
    気づけない (現在は actor が 1 本も引き継げなければ止まるようにしてある)。

    PPO ハイパラ・MLP 幅・正規化の有無は継承元 (=flat 段) の値をそのまま引き継ぐ。
    切り出し方の定数 (直近フレーム数 K / CNN のカーネル・フィルタ・ストライド) は
    :mod:`...walk_kick_dual.agents.rsl_rl_ppo_cfg` から import する。履歴長 H = 100 は
    環境側 (:data:`...walk_kick_dual.walk_kick_dual_env_cfg._OBS_HISTORY_LENGTH`) が
    決め、ネットワークは観測の形 (N, H, D) から H を読む。

    ``num_mini_batches`` だけ上書きする
    -----------------------------------
    継承元は 4。履歴観測 (N, 100, 55) は 1 フレーム観測の 100 倍のメモリを食うので、
    dual 系と同じ 8 に割る (:data:`...walk_kick_dual.agents.rsl_rl_ppo_cfg._NUM_MINI_BATCHES`)。
    **総バッチ量 (1 iteration で舐めるサンプル数) は変わらない** ので、num_envs を
    減らす / H を縮める といった学習結果そのものを変える対策より副作用が小さい。
    関数名と役割 (ネットワークの差し替え) からは外れるが、履歴観測を使う RunnerCfg が
    必ず通る唯一の場所なので、漏れが起きないようここに置く (dual 側と同じ判断)。

    **mirror loss は入れない** (``algorithm.class_name`` を触らない)
    ---------------------------------------------------------------
    dual 系 (:func:`...walk_kick_dual.agents.rsl_rl_ppo_cfg._use_history_cnn_policy`) は
    ``PPOSparseMirror`` + ``_use_mirror_loss`` を掛けるが、こちらは掛けない。
    あちらの目的は「両足で蹴れるようにする」ことで、鏡像対称性
    (policy(mirror(obs)) ≈ mirror(policy(obs))) を promote する損失を足している。

    **このタスクは右足専用**。``kick_inside_contact`` は右足で当てた接触にしか
    払わない (:func:`~...walk_kick.mdp.rewards.kick_inside_contact` の右足ゲート) ので、
    左右対称なポリシーは報酬の定義と矛盾する。対称化は獲得済みの右足インサイドを
    壊す方向にしか働かない。``PPOSparseMirror`` は「mirror loss を 5 ミニバッチに 1 回に
    間引く」実装なので、mirror を使わない以上そちらも差し替える理由が無い
    (:mod:`...walk_lob_rough.agents.rsl_rl_ppo_cfg` と同じ判断)。
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
    # 履歴観測 (N, 100, 55) のメモリ対策。継承元の 4 を 8 に割る (総バッチ量は不変)。
    cfg.algorithm.num_mini_batches = _NUM_MINI_BATCHES


@configclass
class K1WalkInsideKickDualPPORunnerCfg(K1WalkInsideKickPPORunnerCfg):
    """stage 2 (平坦・観測履歴): ``k1_walk_inside_kick`` の checkpoint から始める前提。

    引き継ぎ元は **1 フレーム観測** なので ``--warm_start_from_single_frame`` が要る
    (付けないと actor が 1 本も引き継がれず train.py が止まる)。通しスクリプト
    :file:`scripts/rsl_rl/train_walk_inside_kick_dual.sh` が自動で付ける。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_inside_kick_dual"
        _use_history_cnn_policy(self)


@configclass
class K1WalkInsideKickDualRoughPPORunnerCfg(K1WalkInsideKickPPORunnerCfg):
    """stage 3 (凹凸 + ボール物性 DR): ``k1_walk_inside_kick_dual`` から始める前提。

    環境の差は地形と DR の帯だけ (観測の次元・並びは stage 2 と同一) なので、
    ネットワークと PPO ハイパラは stage 2 と完全に同じ。experiment_name だけ分けて
    ログが混ざらないようにする。履歴 → 履歴なので
    ``--warm_start_from_single_frame`` は **不要**。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_inside_kick_dual_rough"
        _use_history_cnn_policy(self)
