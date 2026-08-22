# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_fewa 系列の PPO ランナー設定 (Stage 1-4)。

移植元
======
**コミット 47b8863 / ブランチ ``fewa/walk_kick_dual_encoder_tune`` の
``walk_long_pass/agents/rsl_rl_ppo_cfg.py`` の逐語コピー**
(experiment_name を ``k1_walk_long_pass_fewa*`` に振り直しただけ)。
この設定の Stage 4 checkpoint が **実機で動いている** ので、
PPO ハイパラ・CNN の形・履歴長は触らない。

このブランチ側で書き換えが要った箇所は無い
(``locomotion.networks.RslRlPpoActorCriticHistoryCnnCfg`` も
``walk_kick`` / ``walk_loop_pass`` の継承元 RunnerCfg もそのまま存在する)。
"""

from isaaclab.utils import configclass

# import した時点で ActorCriticHistoryCNN が rsl_rl の名前空間に登録される
# (OnPolicyRunner は class_name を eval で解決するため)。
from ...locomotion.networks import RslRlPpoActorCriticHistoryCnnCfg
from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickWalkPhasePPORunnerCfg
from ...walk_loop_pass.agents.rsl_rl_ppo_cfg import K1WalkLoopPass360PPORunnerCfg, K1WalkLoopPassPPORunnerCfg

# --------------------------------------------------------------------------- #
# Actor に「そのまま」入れる直近フレーム数 K
#
# 履歴長 H = 50 は環境側 (walk_long_pass_fewa_env_cfg._OBS_HISTORY_LENGTH) が決め、
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
class K1WalkLongPassFewaWalkPhasePPORunnerCfg(K1WalkKickWalkPhasePPORunnerCfg):
    """Stage 1 (歩行のみ) の履歴入力版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_walk_phase"
        _use_history_cnn_policy(self)


@configclass
class K1WalkLongPassFewaLoopPassPPORunnerCfg(K1WalkLoopPassPPORunnerCfg):
    """Stage 2 (限定レンジのループパス) の履歴入力版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_loop_pass"
        _use_history_cnn_policy(self)


@configclass
class K1WalkLongPassFewaLoop360PPORunnerCfg(K1WalkLoopPass360PPORunnerCfg):
    """Stage 3 (全方位ループパス) の履歴入力版。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_loop_360"
        _use_history_cnn_policy(self)


@configclass
class K1WalkLongPassFewaPPORunnerCfg(K1WalkLoopPass360PPORunnerCfg):
    """Stage 4 (ロングパス本体)。actor は 50 フレームの観測履歴を見る。

    観測の中身 (55 次元) と行動空間は Walk-Kick 系と同じだが、actor の入力だけが
    「直近 5 フレームそのまま + 50 フレームの CNN 潜在」に変わっている。PPO の
    ハイパラは継承元のまま。experiment_name だけ分けて、他のキック run とログが
    混ざらないようにする。

    .. note::
        引き継ぎ元は **共用の** loop_pass_360 ではなく、履歴入力版の Stage 3
        (:class:`K1WalkLongPassFewaLoop360PPORunnerCfg`, experiment_name
        ``k1_walk_long_pass_fewa_loop_360``)。共用タスクの checkpoint から
        ``--load_pretrained`` すると actor の重みが 1 つも引き継がれない
        (critic・観測正規化の統計・action noise std だけが残り、起動ログに
        "Skipped 8 tensors" が出る)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa"
        _use_history_cnn_policy(self)


# --------------------------------------------------------------------------- #
# Stage 4 の ablation (walk_long_pass_fewa_ablation_env_cfg)
#
# 環境側の変更は報酬の weight とカリキュラムの終点だけで、観測・行動空間は
# 基底と同一。PPO ハイパラもネットワークも基底から変えない (変えると
# 「1 変種につき 1 箇所」が崩れて比較にならない)。experiment_name だけ分けて
# logs が混ざらないようにする。
#
# 出発 checkpoint はどれも基底 Stage 4 と同じものを使えるので、
# scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh が LP_CKPT を全変種へ配る。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFewaBand6PPORunnerCfg(K1WalkLongPassFewaPPORunnerCfg):
    """Ablation A: 帯の終点を (3.2, 6.0) にした Stage 4。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_band6"


@configclass
class K1WalkLongPassFewaCalmPPORunnerCfg(K1WalkLongPassFewaPPORunnerCfg):
    """Ablation B: 跳ねに効く 3 項の weight を変えた Stage 4。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_calm"


@configclass
class K1WalkLongPassFewaBand6CalmPPORunnerCfg(K1WalkLongPassFewaPPORunnerCfg):
    """Ablation C: A + B。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_band6calm"


@configclass
class K1WalkLongPassFewaGroundedPPORunnerCfg(K1WalkLongPassFewaPPORunnerCfg):
    """Ablation D: 軸足の接地を測る報酬項を足した Stage 4。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_grounded"


@configclass
class K1WalkLongPassFewaBand6GroundedPPORunnerCfg(K1WalkLongPassFewaPPORunnerCfg):
    """帯 6.0 + 軸足接地 (A + D)。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_band6grounded"


@configclass
class K1WalkLongPassFewaBand6CalmGroundedPPORunnerCfg(K1WalkLongPassFewaPPORunnerCfg):
    """帯 6.0 + 跳ね罰 + 軸足接地 (A + B + D)。"""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_long_pass_fewa_band6calmgrounded"
