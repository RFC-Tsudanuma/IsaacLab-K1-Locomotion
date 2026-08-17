# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick_both_feet の RunnerCfg。

mirror loss (:func:`_use_mirror_loss`) の配線はここに集約されている。dual 3 ファミリー
(:mod:`..walk_kick_dual` / :mod:`..walk_weak_kick_dual` / :mod:`..walk_middle_kick_dual`)
の agents からも import されるので、係数と鏡像写像はここ 1 箇所で管理すること。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ...walk_kick.agents.rsl_rl_ppo_cfg import K1WalkKickPPORunnerCfg
from ..symmetry import compute_symmetric_states

# --------------------------------------------------------------------------- #
# mirror loss の重み
#
# 歩行タスク (locomotion/agents/rsl_rl_ppo_cfg.py の K1FlatPPORunnerCfg) と同じ 0.5。
# use_data_augmentation / use_mirror_loss の組み合わせもあちらと同一にしてある
# (拡張サンプルは学習バッチに足さず、mirror loss としてだけ効かせる)。
#
# NOTE: 揃えてあるのは「歩行で実績のある設定から始める」ため。walk_kick 系で
#       調整するなら kick_foot_right_frac (0.5 に寄れば両足) と kick_rate /
#       kick_dir_error_deg を併せて見ること。上げすぎると「左右対称に振る舞う」ことが
#       「指令どおり蹴る」より優先されて、方向精度が落ちる。
# --------------------------------------------------------------------------- #
_MIRROR_LOSS_COEFF = 0.5


def _use_mirror_loss(cfg) -> None:
    """PPO に左右対称性の mirror loss を加える (data augmentation は使わない)。

    ``policy(mirror(obs)) ≈ mirror(policy(obs))`` を促す MSE 損失が PPO 損失に
    加算される。鏡像写像は :func:`..symmetry.compute_symmetric_states`
    (policy 55 / critic 58 / action 12 次元。dual の (N, 100, 55) にもそのまま効く)。

    **both_feet と dual 3 ファミリーの全 stage がこれを呼ぶ。**
    :func:`~...walk_kick_dual.agents.rsl_rl_ppo_cfg._use_history_cnn_policy` と同じ型の
    不変条件で、**1 段でも呼び忘れるとそこで checkpoint の連鎖の意味が壊れる**:
    ネットワークの形は変わらないので重みはそのまま載ってしまい、起動ログにも何も
    出ないまま、その段だけ対称化の圧が消えて片足へ戻る (気づけるのは
    ``Metrics/kick_direction/kick_foot_right_frac`` が 0/1 へ張り付いてから)。

    既存 checkpoint からは始められない
    ----------------------------------
    both_feet / dual の既存 run は **既に片足に収束している**
    (``kick_foot_right_frac`` が run ごとに 0.99 / 0.01 へ張り付く)。そこから
    mirror loss を掛けると、対称化は「獲得済みの蹴り足を壊す」方向にしか働かない。
    mirror loss を入れた系列は **walk phase (stage 1) から回し直すこと**
    (ユーザー判断)。

    NOTE: rsl_rl の symmetry は recurrent 方策では動かない (ミニバッチが
          [時間 T, 軌跡数 N] の 2 次元になり、バッチを batch_size[0] に沿って
          2 倍にする前提が崩れる)。walk_kick 系は MLP / ActorCriticHistoryCNN
          (``is_recurrent = False``) なので該当しない。GRU に差し替えるときは
          この配線を外すこと。
    """
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=True,
        data_augmentation_func=compute_symmetric_states,
        mirror_loss_coeff=_MIRROR_LOSS_COEFF,
    )


@configclass
class K1WalkKickBothFeetPPORunnerCfg(K1WalkKickPPORunnerCfg):
    """Stage 2 (両足キック)。stage 1 の checkpoint から --load_pretrained で始める前提。

    ネットワーク・PPO ハイパラは walk_kick 系と同一に保つ (観測 55 次元も同じ)。
    experiment_name だけ分けて、既存の walk_kick の run とログが混ざらないようにする。

    差は :func:`_use_mirror_loss` (左右対称性の mirror loss) だけ。観測から左右非対称な
    スロット (左足裏) が消えたことで 55 次元用の鏡像規約が書けるようになったので、
    「右足でしか蹴らない」の残る 3 点目をここで潰す。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_both_feet"
        _use_mirror_loss(self)


@configclass
class K1WalkKickBothFeetWalkPhasePPORunnerCfg(K1WalkKickBothFeetPPORunnerCfg):
    """Stage 1 (両足キック版の歩行のみ)。観測・ネットワークは stage 2 と同一に保つ。

    mirror loss は **この段から** 掛ける。歩き出しの遊脚が左右半々になる歩容を
    ここで作っておかないと、stage 2 で対称化だけが新規に入って転移した歩容が一度
    崩れる (位相オフセットを stage 1 から入れているのと同じ理由)。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "k1_walk_kick_both_feet_walk_phase"
        _use_mirror_loss(self)
