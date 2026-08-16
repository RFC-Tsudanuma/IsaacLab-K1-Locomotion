# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""両足で蹴れるようにした walk_kick 系タスク。

既存の walk_kick 系は **右足でしか蹴らない** (``Metrics/kick_direction/kick_foot_right_frac``
が 1.0 に張り付く)。原因は環境側の左右非対称性で、:mod:`..walk_kick.mdp.commands` の
``KickDirectionCommand`` の docstring に挙がっている 3 点のうち **2 点をこのタスクで潰す**。

1. **観測に左足裏しか入っていなかった** (``walk_kick`` の観測スロット 3 = ``sole_pos``)
   → **ボール 3D 位置に差し替える**。

   これは原典 B-Human "A Modular Ball Kicking Behavior with Reinforcement Learning"
   (Reichenberg & Frese) の観測表と突き合わせて分かった実装の取り違えでもある。
   ポスターの表は Gravity3 / AngVel3 / **Current Ball 3D Position 3** / GaitPhase2 /
   JointPos12 / JointVel12 / PrevJointReq12 / PhaseFactorOffset1 / Flags3 / KickDir2 /
   TargetKickVel1 / BallVel2 / PrevBallPos2 = 58 次元 (K1、ネットワーク図の
   ``[58|61] × 256`` と一致) で、Flags を使わない walk_kick 系の 55 次元と
   スロット単位で対応する。``sole_pos`` の根拠として書かれていた「論文の評価表が
   Left Sole 基準」は、評価表のキャプション "Table: Evaluation Left Sole Booster K1"
   (= 左足で蹴るキックの評価) を観測項目と取り違えたもの。

   **報酬側の変更は不要**。報酬関数は ObservationManager を一切参照せず、
   ``d_sole_to_ball`` (左右の足のうちボールに近い方) など全て左右対称な量を
   シーンから直接計算しているため。

2. **歩行位相がエピソード開始時に必ず 0 だった** → ``randomize_phase_offset``
   (mode="reset") で env ごとに {0, π} を振る。

   位相は ``2π·pf·t`` (t = エピソード開始からの経過時間) で決まるので、オフセットが
   無いと全 env・全エピソードで位相 0 から始まる。``_STANCE_RATIO = 0.5`` の
   ``feet_phase`` では位相 0 の直後は「左が支持脚・右が遊脚」なので、**歩き出しの
   遊脚が常に右足**に固定される。蹴り足は接触時に遊脚だった方の足なので、これが
   そのまま「右足でしか蹴らない」に効く。

残る 3 点目 (mirror loss が無効) は手を付けていない。``locomotion.mdp.symmetry`` の
``_mirror_policy_obs`` は歩行タスクの 49 次元観測専用で、55 次元用の鏡像規約を
書き下ろす必要があるため。**このタスクで観測から左右非対称なスロットが消えたので、
mirror loss は原理的には定義可能になった** (次の一手の候補)。

学習の進め方
------------
観測の**意味**が変わる (スロット 3 が足裏 → ボール位置) ので、既存の
``k1_walk_kick_walk_phase`` の checkpoint は流用せず **walk phase から通しで学習する**::

    ./scripts/rsl_rl/train_walk_kick_both_feet.sh

policy の次元 (55) と並びは既存タスクと同じなので ``--load_pretrained`` は形の上では
通ってしまうが、スロット 3 に別の意味の値が入る。critic に至っては 61 → 58 次元に
変わっている (下の定数ブロックの NOTE 参照) ので、そもそも形が合わない。

効果の見方
----------
``Metrics/kick_direction/kick_foot_right_frac`` (0 = 常に左足、1 = 常に右足)。
0.5 付近に寄れば両足で蹴れている。0/1 に張り付いたままなら残る非対称性
(mirror loss) が効いているか、位相オフセットが観測から読めていない。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

from ..locomotion.mdp.events import randomize_phase_offset
from ..locomotion.rough_env_cfg import _COMMAND_THRESHOLD, _PHASE_FREQ
from ..locomotion.velocity_env_cfg import ObservationsCfg
from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    K1WalkKickEnvCfg,
    K1WalkKickWalkPhaseEnvCfg,
    _leg_joints,
)

# --------------------------------------------------------------------------- #
# ボール位置観測の遅延 [制御ステップ]
#
# スロット 3 (Current Ball 3D Position) とスロット 12 (Previous Ball 2D Position) を
# **同じ履歴の違う時点**から取る。論文は前者を遅延なしの現在位置としているが、
# 実機の vision は制御 (50Hz) より遅いので、walk_kick 系の従来方針
# (:func:`..walk_kick.mdp.observations.ball_pos_rel` の NOTE) に合わせて
# policy 側は現在位置スロットも 1 ステップ遅らせ、previous をその 1 つ前にする。
#
# critic は **同じスロットのまま遅延ゼロ** にする (_CRITIC 側の定数)。walk_kick 系の
# critic は元から「policy と同じ項を同じ位置に置いてノイズだけ外す」= その場で真値に
# する作りで、末尾に足すのは policy が持っていない量だけ (``base_lin_vel``) という
# 流儀。遅延だけがその場で外されずに残っていたので、埋め合わせとして
# ``ball_pos_rel`` (遅延ゼロ 3D) が特権スロットに足されていた。スロット 3 が
# ボール位置になった今、両者は式まで同じ値の遅延違いでしかないので、
# **遅延もその場で外して特権スロットは畳む**。critic は 58 次元 (= 55 + base_lin_vel)。
# --------------------------------------------------------------------------- #
_BALL_POS_DELAY = 1
_BALL_POS_PREV_DELAY = 2
# critic 用。0 = 遅延なしの真値、1 = その 1 つ前 (policy 側と同じ「現在と直前」の関係)。
_BALL_POS_DELAY_CRITIC = 0
_BALL_POS_PREV_DELAY_CRITIC = 1


# --------------------------------------------------------------------------- #
# Observations
#
# NOTE: :class:`..walk_kick.walk_kick_env_cfg.K1WalkKickPolicyCfg` の写しに対し
#       **スロット 3 だけ** を差し替えたもの。ObsGroup は宣言順に連結されるので、
#       項の順序を変えないこと。項名も ``ball_pos`` 以外は元と揃えてある
#       (walk phase の ``__post_init__`` が項名でスロットを差し替えるため)。
#       walk_kick 側の観測を変更したときはこちらも追随させること。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickBothFeetPolicyCfg(ObsGroup):
    """Actor 用 55 次元観測。walk_kick との差はスロット 3 のみ。

    論文の表と同じ並び:
      gravity(3) + ang_vel(3) + **ball_pos(3)** + gait_phase(2) + joint_pos(12)
      + joint_vel(12) + prev_joint_request(12) + phase_factor_offset(1)
      + kick_direction(2) + target_kick_velocity(1) + ball_vel(2)
      + prev_ball_pos(2) = 55
    """

    # 1. Gravity (3)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    # 2. Angular Velocity (3)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    # 3. Current Ball 3D Position (3) — 1 ステップ遅延。walk_kick では左足裏だったスロット。
    ball_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        noise=Unoise(n_min=-0.02, n_max=0.02),
        params={"delay_steps": _BALL_POS_DELAY, "dim": 3},
    )
    # 4. Gait Phase (2)
    gait_phase = ObsTerm(
        func=mdp.gait_phase_sincos,
        params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD},
    )
    # 5. Joint Position (12)
    joint_pos = ObsTerm(
        func=loco_mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.03, n_max=0.03),
        params={"asset_cfg": _leg_joints()},
    )
    # 6. Joint Velocity (12)
    joint_vel = ObsTerm(
        func=loco_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": _leg_joints()},
    )
    # 7. Previous Joint Request (12)
    prev_joint_request = ObsTerm(func=loco_mdp.last_action)
    # 8. Gait Phase Factor Offset (1)
    gait_phase_factor_offset = ObsTerm(
        func=mdp.gait_phase_factor_offset,
        params={"base_phase_freq": _PHASE_FREQ},
    )
    # 9. Kick Direction (2)
    kick_direction = ObsTerm(func=mdp.kick_dir_b, params={"command_name": "kick_direction"})
    # 10. Target Kick Velocity (1)
    target_kick_velocity = ObsTerm(
        func=mdp.target_kick_velocity,
        params={"command_name": "kick_direction"},
    )
    # 11. Ball 2D Velocity (2)
    ball_vel = ObsTerm(func=mdp.ball_vel_b, noise=Unoise(n_min=-0.4, n_max=0.4))
    # 12. Previous Ball 2D Position (2) — スロット 3 よりさらに 1 ステップ古い標本
    prev_ball_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        noise=Unoise(n_min=-0.02, n_max=0.02),
        params={"delay_steps": _BALL_POS_PREV_DELAY, "dim": 2},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1WalkKickBothFeetCriticCfg(ObsGroup):
    """Critic 用 58 次元。policy と同じ項を **ノイズ無し・遅延無し** で並べる。

    末尾の特権情報は真の base 線速度 (3) だけ。walk_kick の critic にあった
    ``ball_pos_rel`` (遅延ゼロのボール 3D 位置) は **持たない**: スロット 3 の
    ``ball_pos`` を ``delay_steps=0`` にすれば同じ値がその場で得られるので、
    別スロットに複製する意味が無い (定数ブロックの NOTE 参照)。

    ``prev_ball_pos`` も policy 側の「現在と直前」という関係だけを保って 1 段前に
    詰める。ボール速度は ``ball_vel`` がノイズ無しで入っているので、この 2 スロットに
    差分から速度を作らせる役目は無い。
    """

    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel)
    ball_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        params={"delay_steps": _BALL_POS_DELAY_CRITIC, "dim": 3},
    )
    gait_phase = ObsTerm(
        func=mdp.gait_phase_sincos,
        params={"phase_freq": _PHASE_FREQ, "cmd_threshold": _COMMAND_THRESHOLD},
    )
    joint_pos = ObsTerm(func=loco_mdp.joint_pos_rel, params={"asset_cfg": _leg_joints()})
    joint_vel = ObsTerm(func=loco_mdp.joint_vel_rel, params={"asset_cfg": _leg_joints()})
    prev_joint_request = ObsTerm(func=loco_mdp.last_action)
    gait_phase_factor_offset = ObsTerm(
        func=mdp.gait_phase_factor_offset,
        params={"base_phase_freq": _PHASE_FREQ},
    )
    kick_direction = ObsTerm(func=mdp.kick_dir_b, params={"command_name": "kick_direction"})
    target_kick_velocity = ObsTerm(
        func=mdp.target_kick_velocity,
        params={"command_name": "kick_direction"},
    )
    ball_vel = ObsTerm(func=mdp.ball_vel_b)
    prev_ball_pos = ObsTerm(
        func=mdp.delayed_ball_pos_b,
        params={"delay_steps": _BALL_POS_PREV_DELAY_CRITIC, "dim": 2},
    )

    # -- 特権情報: policy がそもそも持っていない量だけ
    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1WalkKickBothFeetObservationsCfg(ObservationsCfg):
    policy: K1WalkKickBothFeetPolicyCfg = K1WalkKickBothFeetPolicyCfg()
    critic: K1WalkKickBothFeetCriticCfg = K1WalkKickBothFeetCriticCfg()


# --------------------------------------------------------------------------- #
# 位相オフセットの event
# --------------------------------------------------------------------------- #
# {0, π} の二値。歩容の位相構造は今までと同じまま、歩き出しの遊脚が左右ちょうど
# 半々になる。連続一様 ([0, 2π)) にすると「周期の途中から歩き出す」ぶん難しくなるので、
# まずは二値で「両足で蹴れるか」だけを切り分ける。切り替えは mode="uniform"。
_PHASE_OFFSET_MODE = "binary"


def _apply_phase_offset(cfg) -> None:
    """エピソードごとに歩行位相の初期オフセットを振る event を追加する。

    ``mode="reset"`` なので毎エピソード引き直す。位相を読む観測 (``gait_phase``) と
    報酬 (``feet_phase``) は同じ ``get_phase_offset`` を経由するので自動的に整合する。
    ``__post_init__`` の最後 (基底クラスの設定が全部済んだ後) に呼ぶこと。
    """
    cfg.events.randomize_phase_offset = EventTerm(
        func=randomize_phase_offset,
        mode="reset",
        params={"mode": _PHASE_OFFSET_MODE},
    )


# --------------------------------------------------------------------------- #
# Stage 2: キック
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickBothFeetEnvCfg(K1WalkKickEnvCfg):
    """両足キック版の walk_kick (stage 2)。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickEnvCfg` との差は
    **観測スロット 3 と位相オフセットの 2 点だけ** (モジュール docstring 参照)。
    報酬・コマンド・地形・カリキュラム・終了条件は一切変えていないので、
    ``kick_foot_right_frac`` の差がそのままこの 2 点の効果になる。

    学習は :class:`K1WalkKickBothFeetWalkPhaseEnvCfg` の checkpoint から::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-v0 \
            --headless --num_envs 4096 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_both_feet_walk_phase/<run>/model_<N>.pt
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_phase_offset(self)


@configclass
class K1WalkKickBothFeetEnvCfg_PLAY(K1WalkKickBothFeetEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# --------------------------------------------------------------------------- #
# Stage 1: 歩行のみ
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickBothFeetWalkPhaseEnvCfg(K1WalkKickWalkPhaseEnvCfg):
    """両足キック版の walk phase (stage 1)。ボール無しで歩行だけを学習する。

    基底 (:class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickWalkPhaseEnvCfg`) が
    ボール由来のスロットを歩行コマンドに差し替えるところまでやるので、ここでは
    **新設したスロット 3 (ball_pos)** の面倒だけを見る。

    スロット 3 をゼロ埋めではなく歩行コマンド (vx, vy, 0) にするのは、基底が
    ``prev_ball_pos`` に対してやっているのと同じ理由: 「スロットが指す方へ歩く」という
    入力→挙動の対応を kick phase と共通にしておくと歩容がそのまま転移する。
    このタスクではスロット 3 と 12 が kick phase でもほぼ同じ値 (1 ステップ違いの
    ボール位置) なので、walk phase でも両方に同じ値を載せるのが素直な対応になる。

    位相オフセットは **stage 1 から入れる**。歩き出しの遊脚が左右半々になる歩行を
    ここで獲得しておかないと、stage 2 で位相オフセットだけ新規に入ることになり、
    転移した歩容が一度崩れる。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- スロット 3 (ボール 3D 位置) を歩行コマンドに差し替える。
        #    ボールはシーンごと消えているので、実体を読む関数を残すと落ちる。
        for _group in (self.observations.policy, self.observations.critic):
            _group.ball_pos.func = mdp.walk_command_xyz
            _group.ball_pos.params = {"command_name": "base_velocity"}

        _apply_phase_offset(self)


@configclass
class K1WalkKickBothFeetWalkPhaseEnvCfg_PLAY(K1WalkKickBothFeetWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
