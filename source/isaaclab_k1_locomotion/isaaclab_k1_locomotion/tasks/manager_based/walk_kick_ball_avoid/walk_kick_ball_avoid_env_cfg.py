# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick の「Ball Avoidance を原典の解釈で入れ直した」版。

非 dual 系の :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickEnvCfg` を継承する。
**1-2 が当初からの差分** (観測スロット 3 と接近系の罰の読み直し)、**3-5 は初回 run
(k1_walk_kick_ball_avoid, iter 1600) の結果を受けた報酬バランスの修正**。
他 (コマンド・ボール配置・終了条件・地形・他のキック報酬とそのカリキュラム) は
継承元とまったく同じ。

1. **観測スロット 3 = 現在のボール 3D 位置** (元は左足裏 ``sole_pos``)

   原典 B-Human "A Modular Ball Kicking Behavior with Reinforcement Learning"
   (Reichenberg & Frese) の観測表は "Current Ball 3D Position" で、``sole_pos`` は
   評価表のキャプション ("Evaluation Left Sole Booster K1") との取り違えだった
   (経緯は :mod:`..walk_kick_both_feet` の docstring)。

   観測グループの作り方は dual 系 (:mod:`..walk_kick_both_feet` →
   :mod:`..walk_kick_dual`) の流儀を踏襲し、**スロット 3 だけを差し替えた写し**を
   新しい ObsGroup として持つ (項の並びと項名は他スロットと完全に同じ)。
   ただし **中身は遅延なしの** :func:`~..walk_kick.mdp.observations.ball_pos_rel` に
   する。論文の "Current" にそのまま従うためで、dual 系が使う
   ``delayed_ball_pos_b`` (1 ステップ遅延) とはここが違う。

2. **``approach_penalty`` → ``ball_avoidance_exec``**

   ポスターの Ball Avoidance ``f(d_soleToBall)·p_kickPose`` (weight −3) を
   「構えが完成しているのに足がボールから遠い」ことへの罰として読み直したもの。
   キック接触の瞬間に距離側が 0 になって罰が消えるのが核心。式・パラメータ・
   両足平均を使う理由は :func:`~..walk_kick.mdp.rewards.ball_avoidance_exec` の
   docstring を参照。

初回 run の診断 (3-5 の前提)
---------------------------
キックは iter 300-400 に一度立ち上がった (kick_rate 0.19) 後、iter 1300 以降 0.00 に
消滅した。その間 mean_reward は −4.4 → +16 と単調増加しており、勾配死ではなく
**「蹴らない方が儲かる」を正しく学習した** 結果である。実測された経済は「蹴らずに
生きる dense 収入 ≈ +1.6/秒」「kick_finished (latch+2 秒) の早期終了で残り 4-5 秒ぶん
≈ +6〜8 を没収」「学習初期の蹴り 1 回の実収入 ≈ +0.3 (満額 7.8 の 4%)」で、
**蹴る = 約 −6 の取引** になっていた。旧 ``approach_penalty`` は「ボール近傍に居ない
こと」への恒常税だったのでこのバイアスを偶然相殺していたが、``ball_avoidance_exec``
(構えたときだけ課税) はその仕事を引き継いでいない。

3. **キック 3 項の end_weight を ×3** (:data:`_KICK_W_BOOST`)

   ``kick_direction`` 1.8 → 5.4、``kick_velocity_scaled`` 1.2 → 3.6、
   ``kick_velocity_strong`` 0.9 → 2.7。蹴り 1 回の期待収入そのものを持ち上げる。
   B-Human ポスター stage 2 の "Higher rewards are used to speed up training" に倣う。
   ランプ窓 (0 → 500 iteration) は継承元のまま。

4. **``kick_latch_bonus`` を新設** (weight :data:`_KICK_LATCH_BONUS_WEIGHT` = 4.0)

   latch からエピソード終了まで定額で払う項 (総額 ≈ weight × 2 秒 = +8)。キックの
   巧拙を問わず、早期終了で没収される歩行収入を相殺して「蹴る/蹴らない」の選択を
   収支中立に戻す。カリキュラムランプは付けず最初から有効。
   :func:`~..walk_kick.mdp.rewards.kick_latch_bonus` 参照。

5. **``ball_avoidance_exec`` のランプを後ろ倒し** (0 → 500 から **500 → 1000** へ)

   実行圧はキック行動が確立してから掛ける税。キック 3 項と同時にランプさせた初版では、
   蹴りの収入がまだ小さい時期に構えを課税してしまい、蹴り放棄への傾きを助長した。
   終端 weight (−3.0) は据え置き。

観測は **55 次元・並びとも継承元と同一** (critic も 61 次元のまま)。ただし
**スロット 3 の意味が変わる**ので、既存の ``k1_walk_kick_walk_phase`` /
``k1_walk_kick`` の checkpoint は流用しないこと (形の上では ``--load_pretrained`` が
通ってしまう)。この系列は :class:`K1WalkKickBallAvoidWalkPhaseEnvCfg` から
通しで学習する。

2 段レシピ::

    # stage 1 (歩行のみ)
    _labpython2 scripts/rsl_rl/train.py \\
        --task Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-Walk-Phase-v0 \\
        --headless --num_envs 4096
    # stage 2 (stage 1 の checkpoint から)
    _labpython2 scripts/rsl_rl/train.py \\
        --task Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-v0 \\
        --headless --num_envs 4096 \\
        --load_pretrained logs/rsl_rl/k1_walk_kick_ball_avoid_walk_phase/<run>/model_<N>.pt
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..locomotion.rough_env_cfg import _COMMAND_THRESHOLD, _PHASE_FREQ
from ..locomotion.velocity_env_cfg import ObservationsCfg
from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _KICK_STATE_PARAMS,
    _leg_joints,
    K1WalkKickEnvCfg,
    K1WalkKickWalkPhaseEnvCfg,
)

# --------------------------------------------------------------------------- #
# ball_avoidance_exec のパラメータ
#
# 値の根拠は :func:`~..walk_kick.mdp.rewards.ball_avoidance_exec` の docstring。
#
#   d_contact = 0.18 : 綺麗なインサイドの構えでの両足平均。ここで f = 0 に張り付く
#                      ので「接触したら罰ゼロ」が成立する。
#   d_sat     = 0.45 : 突き出し退行解 (平均 ≈ 0.32) で f ≈ 0.5、以遠は飽和。
#   sigma_pose = 0.3 : 継承元の approach_penalty / ball_avoidance と同じ値。構えの
#                      一致度 (P_kick からの距離) の効き幅を変えないため。
# --------------------------------------------------------------------------- #
_BALL_AVOIDANCE_EXEC_PARAMS = {
    "d_contact": 0.18,
    "d_sat": 0.45,
    "sigma_pose": 0.3,
}

# ball_avoidance_exec のフェードイン窓。継承元 (walk_kick_env_cfg の _phase2) の
# 0 → 500 に対し **500 iteration 遅らせてある**。実行圧は「キック行動が確立してから
# 掛ける税」であって、立ち上げの補助ではないため。初版はキック 3 項と同じ 0 → 500 で
# ランプさせたが、蹴りの収入がまだ小さい時期 (iter 0-400) に構えを課税してしまい、
# 蹴り放棄への傾きを助長した (診断は :class:`K1WalkKickBallAvoidEnvCfg` の
# ``__post_init__`` のコメント、およびモジュール docstring の 3-5 を参照)。
# steps_per_iteration は継承元と同じ 24。あちらは ``__post_init__`` のローカル変数
# なので参照できず、リテラルで持つ (360 の ball_avoidance_weight も同じ流儀)。
_BALL_AVOIDANCE_EXEC_PHASE = {"start_step": 500, "end_step": 1000, "steps_per_iteration": 24}

# ball_avoidance_exec の終端 weight。ポスターの Ball Avoidance と同じ −3
# (継承元の approach_penalty / 360 の ball_avoidance とも同値)。窓だけ遅らせ、
# 終端値は据え置く。
_BALL_AVOIDANCE_EXEC_WEIGHT = -3.0

# --------------------------------------------------------------------------- #
# キック 3 項 (項1-3) の end_weight 倍率
#
# 継承元は 6.0/4.0/3.0 × _KICK_W_SCALE(=0.3) = 1.8 / 1.2 / 0.9。これを一律 ×3 して
# 5.4 / 3.6 / 2.7 にする。根拠は下の「蹴らない方が儲かる」診断 (A) と、B-Human
# ポスター stage 2 の "Higher rewards are used to speed up training"。
# --------------------------------------------------------------------------- #
_KICK_W_BOOST = 3.0

# --------------------------------------------------------------------------- #
# kick_latch_bonus の weight
#
# latch からエピソード終了までの delay_steps(100) × dt(0.02) = 2 秒に定額で払うので、
# 1 キックあたりの総額は weight × 2.0 = +8.0。これは kick_finished による早期終了で
# 没収される dense な歩行収入 (≈ +1.6/秒 × 残り 4-5 秒 = +6〜8) の相殺額。
#
# NOTE: 総額が delay_steps に連動する。継承元の _KICK_DELAY_STEPS を変えたらこの値も
#       見直すこと (:func:`~..walk_kick.mdp.rewards.kick_latch_bonus` の NOTE 参照)。
# --------------------------------------------------------------------------- #
_KICK_LATCH_BONUS_WEIGHT = 4.0


# --------------------------------------------------------------------------- #
# Observations
#
# NOTE: :class:`..walk_kick.walk_kick_env_cfg.K1WalkKickPolicyCfg` /
#       ``K1WalkKickCriticCfg`` の写しに対し **スロット 3 だけ** を差し替えたもの。
#       ObsGroup は宣言順に連結されるので、項の順序を変えないこと。項名も
#       ``ball_pos`` 以外は元と揃えてある (walk phase の ``__post_init__`` が
#       項名でスロットを差し替えるため)。walk_kick 側の観測を変更したときは
#       こちらも追随させること。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickBallAvoidPolicyCfg(ObsGroup):
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
    # 3. Current Ball 3D Position (3) — 論文どおり **遅延なし**。walk_kick では左足裏
    #    (sole_pos) だったスロット。ノイズは他のボール位置観測 (prev_ball_pos) と揃える。
    ball_pos = ObsTerm(func=mdp.ball_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02))
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
    # 12. Previous Ball 2D Position (2) — 1 ステップ遅延
    prev_ball_pos = ObsTerm(func=mdp.prev_ball_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class K1WalkKickBallAvoidCriticCfg(ObsGroup):
    """Critic 用 61 次元。policy と同じ項をノイズ無しで並べ、特権情報を追加する。

    継承元 (:class:`..walk_kick.walk_kick_env_cfg.K1WalkKickCriticCfg`) との差も
    **スロット 3 だけ**。末尾の特権スロット ``ball_pos_rel`` は
    **そのまま残してある**: スロット 3 と同じ値になって重複するが、critic の次元
    (61) を継承元と揃えておくと、この系列の cfg を後から walk_kick 系の他タスクと
    並べたときに critic 側の形で悩まなくて済む。重複自体は学習上は無害
    (both_feet 系は逆に特権スロットを畳んで 58 次元にしている。流儀の違い)。
    """

    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel)
    ball_pos = ObsTerm(func=mdp.ball_pos_rel)
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
    prev_ball_pos = ObsTerm(func=mdp.prev_ball_pos_b)

    # -- 特権情報
    base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
    ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class K1WalkKickBallAvoidObservationsCfg(ObservationsCfg):
    policy: K1WalkKickBallAvoidPolicyCfg = K1WalkKickBallAvoidPolicyCfg()
    critic: K1WalkKickBallAvoidCriticCfg = K1WalkKickBallAvoidCriticCfg()


# --------------------------------------------------------------------------- #
# Stage 2: キック
# --------------------------------------------------------------------------- #
@configclass
class K1WalkKickBallAvoidEnvCfg(K1WalkKickEnvCfg):
    """Ball Avoidance (execution 解釈) 版の walk_kick (stage 2)。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickEnvCfg` との差は
    **観測スロット 3・接近系の罰・キック報酬のバランス (3 点)** だけ
    (計 5 項目。モジュール docstring 参照)。ボール配置 (±60° / 0.5-0.8 m)・
    蹴り方向 (±45°)・キック報酬項の中身・終了条件は一切変えていない
    (キック 3 項は end_weight の倍率だけ、ランプ窓は継承元のまま)。

    学習は :class:`K1WalkKickBallAvoidWalkPhaseEnvCfg` の checkpoint から::

        _labpython2 scripts/rsl_rl/train.py \\
            --task Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-v0 \\
            --headless --num_envs 4096 \\
            --load_pretrained logs/rsl_rl/k1_walk_kick_ball_avoid_walk_phase/<run>/model_<N>.pt
    """

    observations: K1WalkKickBallAvoidObservationsCfg = K1WalkKickBallAvoidObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- approach_penalty → ball_avoidance_exec
        #
        # approach_penalty は「姿勢が悪くても足をボールに近づけるだけで罰が減る」ので、
        # 構えの完成を待たずに突っ込む行動を後押ししてしまう。ball_avoidance_exec は
        # 逆に「構えが完成しているのに足が遠い」ことだけを罰し、接触した瞬間に
        # 距離側が 0 になって自分で消える (差し替えの流儀は 360 の
        # approach_penalty → ball_avoidance と同じ)。
        self.rewards.approach_penalty = None
        self.curriculum.approach_penalty_weight = None
        self.rewards.ball_avoidance_exec = RewTerm(
            func=mdp.ball_avoidance_exec,
            weight=0.0,
            params={**_KICK_STATE_PARAMS, **_BALL_AVOIDANCE_EXEC_PARAMS},
        )
        self.curriculum.ball_avoidance_exec_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "ball_avoidance_exec",
                "start_weight": 0.0,
                "end_weight": _BALL_AVOIDANCE_EXEC_WEIGHT,
                # 差分 5. ランプを 500 → 1000 に後ろ倒し (定数側のコメント参照)。
                **_BALL_AVOIDANCE_EXEC_PHASE,
            },
        )

        # ------------------------------------------------------------------ #
        # 初回 run (k1_walk_kick_ball_avoid, iter 1600) の診断
        # (モジュール docstring の差分 3-5 の共通の前提)
        #
        # 症状: キックは iter 300-400 に一度立ち上がった (kick_rate 0.19) 後、
        #       iter 1300 以降 0.00 に消滅した。その間 mean_reward は −4.4 → +16 と
        #       単調増加しており、勾配死ではなく **「蹴らない方が儲かる」を正しく
        #       学習した** 結果である。
        #
        # 実測された経済:
        #   * 蹴らずに生きる dense 収入 (feet_phase 等の歩行系正報酬) ≈ +1.6 / 秒
        #   * kick_finished が latch + 2 秒でエピソードを打ち切るので、蹴ると残り
        #     4-5 秒ぶん ≈ +6〜8 を没収される
        #   * 学習初期の蹴り 1 回の実収入 ≈ +0.3 (方向・速度が未熟で満額 7.8 の 4%)
        #   → 蹴る = 約 −6 の取引。
        #
        # 旧 approach_penalty は「ボール近傍に居ないこと」への恒常税だったので、この
        # バイアスを偶然相殺していた。ball_avoidance_exec (構えたときだけ課税) は
        # その仕事を引き継いでいないため、下の差分 3 / 4 で明示的に埋める
        # (差分 5 = 上の ball_avoidance_exec のランプ後ろ倒しも同じ診断から)。
        # ------------------------------------------------------------------ #

        # -- 差分 3. キック 3 項の end_weight を ×3
        #
        # 蹴り 1 回の期待収入そのものを持ち上げて、没収される dense 収入との差を詰める。
        # 継承元の end_weight (6.0/4.0/3.0 × _KICK_W_SCALE) を掛け算で上書きするので、
        # 継承元が配分や _KICK_W_SCALE を変えても比率はそのまま追随する。
        # B-Human ポスター stage 2 の "Higher rewards are used to speed up training"
        # に倣った処置でもある。ランプ窓 (0 → 500) は継承元のまま。
        #   kick_direction        : 1.8 → 5.4
        #   kick_velocity_scaled  : 1.2 → 3.6
        #   kick_velocity_strong  : 0.9 → 2.7
        self.curriculum.kick_direction_weight.params["end_weight"] *= _KICK_W_BOOST
        self.curriculum.kick_velocity_scaled_weight.params["end_weight"] *= _KICK_W_BOOST
        self.curriculum.kick_velocity_strong_weight.params["end_weight"] *= _KICK_W_BOOST

        # -- 差分 4. latch 後の定額ボーナス
        #
        # 差分 3 だけだと「上手く蹴れたときだけ」割に合うので、方向・速度が未熟な立ち上がり
        # 期には依然として蹴りが損になる。この項はキックの巧拙を問わず latch から
        # エピソード終了まで定額で払い (総額 ≈ +8)、早期終了で没収される歩行収入を
        # 相殺して「蹴る/蹴らない」の選択を収支中立に戻す。品質は項1-3 が見る。
        #
        # カリキュラムランプは付けない。相殺すべきバイアスは最初から存在するので、
        # 遅らせると立ち上がり期 (まさに蹴りが消えた区間) を素通りしてしまう。
        self.rewards.kick_latch_bonus = RewTerm(
            func=mdp.kick_latch_bonus,
            weight=_KICK_LATCH_BONUS_WEIGHT,
            params={**_KICK_STATE_PARAMS},
        )


@configclass
class K1WalkKickBallAvoidEnvCfg_PLAY(K1WalkKickBallAvoidEnvCfg):
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
class K1WalkKickBallAvoidWalkPhaseEnvCfg(K1WalkKickWalkPhaseEnvCfg):
    """Ball Avoidance 版の walk phase (stage 1)。ボール無しで歩行だけを学習する。

    基底 (:class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickWalkPhaseEnvCfg`) が
    ボール由来のスロット (``prev_ball_pos`` / ``kick_direction`` / ``ball_vel`` /
    ``target_kick_velocity`` と critic の ``ball_pos_rel``) を歩行コマンドや 0 に
    差し替えるところまでやるので、ここでは **新設したスロット 3 (ball_pos)** の
    面倒だけを見る (:mod:`..walk_kick_both_feet` の walk phase と同じ手当て)。

    スロット 3 をゼロ埋めではなく歩行コマンド (vx, vy, 0) にするのは、基底が
    ``prev_ball_pos`` に対してやっているのと同じ理由: 「スロットが指す方へ歩く」という
    入力→挙動の対応を kick phase と共通にしておくと歩容がそのまま転移する。

    報酬側の追加作業は無い。``approach_penalty`` とそのカリキュラムは基底が既に
    None 化しており、``ball_avoidance_exec`` / ``kick_latch_bonus`` とキック 3 項の
    weight 倍率は :class:`K1WalkKickBallAvoidEnvCfg` の ``__post_init__`` でしか
    触られない (この段は継承していない) ため。
    """

    observations: K1WalkKickBallAvoidObservationsCfg = K1WalkKickBallAvoidObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- スロット 3 (ボール 3D 位置) を歩行コマンドに差し替える。
        #    ボールはシーンごと消えているので、実体を読む関数を残すと落ちる。
        for _group in (self.observations.policy, self.observations.critic):
            _group.ball_pos.func = mdp.walk_command_xyz
            _group.ball_pos.params = {"command_name": "base_velocity"}


@configclass
class K1WalkKickBallAvoidWalkPhaseEnvCfg_PLAY(K1WalkKickBallAvoidWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
