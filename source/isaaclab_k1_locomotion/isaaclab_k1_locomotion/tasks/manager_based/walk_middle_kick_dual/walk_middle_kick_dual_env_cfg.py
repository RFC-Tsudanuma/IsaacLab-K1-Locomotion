# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_middle_kick の dual 版 (dual encoder + 両足キック用の観測 2 変更)。

移植元は ``fewa/walk_kick_dual_encoder_tune`` が walk_long_pass に入れた dual encoder 化と、
:mod:`..walk_kick_both_feet` の 2 変更。共通ヘルパー
(:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history` ほか) は
:mod:`..walk_kick_dual` にあり、履歴長 H = 100 と遅延上限 0.02 s はそちらで 1 箇所管理。

dual に入っているもの
---------------------
1. **dual encoder**: actor の入力を 100 フレームの観測履歴にする。
2. **観測スロット 3 = ボール 3D 位置** (元は左足裏 ``sole_pos``)。
3. **歩行位相の初期オフセット {0, π}** (歩き出しの遊脚が右足に固定されるのを防ぐ)。
4. **着地 shaping 3 項の無効化** (fewa 47b8863 由来): 全 stage。
5. **feet_phase の weight 2.0 → 0.8** (fewa 47b8863 由来): 蹴りのある全 stage
   (この系列は walk phase を持たないので実質全 stage)。
6. **ボール観測の DR** (fewa 47b8863 由来): DR 段 (stage 4) のみ。
7. **軸足誘導 ``kick_plant_foot`` の無効化** (この系列だけの引き算): 全 stage で
   :func:`disable_kick_plant_foot`。既存 middle 3 run が一度も学習していない項で、
   実機転移に成功したレシピは誘導なし。dual 化と交絡させないために外す
   (理由と戻し方はあちらの docstring)。**walk_middle_kick 本体には残してある。**

2 と 3 を別 variant にせず畳み込むのは、dual 系がまだ未学習で失う互換性が無いため。
both_feet の stage 2 実測: ``kick_foot_right_frac`` 1.0 → 0.39、``kick_dir_error``
4.5°、``kick_rate`` 0.998 (方向精度・成功率を落とさずに左右が割れた)。
副作用として critic は 58 次元になる。

段構成 (**4 段**、fewa 47b8863 と同じ配置)
------------------------------------------
Stage 1 (walk phase) は :mod:`..walk_kick_dual` と共用::

    walk phase (walk_kick_dual) → middle → 360Middle (クリーン) → 360Middle DR (最終)

各段に何が載るかの表は :mod:`..walk_kick_dual.walk_kick_dual_env_cfg` の docstring
にある。**4 段構成と DR の配置は fewa 47b8863 に合わせた。最終段のボール観測 DR は
一様ノイズ + 連続遅延** (feat 側のガウスパイプラインは不採用、2026-08-17)。

middle のレシピ (weak の 3 点セット + 指令帯 (3.2, 4.5) + σ 終点 0.5 + 軸足配置
``kick_plant_foot`` + ボール物性 DR) は継承元の ``__post_init__`` で全て済んでおり、
このファイルは **観測の見え方 + 報酬 6 項 (着地 shaping 3 + feet_phase +
σ_direction + kick_plant_foot) しか触らない**。指令帯・ボール配置・終了条件は
1 バイトも変えていない。

``walk_long_pass_history`` と混同しないこと
-------------------------------------------
あちらは ``flatten_history_dim = True`` の **項単位 history** で、フレーム単位の
(N, H, 55) には戻せない別方式。checkpoint に互換性は無い
(詳細は :mod:`..walk_kick_dual.walk_kick_dual_env_cfg` の docstring)。

既存 1 フレームタスクとの checkpoint 互換性
--------------------------------------------
1 フレーム観測の checkpoint から始めるときは ``--warm_start_from_single_frame`` を
付ける。dual 系どうしを繋ぐときは不要。**引き継ぎ元は both_feet 系に限る**
(``k1_walk_kick_both_feet_walk_phase`` / ``k1_walk_kick_dual_walk_phase`` など)。
旧 sole_pos 系はスロット 3 の意味が違うので不可。

学習の通し実行は ``scripts/rsl_rl/train_walk_kick_middle_dual.sh``。
"""

from isaaclab.utils import configclass

from ..walk_kick_both_feet.walk_kick_both_feet_env_cfg import (
    _apply_phase_offset,
    K1WalkKickBothFeetObservationsCfg,
)
from ..walk_kick_dual.walk_kick_dual_env_cfg import (
    apply_dr_stage_recipe,
    apply_sigma_direction_anneal,
    disable_landing_shaping,
    enable_obs_history,
    hold_sigma_direction,
    rebalance_gait_vs_kick,
)
from ..walk_middle_kick.walk_middle_kick_env_cfg import (
    K1WalkMiddleKick360EnvCfg,
    K1WalkMiddleKick360EnvCfg_PLAY,
    K1WalkMiddleKickEnvCfg,
    K1WalkMiddleKickEnvCfg_PLAY,
)

# --------------------------------------------------------------------------- #
# DR 段 (stage 4) で定数化するフェードインの終点 [iteration]
#
# middle の **フェードイン** は 3 つで、どれも窓が 0 → 500:
#
#   * 基底 walk_kick のキック報酬ランプ (_phase2)
#   * middle が足す kick_plant_foot_weight (発見期に満額で乗せるため同じ窓)
#     — ただし dual では :func:`disable_kick_plant_foot` が項ごと消すので、
#       ここに残るのは 2 つ
#   * 360 の ball_avoidance_weight
#
# 後段に置かれている 3 つ (strong の折れ線 / σ アニール / overshoot 罰の
# 1500 → 3000 / plant_foot の σ_lon アニール) は対象外。
# _freeze_fade_in_curricula は func が linear_reward_weight の項だけを見て、さらに
# end_step > before_iter を除くので自動的にそうなる。
#
# NOTE: ball_avoidance_weight は Stage 3 (クリーン 360) でランプが完走するので、
#       DR 段では **除外なしの全凍結** でよい (fewa 47b8863 の Stage 4 と同じ)。
#
# NOTE: 指令帯 (3.2, 4.5) はカリキュラムで動かしていない (最初から固定) ので、
#       帯まわりで定数化するものは無い。fewa の ``kick_rate_gated_speed_range`` を
#       差し込む先も無いため、そちらは移植していない (mdp 側には関数だけある)。
# --------------------------------------------------------------------------- #
_FADE_IN_END_ITER = 500

# --------------------------------------------------------------------------- #
# 軸足誘導 (kick_plant_foot) を dual 系列から外すために消す項
#
# 継承元 (:func:`~..walk_middle_kick.walk_middle_kick_env_cfg._apply_middle_kick_recipe`)
# が stage 2/3/4 の全てで足す 1 報酬 + 2 カリキュラム。項名はあちらの実装と一致して
# いること (存在しなくなっていたら :func:`disable_kick_plant_foot` が例外を投げる)。
# --------------------------------------------------------------------------- #
_PLANT_FOOT_REWARD_TERM = "kick_plant_foot"
_PLANT_FOOT_CURRICULUM_TERMS = (
    "kick_plant_foot_weight",
    "kick_plant_foot_sigma_lon",
)


def disable_kick_plant_foot(cfg) -> None:
    """軸足配置の誘導項 ``kick_plant_foot`` とその 2 カリキュラムを無効化する。

    **middle_dual の全 stage から呼ぶ** (stage 2 / 3 / 4 とその PLAY)。継承元の
    ``_apply_middle_kick_recipe`` は 3 段すべてで走るので、1 段でも呼び忘れると
    そこだけ誘導が復活し、段の間で報酬構成が変わって前段の蹴り方が次段で罰せられる
    (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.disable_landing_shaping` と同じ
    型の不変条件)。``super().__post_init__()`` の直後、``apply_sigma_direction_anneal``
    や ``apply_dr_stage_recipe`` より **前** に呼ぶこと (後だと、消したはずの項に
    σ_direction のアニール項が付いた状態が一瞬できる)。

    なぜ dual では外すのか
    ---------------------
    ``kick_plant_foot`` は commit 1d0fac2 で walk_middle_kick に追加された項だが、
    **既存の middle 3 run はすべてそれ以前の学習で、この項を一度も学習していない**
    (各 run の ``env.yaml`` で確認済み)。つまり実機転移に成功したレシピは
    **軸足誘導なし**であり、middle_dual の初回 run でこれを入れると
    「dual 化の効果」と「誘導の効果」が交絡する。実績のあるレシピとの連続性を
    優先して、dual 側でだけ外す (**walk_middle_kick 本体には残してある**)。

    toe-poke の診断そのものは正しい
    -------------------------------
    ``middle_360_noisy`` の実測は plant_lon −0.45 / sole_height_at_kick 10.6 cm /
    仰角 4.4°。「軸足がボールの 45cm 後方 = 歩幅の途中で爪先で弾いている」という
    読み自体は妥当で、項の設計 (目標 −0.03、σ_lon 0.30 → 0.10) も理屈は通っている。
    外すのは効果を否定したからではなく、**1 度に 1 つしか変えない**ため。

    実機で距離不足やばらつきが出たら、**+誘導だけの ablation を別 run で立てる**こと
    (この関数の呼び出しを外せばそのまま「dual + 誘導」になる)。
    """
    if getattr(cfg.rewards, _PLANT_FOOT_REWARD_TERM, None) is None:
        raise AttributeError(
            f"報酬に '{_PLANT_FOOT_REWARD_TERM}' がありません。継承元の middle レシピが"
            " 変わった可能性があります (この関数はもう不要かもしれません)。"
        )
    setattr(cfg.rewards, _PLANT_FOOT_REWARD_TERM, None)

    for name in _PLANT_FOOT_CURRICULUM_TERMS:
        # 親の構成が変わって項が無くなっていても落ちないように getattr でガードする
        # (報酬本体と違い、カリキュラムは無くても「誘導が消えている」は成立する)。
        if getattr(cfg.curriculum, name, None) is not None:
            setattr(cfg.curriculum, name, None)


# --------------------------------------------------------------------------- #
# Stage 2 (middle, dual): 限定レンジ
# --------------------------------------------------------------------------- #
@configclass
class K1WalkMiddleKickDualEnvCfg(K1WalkMiddleKickEnvCfg):
    """Stage 2 (middle) の dual 版。

    継承元 (:class:`~..walk_middle_kick.walk_middle_kick_env_cfg.K1WalkMiddleKickEnvCfg`)
    との差は both_feet の観測 2 変更 + 履歴入力 + 報酬の 2 調整だけ。middle のレシピは
    基底の ``__post_init__`` で済んでおり、そちらは以下に触らないので衝突しない:

    * :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.disable_landing_shaping`
      (着地 shaping 3 項。全段で外す)
    * :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.rebalance_gait_vs_kick`
      (``feet_phase`` の weight 2.0 → 0.8。蹴りのある段だけ)
    * :func:`disable_kick_plant_foot` (軸足誘導を外す。理由はあちらの docstring)
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        disable_kick_plant_foot(self)
        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


@configclass
class K1WalkMiddleKickDualEnvCfg_PLAY(K1WalkMiddleKickEnvCfg_PLAY):
    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        disable_kick_plant_foot(self)
        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


# --------------------------------------------------------------------------- #
# Stage 3 (middle, dual): 全方位 (クリーン)。DR はまだ入れない
# --------------------------------------------------------------------------- #
@configclass
class K1WalkMiddleKickDual360EnvCfg(K1WalkMiddleKick360EnvCfg):
    """Stage 3 (middle, 全方位、クリーン) の dual 版。**最終段ではない** (次が DR 段)。

    差は both_feet の観測 2 変更 + 履歴入力 + 報酬の 2 調整
    (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.disable_landing_shaping` /
    :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.rebalance_gait_vs_kick`)。

    加えて **σ_direction のアニール** (0.35 → 0.15、1500 → 3000 iteration) が入る。
    方向の採点を締めて精度圧を上げるもので、**アニールの本体はこの段だけ**
    (次の DR 段は
    :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.hold_sigma_direction` で終値固定)。
    値と窓の根拠、および ``sigma_velocity`` のアニールとの違いは
    :data:`~..walk_kick_dual.walk_kick_dual_env_cfg._SIGMA_DIRECTION_ANNEAL_END`
    のコメント参照。**``--max_iterations`` は 3000 以上**。

    **観測 DR は入れない** (センサ遅延・ボール観測のノイズ拡大は次の DR 段の担当)。
    ``ball_avoidance_weight`` のランプもこの段で完走させる。

    軸足誘導 (``kick_plant_foot``) は :func:`disable_kick_plant_foot` で外す。
    σ_direction のアニールより **前** に呼ぶこと (消した項にアニールを付けない)。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        disable_kick_plant_foot(self)
        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)
        apply_sigma_direction_anneal(self)


@configclass
class K1WalkMiddleKickDual360EnvCfg_PLAY(K1WalkMiddleKick360EnvCfg_PLAY):
    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        disable_kick_plant_foot(self)
        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)
        # 学習側 (Stage 3) は σ_direction を 0.35 → 0.15 に絞る。PLAY では
        # カリキュラムが 0 iteration 目から始まって開始値に巻き戻るので、終値で固定する。
        hold_sigma_direction(self)


# --------------------------------------------------------------------------- #
# Stage 4 (middle, dual, 最終): 全方位 + 観測 DR
# --------------------------------------------------------------------------- #
@configclass
class K1WalkMiddleKickDual360DREnvCfg(K1WalkMiddleKick360EnvCfg):
    """Stage 4 (middle, 全方位 + 観測 DR) の dual 版。**この系列の最終 stage**。

    継承元は **クリーン 360** (:class:`~..walk_middle_kick.walk_middle_kick_env_cfg.K1WalkMiddleKick360EnvCfg`)。
    Stage 3 (:class:`K1WalkMiddleKickDual360EnvCfg`) と同じ土台に、
    :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.apply_dr_stage_recipe` を足しただけ:

    * ``observations`` を both_feet 版に差し替え + ``_apply_phase_offset``
    * :func:`enable_obs_history` (100 フレーム履歴)
    * :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.disable_landing_shaping` /
      :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.rebalance_gait_vs_kick`
    * :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.apply_dr_stage_recipe`
      = :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_delay`
      (IMU / エンコーダ ≤ 0.02 s + **ボール観測の遅延 DR とノイズ拡大**
      位置 ±0.07 m / 速度 ±0.5 m/s)
      + :func:`~..walk_kick_dual.walk_kick_dual_env_cfg._freeze_fade_in_curricula`
      (0 → 500 のランプ (キック報酬 / ball_avoidance) を除外なしで全凍結。
      ``kick_plant_foot_weight`` は :func:`disable_kick_plant_foot` が先に消す)
    * :func:`disable_kick_plant_foot` (軸足誘導。全 stage で外す)
      + :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.hold_sigma_direction`
      (σ_direction を Stage 3 の終値 0.15 で固定。アニールし直さない)

    **σ_direction のアニールは呼ばない。** Stage 3 で完走しているので、ここで
    やり直すと段の境界で 0.35 に戻り、詰めた方向精度が流れる。

    **ボール観測 DR は一様ノイズ + 連続遅延** (fewa 47b8863 準拠)。ガウスの認識
    パイプライン (``noisy_ball_pos_b``) は不採用なので、継承元も NoisyBall 系では
    なくクリーン 360 にしてある (ユーザー判断 2026-08-17。理由と戻し方は
    :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.apply_noisy_ball_pos_obs` の NOTE)。

    地形は継承元のまま (完全平面)。観測 55 次元・並びはどちらも不変なので、
    Stage 3 (dual) の checkpoint がそのまま載る::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-360-DR-v0 \
            --headless --num_envs 4096 --max_iterations 10000 \
            --load_pretrained logs/rsl_rl/k1_walk_middle_kick_dual_360/<run>/model_<N>.pt

    NOTE: 観測遅延は軸足の置き方に効き得る。報酬としての誘導は外してあるが
          (:func:`disable_kick_plant_foot`)、``Metrics/kick_direction/plant_lon`` /
          ``plant_lat`` は診断として出続けるので Stage 3 の run と併せて見ること。
          ここが Stage 3 より悪化していて、かつ実機で距離不足やばらつきが出るなら、
          **+誘導だけの ablation** を別 run で立てる (呼び出しを外すだけ)。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # 軸足誘導を外すのは apply_dr_stage_recipe より前。後だと
        # _freeze_fade_in_curricula が kick_plant_foot_weight を先に定数化してしまい、
        # 「消したはずの項のランプだけがログに残る」紛らわしい状態になる。
        disable_kick_plant_foot(self)
        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)
        apply_dr_stage_recipe(self, fade_in_end_iter=_FADE_IN_END_ITER)


@configclass
class K1WalkMiddleKickDual360DREnvCfg_PLAY(K1WalkMiddleKickDual360DREnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        # Unoise (ボール位置 ±0.07 / 速度 ±0.5 を含む) を切る。遅延は観測関数の側に
        # 入っているのでここでは残る (実機のレイテンシは PLAY でも起きるため)。
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
