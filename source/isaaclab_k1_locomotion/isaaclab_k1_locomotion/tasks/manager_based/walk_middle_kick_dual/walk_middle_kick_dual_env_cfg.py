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
このファイルは **観測の見え方 + 報酬 5 項 (着地 shaping 3 + feet_phase + σ_direction)
しか触らない**。指令帯・ボール配置・終了条件は 1 バイトも変えていない。

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
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_phase_offset(self)
        enable_obs_history(self)
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)


@configclass
class K1WalkMiddleKickDualEnvCfg_PLAY(K1WalkMiddleKickEnvCfg_PLAY):
    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
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
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
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
      (0 → 500 のランプ (キック報酬 / kick_plant_foot / ball_avoidance) を
      除外なしで全凍結)
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

    NOTE: 観測遅延は軸足配置 (``kick_plant_foot``) の実測値に効き得る。
          ``Metrics/kick_direction/plant_lon`` / ``plant_lat`` を Stage 3 の run と
          併せて見ること。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

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
