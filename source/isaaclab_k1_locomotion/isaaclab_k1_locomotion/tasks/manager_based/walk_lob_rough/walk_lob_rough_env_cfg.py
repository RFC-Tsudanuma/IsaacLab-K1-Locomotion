# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_lob の凹凸地形 + 履歴入力版。**歩行から通しで学習し直す 2 段構成**。

``k1_walk_lob/2026-08-16_08-08-20/model_11600.pt`` を実機に載せたところ「一度だけ
大きく浮き、他の試行も浮きそうな傾向はある」という結果になった。その run の
tfevents を読んで分かったことを起点に、**浮きの再現性** を取りにいくための系列。

実測から分かったこと (2026-08-16 の run、iteration 9998-11624)
--------------------------------------------------------------
======================================  ==========  ==================================
メトリクス                                実測         備考
======================================  ==========  ==================================
``kick_rate``                            0.997       蹴れてはいる
``kick_apex_height``                     0.425 m     ボール中心の絶対高さ。目標 0.9 m
``kick_elevation_deg``                   23.7°       ``phi_sat`` 60° に対して遠い
``foot_vz``                              0.81 m/s    ``vz_foot_sat`` 2.0 の 40%
``plant_lon``                            −0.42 m     目標 −0.03、σ_lon 0.10
``sole_height_at_kick``                  0.083 m     ボール中心 0.11 の 2.7 cm 下
``Episode_Reward/kick_plant_foot``       0.0002      **完全に死んでいる**
======================================  ==========  ==================================

1. **``kick_plant_foot`` が一度も効いていない。** 実測 plant_lon = −0.42 に対して
   目標 −0.03・σ_lon = 0.10 なので f_lon = exp(−0.39²/(2·0.10²)) ≈ 5e-4。
   ガウスの裾の完全に外側で、**報酬も勾配もゼロ**。実際 1600 iteration のあいだ
   plant_lon は −0.421 → −0.433 と一切動いていない。
   ``..walk_lob.walk_lob_env_cfg`` の ``_PLANT_LON_TARGET`` のコメントが
   「現状が −0.25 のように大きく後方ならカリキュラムで目標を動かす方が素直」と
   予告していたケースにそのまま該当する。

2. **run 全体がプラトーしている。** it10150 以降、apex 0.402 → 0.409、
   foot_vz 0.755 → 0.806、elevation 23.7° → 23.7°。1500 iteration 動いていない。

3. **浮きはすくい上げで作られていない。** apex 0.425 m = 上昇 0.315 m ⇔ 打ち出し
   vz ≈ 2.49 m/s に対して ``foot_vz`` は 0.81 m/s。差分を作っているのは接触法線の
   向き、つまり「ボール中心より下に速い水平速度で当てている」ことの方。
   足裏高さ h = 0.083 での法線仰角は asin((0.11−0.083)/0.11) = 14° で、実測の
   射出仰角 25° は 14° に foot_vz 分が乗った値として整合する。

   このメカニズム自体は運動学なので実機にも乗る。ただし **「ボール中心の 2.7 cm 下」
   という狭い窓を当てられたときだけ** 成立するので、実機で「一度だけ浮いた」という
   結果になる。狙うべきは窓を広げること = 当たり所をもっと下げること。

walk_lob からの変更点
---------------------
1. **凹凸地形** (:func:`~..walk_kick.walk_kick_env_cfg._apply_rough_terrain`)。
   段差・坂道なし、起伏 0-4 cm のランダムノイズのみ (``WALK_KICK_ROUGH_TERRAIN_CFG``)。
   **stage 1 (歩行) から入れる**。段の間で地形が変わると転移した歩容が一度崩れるので、
   通しで同じ条件にする。ボールは起伏ぶん浮かせてから落とす
   (``reset_ball.params["spawn_clearance"]``、_apply_rough_terrain が面倒を見る)。

2. **観測履歴 100 フレーム + HistoryCNN**
   (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)。
   これも stage 1 から。ネットワークが
   :class:`~..locomotion.networks.ActorCriticHistoryCNN` に変わるので、
   **1 フレーム観測の既存 checkpoint とは互換性が無い** (walk phase からやり直す
   理由の一つ)。

3. **センサ遅延 DR** (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_delay`)。
   IMU / エンコーダは stage 1 から (``sources=("body",)``)、ボール観測 (視覚) は
   ボールが存在する stage 2 から。遅延の有無で歩き方が変わるので、内界センサ側は
   段をまたいで条件を揃える。

4. **観測スロット 3 を左足裏 → ボール 3D 位置に差し替える**
   (:class:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg.K1WalkKickBothFeetObservationsCfg`
   をそのまま使う)。B-Human 原典の観測表では
   スロット 3 は Current Ball 3D Position で、walk_kick 系の ``sole_pos`` は
   評価表キャプション "Left Sole" の誤読だった (詳細は walk_kick_both_feet の
   モジュール docstring)。``sole_pos`` は joint_pos 12 次元から FK で完全に決まる
   冗長情報なので、差し替えは論文準拠と情報量の両面で正味プラス。
   **歩行から学習し直すこの系列では checkpoint 互換の制約が無いので、ここで直す。**

   **両足キック化 (位相オフセット {0, π}) と mirror loss は入れない。** dual 系は
   observation の変更とセットでこの 2 つも入れているが、こちらの目的は apex 高さで
   あって両足で蹴れることではない。両足を学ばせるぶん学習が重くなるのを避ける
   (ユーザー判断 2026-08-18)。したがって ``kick_foot_right_frac`` は 1.0 付近に
   張り付いたままになる想定で、それは異常ではない。

5. **``kick_plant_foot`` の目標と σ をカリキュラムで動かす** (下の
   ``_PLANT_*_START`` 群)。固定目標では届かないことが実測で分かったので、
   **実測値の側から始めて目標へ引っ張る**。これが本系列の本丸。

6. **``kick_contact_height`` (新規) を足す**
   (:func:`~..walk_kick.mdp.rewards.kick_contact_height`)。
   接触時の足裏高さを直接下げさせる項。5 と表裏の関係で、あちらが原因側 (構え)、
   こちらが結果側 (当たり所)。ガウスではなく線形ランプなので、どこから始めても
   勾配が死なない = カリキュラム不要。

7. **``kick_foot_lift`` の重みを 2.0 → 4.0 に上げる**。実測 foot_vz 0.81 は
   ``vz_foot_sat`` 2.0 の 40% で飽和には遠く、圧力を上げる余地がある。ただし
   5 が効かないと運動学的に足を上へ振れないので、単独では動かない想定。

継承したまま変えないもの
------------------------
* ロブの報酬設計 (``kick_velocity_scaled`` 撤去 / ``vz_sat`` 5.0 / ``phi_sat`` 60° /
  σ_direction 0.6)。walk_lob の設計をそのまま使う。
* ``disable_landing_shaping`` / ``rebalance_gait_vs_kick`` は **呼ばない**。
  どちらも dual 系 (fewa 由来) のレシピで、walk_lob の歩容設計とは別系統。
  変更を「地形・履歴・遅延・当たり所」に絞って切り分けを保つ。
* ボール物性の DR。

  .. note::
     ボールの反発係数について。``soccer_ball`` の spawn material は
     restitution 0.6 / combine_mode ``average`` だが、地面 (terrain) とロボットの
     material は restitution 0.0 / combine_mode ``multiply`` で、``sim.physics_material``
     も terrain のものが使われる。PhysX は 2 材質のうち **優先度の高い combine mode**
     (average < min < multiply < max) を採るので実効は ``multiply`` になり、
     ボール↔地面・ボール↔足はいずれも 0.6 × 0.0 = **0.0** になるはず。
     つまり ``ball_physics_material`` の DR (restitution 0.0-0.7) は実質効いておらず、
     すでに実機と同じ e≈0 で学習していることになる。``kick_foot_lift`` の docstring が
     前提にしている「Isaac 既定 e≈0.6」はこの cfg には当てはまらない。
     上の実測 3 (浮きが接触法線ジオメトリ由来である) とも整合する。
     **sim を動かして確かめてはいないので、物理側は今回変更していない。**
     実測で e が 0 でないと分かった場合だけ DR 範囲を 0.0-0.2 に絞ること。

学習の進め方
------------
観測の意味 (スロット 3) もネットワーク (履歴 CNN) も変わるので、既存の
``k1_walk_lob`` / ``k1_walk_lob_walk_phase`` の checkpoint は **一切流用しない**。
walk phase から通しで学習する::

    ./scripts/rsl_rl/train_walk_lob_rough.sh

効果の見方
----------
``Metrics/kick_direction/`` の 4 つを並べて見る。

* ``plant_lon``          : −0.42 から −0.03 側へ動くか (5 が効いているか)
* ``sole_height_at_kick``: 0.083 から 0.03 台へ下がるか (6 が効いているか)
* ``kick_elevation_deg`` : 24° から上がるか
* ``kick_apex_height``   : 0.42 m から上がるか (最終目標 0.9 m)

前 2 つが動いたのに後ろ 2 つが動かないなら「当たり所は仮説どおり下がったが仰角の
律速はそこではなかった」ということなので、``kick_foot_lift`` 側 (すくい上げ) か
足の形状 (collider が convex_hull なので、そもそもつま先がボールの 3 cm 下に
入れるか) を疑うこと。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _apply_rough_terrain,
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
)
from ..walk_kick_both_feet.walk_kick_both_feet_env_cfg import (
    _BALL_POS_DELAY,
    _BALL_POS_PREV_DELAY,
    K1WalkKickBothFeetObservationsCfg,
)
from ..walk_kick_dual.walk_kick_dual_env_cfg import (
    _OBS_DELAY_MAX_S,
    enable_obs_delay,
    enable_obs_history,
)
from ..walk_lob.walk_lob_env_cfg import (
    _LOB_SIGMA_DIRECTION,
    _PLANT_LON_TARGET,
    _PLANT_SIGMA_LON,
    K1WalkLobEnvCfg,
    K1WalkLobWalkPhaseEnvCfg,
)

# --------------------------------------------------------------------------- #
# カリキュラムの時間単位
#
# ``linear_reward_weight`` / ``linear_reward_param`` は
# ``step = common_step_counter // steps_per_iteration`` を使うので、
# ``steps_per_iteration`` に 1 iteration あたりの env ステップ数 (= RunnerCfg の
# ``num_steps_per_env``) を渡すと start/end_step が **iteration 単位** になる。
# walk_kick 系のカリキュラムは全てこの流儀で 24 を使っている。
# --------------------------------------------------------------------------- #
_STEPS_PER_ITERATION = 24

# キック報酬のフェードイン (weight 0 → 最終値) が完了する iteration。
# walk_lob から継承する全カリキュラムがこの値で、ここでも揃える。
_KICK_FADE_IN_END_ITER = 500

# --------------------------------------------------------------------------- #
# 軸足配置 (kick_plant_foot) のカリキュラム
#
# **なぜ必要か**: 固定目標 (_PLANT_LON_TARGET = −0.03, _PLANT_SIGMA_LON = 0.10) では
# 実測 plant_lon = −0.42 に対して f_lon ≈ 5e-4 で、報酬も勾配もゼロだった
# (モジュール docstring の実測 1)。ガウスは「正解の位置に立ったら褒める」形なので、
# そこへ至る坂が無いと一切動かない。**目標の側を実測値から出発させて引っ張る。**
#
#   _PLANT_LON_TARGET_START = -0.42 : 2026-08-16 run の収束値。ここなら f_lon ≈ 1。
#   _PLANT_SIGMA_LON_START  = 0.25  : 出発時点の許容幅。歩行から学習し直すので初期の
#       plant_lon は −0.42 ちょうどではない (あの値は収束後の値)。σ を広めに取って
#       出発点のばらつきを覆う。0.25 なら −0.42 ± 0.25 = [−0.67, −0.17] で半値以上。
#   終値は walk_lob の定数をそのまま使う (−0.03 / 0.10)。
#
#   アニール区間 [500, 4000] iteration: 開始をキック報酬のフェードイン完了
#   (_KICK_FADE_IN_END_ITER) に合わせ、重みが立ち上がってから目標を動かし始める。
#   終了 4000 は「stage 2 を 15000 iteration 回す」想定に対して前半で締め切る値。
#
# NOTE: 目標と σ を **同時に** 動かす。目標だけ動かすと σ=0.25 のまま緩い採点が
#       残り、σ だけ絞ると届かないまま裾の外に落ちる (元の失敗の再現)。
# NOTE: 効いているかは Metrics/kick_direction/plant_lon が −0.42 から離れて
#       いくかで見る。Curriculum/kick_plant_foot/lon_target に目標側の値が出るので、
#       2 本を重ねると「目標に追随できているか」がそのまま読める。
# --------------------------------------------------------------------------- #
_PLANT_LON_TARGET_START = -0.42
_PLANT_SIGMA_LON_START = 0.25
_PLANT_ANNEAL_START_ITER = _KICK_FADE_IN_END_ITER
_PLANT_ANNEAL_END_ITER = 4000

# --------------------------------------------------------------------------- #
# 接触高さ報酬 (kick_contact_height) の定数
#
# f_low = clamp((ball_radius − h) / (ball_radius − h_sat), 0, 1)。導出と根拠は
# :func:`~..walk_kick.mdp.rewards.kick_contact_height` の docstring 参照。
#
#   _CONTACT_HEIGHT_BALL_RADIUS = 0.11 : ボール半径。ここに当てると法線が水平 = 0 点。
#   _CONTACT_HEIGHT_SAT         = 0.03 : 満点になる足裏高さ。法線仰角 45° 相当
#       (asin((0.11−0.03)/0.11) = 47°)。**これより下げても得をしない** ようにして、
#       つま先を地面へ掻き込むだけの解に動機を与えない。
#   _CONTACT_HEIGHT_WEIGHT      = 2.0  : kick_plant_foot / kick_foot_lift と同じ。
#       direction (6.0) / loft (5.0) / elevation (5.0) より明確に下げる。この項も
#       目的そのものではなく **「蹴り方」の指定** なので、大きくすると高さより
#       当たり所の最適化に学習が引っ張られる。
#
# NOTE: 線形ランプなので実測 h = 0.083 でも f_low = 0.34 が付き、勾配も生きている。
#       kick_plant_foot と違ってカリキュラムは要らない。
# --------------------------------------------------------------------------- #
_CONTACT_HEIGHT_BALL_RADIUS = 0.11
_CONTACT_HEIGHT_SAT = 0.03
_CONTACT_HEIGHT_WEIGHT = 2.0

# --------------------------------------------------------------------------- #
# すくい上げ (kick_foot_lift) の重み。walk_lob の 2.0 から引き上げる。
#
# 実測 foot_vz 0.81 m/s は vz_foot_sat 2.0 の 40% で飽和には遠く、圧力を上げる
# 余地がある。4.0 は loft (5.0) / elevation (5.0) にほぼ並ぶ水準で、「浮かせろ」と
# 「すくい上げで浮かせろ」をほぼ対等に要求する形になる。
#
# NOTE: それでも direction (6.0) は超えない。方向ゲートを最上位に保つのは
#       walk_lob から一貫した設計 (踏みつけ / かすらせ exploit を塞ぐ構造)。
# NOTE: 単独では動かない想定。軸足が −0.42 のまま (脚が伸び切った姿勢) では
#       蹴り足を上へ振る余地が運動学的に無いので、kick_plant_foot のカリキュラムが
#       効いて初めてこの項が動く、という順序を見込んでいる。
# --------------------------------------------------------------------------- #
_FOOT_LIFT_WEIGHT = 4.0


def _restore_vision_ball_obs(cfg) -> None:
    """walk_lob が入れたガウス認識パイプラインを外し、``delayed_ball_pos_b`` に戻す。

    :class:`~..walk_lob.walk_lob_env_cfg.K1WalkLobEnvCfg` は ``__post_init__`` の
    最後で :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` を呼び、
    policy の ``prev_ball_pos`` を
    :func:`~..walk_kick.mdp.observations.noisy_ball_pos_b` (エピソードごとランダム
    遅延 + 30Hz サンプル&ホールド + ガウスジッタ) に差し替える。

    こちらの系列は代わりに :func:`enable_obs_delay` (連続遅延 + 一様ノイズ ±7 cm、
    fewa 実機実績準拠) を使うので、**2 つのパイプラインが混ざらないよう先に戻す**。
    戻さないと:

    * ``enable_obs_delay`` は認識パイプラインが載ったスロットを自動でスキップする
      (``_PIPELINE_BALL_POS_FUNCS`` ガード) ので、スロット 3 だけ連続遅延・
      スロット 12 だけガウスパイプラインという食い違った状態になる。
    * ``params`` にガウス側のキー (``jitter_std`` など) が残り、差し替え後の関数へ
      未知のキーワード引数として渡ってしまう。

    どちらのモデルを採るかは思想の違いで、ガウス側 (walk_lob) は「たまに大きく外す」
    裾を、一様+連続遅延側 (fewa) は実機で実績のある形をそれぞれ狙う。ここで後者に
    寄せるのは、履歴入力とセンサ遅延 DR を fewa 由来の一式で揃えるため。
    **ガウス側に戻すなら、この関数を呼ばず ``enable_obs_delay`` を
    ``sources=("body",)`` にすること** (そうすればボール 3 項には触らない)。
    """
    policy = cfg.observations.policy
    policy.ball_pos.func = mdp.delayed_ball_pos_b
    policy.ball_pos.params = {"delay_steps": _BALL_POS_DELAY, "dim": 3}
    policy.ball_pos.noise = Unoise(n_min=-0.02, n_max=0.02)
    policy.prev_ball_pos.func = mdp.delayed_ball_pos_b
    policy.prev_ball_pos.params = {"delay_steps": _BALL_POS_PREV_DELAY, "dim": 2}
    policy.prev_ball_pos.noise = Unoise(n_min=-0.02, n_max=0.02)


def _apply_plant_foot_curriculum(cfg) -> None:
    """``kick_plant_foot`` の ``lon_target`` と ``sigma_lon`` をアニールする。

    出発値を報酬項の params に直接書き込んでおくのが要点。カリキュラムは
    ``start_step`` 以前も ``start_value`` を書き戻すが、**最初の 1 回が走るまでは
    cfg の初期値がそのまま使われる**ので、初期値を −0.03 / 0.10 のままにしておくと
    その間だけ元の (届かない) 採点になる。

    定数の根拠は :data:`_PLANT_LON_TARGET_START` のコメント参照。
    """
    cfg.rewards.kick_plant_foot.params["lon_target"] = _PLANT_LON_TARGET_START
    cfg.rewards.kick_plant_foot.params["sigma_lon"] = _PLANT_SIGMA_LON_START

    cfg.curriculum.kick_plant_foot_lon_target = CurrTerm(
        func=mdp.linear_reward_param,
        params={
            "term_name": "kick_plant_foot",
            "param_name": "lon_target",
            "start_value": _PLANT_LON_TARGET_START,
            "end_value": _PLANT_LON_TARGET,
            "start_step": _PLANT_ANNEAL_START_ITER,
            "end_step": _PLANT_ANNEAL_END_ITER,
            "steps_per_iteration": _STEPS_PER_ITERATION,
        },
    )
    cfg.curriculum.kick_plant_foot_sigma_lon = CurrTerm(
        func=mdp.linear_reward_param,
        params={
            "term_name": "kick_plant_foot",
            "param_name": "sigma_lon",
            "start_value": _PLANT_SIGMA_LON_START,
            "end_value": _PLANT_SIGMA_LON,
            "start_step": _PLANT_ANNEAL_START_ITER,
            "end_step": _PLANT_ANNEAL_END_ITER,
            "steps_per_iteration": _STEPS_PER_ITERATION,
        },
    )


def _add_contact_height_reward(cfg) -> None:
    """``kick_contact_height`` (接触時の足裏高さ) を足す。

    既存のキック報酬とは **加算** で並べる (``kick_loft`` に掛けない)。項の内部は
    ``r_direction`` への乗算なので、方向ゲート・kick_done ゲート・胴体の正対を
    通過した蹴りにしか払われない。``sigma_direction`` は他のキック報酬と揃える
    (:data:`~..walk_lob.walk_lob_env_cfg._LOB_SIGMA_DIRECTION`)。

    weight のフェードインは他のキック報酬と同じ [0, 500] iteration。
    """
    cfg.rewards.kick_contact_height = RewTerm(
        func=mdp.kick_contact_height,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _LOB_SIGMA_DIRECTION,
            "ball_radius": _CONTACT_HEIGHT_BALL_RADIUS,
            "h_sat": _CONTACT_HEIGHT_SAT,
        },
    )
    cfg.curriculum.kick_contact_height_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_contact_height",
            "start_weight": 0.0,
            "end_weight": _CONTACT_HEIGHT_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": _KICK_FADE_IN_END_ITER,
            "steps_per_iteration": _STEPS_PER_ITERATION,
        },
    )


def _raise_foot_lift_weight(cfg) -> None:
    """``kick_foot_lift`` の最終重みを引き上げる (:data:`_FOOT_LIFT_WEIGHT`)。

    walk_lob 側はカリキュラム (``kick_foot_lift_weight``) で 0 → 終値へランプする
    構成なので、**報酬項の weight ではなくカリキュラムの ``end_weight`` を書き換える**。
    項側の weight を直接上げてもカリキュラムに毎ステップ上書きされて効かない。
    """
    cfg.curriculum.kick_foot_lift_weight.params["end_weight"] = _FOOT_LIFT_WEIGHT * _KICK_W_SCALE


def _apply_play_tweaks(cfg) -> None:
    """PLAY 共通の間引き。walk_lob の PLAY と同じ内容。

    ``_disable_ball_obs_jitter`` は **呼ばない**。この系列のボール観測は
    ガウスジッタではなく一様ノイズ + 連続遅延 (:func:`_restore_vision_ball_obs`
    と :func:`enable_obs_delay`) なので、``enable_corruption = False`` だけで
    ノイズが落ちる。遅延は観測パイプラインの構造なので PLAY でも残る。
    """
    cfg.scene.num_envs = 20
    cfg.scene.env_spacing = 4
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


# --------------------------------------------------------------------------- #
# Stage 1: 歩行のみ (凹凸地形 + 履歴)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobRoughWalkPhaseEnvCfg(K1WalkLobWalkPhaseEnvCfg):
    """Stage 1: 凹凸地形の上で通常の歩行だけを学習する。履歴入力版。

    観測グループを both_feet 版 (スロット 3 = ボール 3D 位置) に差し替えているので、
    継承元がボール由来スロットを歩行コマンドへ読み替える処理に **スロット 3 のぶんを
    足す** 必要がある。ボールはこの段でシーンごと消えているため、実体を読む関数を
    残すと落ちる。

    スロット 3 をゼロ埋めではなく歩行コマンド (vx, vy, 0) にするのは、継承元が
    ``prev_ball_pos`` に対してやっているのと同じ理由:「スロットが指す方へ歩く」という
    入力→挙動の対応を stage 2 と共通にしておくと歩容がそのまま転移する。
    スロット 3 と 12 は stage 2 でもほぼ同じ値 (1 フレーム違いのボール位置) なので、
    両方に同じ値を載せるのが素直な対応になる。

    センサ遅延 DR は ``sources=("body",)`` で **IMU / エンコーダだけ**。ボール 3 項は
    歩行コマンドに化けているので触らない。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- スロット 3 (ボール 3D 位置) を歩行コマンドに差し替える
        for _group in (self.observations.policy, self.observations.critic):
            _group.ball_pos.func = mdp.walk_command_xyz
            _group.ball_pos.params = {"command_name": "base_velocity"}

        _apply_rough_terrain(self)
        enable_obs_history(self)
        enable_obs_delay(self, _OBS_DELAY_MAX_S, sources=("body",))


@configclass
class K1WalkLobRoughWalkPhaseEnvCfg_PLAY(K1WalkLobRoughWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_play_tweaks(self)


# --------------------------------------------------------------------------- #
# Stage 2: ロブキック (凹凸地形 + 履歴 + 観測 DR + 当たり所の誘導)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobRoughEnvCfg(K1WalkLobEnvCfg):
    """Stage 2: 凹凸地形でロブキックを学習する。履歴入力・観測 DR つき。

    引き継ぎ元は Stage 1 (この系列) の checkpoint のみ。観測スロット 3 の意味も
    ネットワーク (履歴 CNN) も既存 walk_lob と違うので、``k1_walk_lob`` /
    ``k1_walk_lob_walk_phase`` の checkpoint は **形の上でも意味の上でも載らない**。

    ``__post_init__`` の順序に意味がある:

    1. ``super()``                     — walk_lob のロブ報酬一式 + ガウスボール観測
    2. :func:`_restore_vision_ball_obs` — ガウスパイプラインを外す (1 の後始末)
    3. 地形・報酬の変更
    4. :func:`enable_obs_history`      — 観測グループの構成が固まった後
    5. :func:`enable_obs_delay`        — 2 でパイプラインを外してあるので全項に掛かる
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- walk_lob が入れたガウス認識パイプラインを外す (5 と二重掛けになるため)
        _restore_vision_ball_obs(self)

        # -- 凹凸地形 (ボールの spawn_clearance もここで入る)
        _apply_rough_terrain(self)

        # -- 当たり所を下げるための 3 点
        _apply_plant_foot_curriculum(self)
        _add_contact_height_reward(self)
        _raise_foot_lift_weight(self)

        # -- 履歴入力とセンサ遅延 DR (内界センサ + 視覚)
        enable_obs_history(self)
        enable_obs_delay(self, _OBS_DELAY_MAX_S)


@configclass
class K1WalkLobRoughEnvCfg_PLAY(K1WalkLobRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_play_tweaks(self)
