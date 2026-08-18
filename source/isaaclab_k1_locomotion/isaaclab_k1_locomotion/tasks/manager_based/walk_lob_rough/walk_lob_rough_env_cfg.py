# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_lob の履歴入力版。平坦 / 凹凸の両方を持つ **3 段構成**。

``k1_walk_lob/2026-08-16_08-08-20/model_11600.pt`` を実機に載せたところ「一度だけ
大きく浮き、他の試行も浮きそうな傾向はある」という結果になった。その run の実測を
起点に「浮きの再現性」を取りにいくための系列。

なぜ 3 段なのか (2026-08-18 の失敗から)
---------------------------------------
最初は 2 段 (walk phase → lob) で組んで回したが、**stage 2 が立ち上がらなかった**。
``k1_walk_lob_rough/2026-08-18_04-42-43`` の実測:

    iteration    0   25   50  100  200  300  400
    eplen     13.7 25.6 28.5 26.3 25.4 24.6 24.7
    base_ht   0.71 1.00 1.00 1.00 0.99 0.99 0.99
    kick_rate 0.00 0.00 0.01 0.01 0.01 0.00 0.01

エピソードが 25 ステップ (0.5 秒) で ``base_height`` 終了し、400 iteration 経っても
改善しない。つまり「蹴らない」のではなく **蹴る前に立てていない**。

段の切り替え直後に一度崩れること自体は正常で、参照 run も同じ形で始まる。
違うのは **戻ってくるかどうか**::

    it            0   25   50  100  200  300  400
    both_feet  19.7 44.5 121  256  465  455  467     base_ht 0.59 → 0.011
    dual       47.2 56.2 62.5 202  412  431  432     base_ht 0.12 → 0.037
    walk_lob   13.7 25.6 28.5 26.3 25.4 24.6 24.7    base_ht 0.71 → 0.991  ← 戻らない

参照 2 つは 100-200 iteration で回復する。こちらだけ戻らない理由は **lob の報酬集合が
歩行ポリシーからブートストラップできないほど疎** であること:

* ``kick_velocity_scaled`` (「指令の速さでボールに当てろ」) を項ごと撤去してある。
  これは walk_kick では **最初にボールへ触りにいく動機を作っている項** で、
  loft / elevation は「当たった後の飛び方」しか見ないので、まだ一度も当てられない
  段階では全部 0 のまま勾配が出ない。
* 残る密な信号は ``approach_penalty`` (負) と ``walk_speed`` だけで、
  ``kick_pose_overshoot`` の −50 も含めて **負の圧力が先に効く**。

実際、既存の flat walk_lob で成功していた run
(``k1_walk_lob/2026-08-16_05-46-55`` 以降) も **walk phase からではなく、
既にボールを蹴れる checkpoint からの resume** で立ち上がっている
(``agent.yaml``: ``resume: true`` / ``load_run: 2026-08-16_03-16-39``)。
``train_walk_lob.sh`` の「loop_shoot / loop_pass の checkpoint から stage 2 だけ
始めても問題ない」という NOTE はこの経緯を指している。
**walk phase → lob の直行は平坦でも一度も成功していない。**

そこで間に **キック段** を挟む::

    Stage 1  walk phase        歩くだけ                         (ボール無し)
    Stage 2  kick              walk_kick の報酬集合で「当てる」   ← 新設
    Stage 3  lob               高さ特化の報酬集合で「上げる」

Stage 2 は :class:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg.K1WalkKickBothFeetEnvCfg`
(= walk_kick の報酬 + スロット 3 がボール 3D 位置の観測) をそのまま使う。
これは both_feet / dual で 2 回とも立ち上がりが確認できている構成そのもの。
**位相オフセット (両足キック化) だけは外す** (下の :func:`_disable_phase_offset`)。

地形について
------------
**平坦版 (Flat-*) と凹凸版 (Rough-*) の両方を登録してある。まず平坦で通すこと。**

凹凸地形 + ボールの組み合わせはこのリポジトリで一度も学習を通したことがない
(``k1_walk_kick_rough`` 系の log が存在しない)。段の切り替えが失敗している状態で
未検証の条件を重ねると切り分けができないので、

1. まず ``Flat-*`` の 3 段を通して kick_rate と apex が出ることを確認する
2. その checkpoint から ``Rough-*`` の stage 3 だけを fine-tune する

の順にする。歩行だけなら凹凸でも問題なく学習できることは確認済み
(``k1_walk_lob_rough_walk_phase``: eplen 962/1000、base_height 終了 6.4%)。

実測から分かったこと (2026-08-16 の flat run、iteration 9998-11624)
-------------------------------------------------------------------
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

2. **run 全体がプラトーしている。** it10150 以降、apex 0.402 → 0.409、
   foot_vz 0.755 → 0.806、elevation 23.7° → 23.7°。

3. **浮きはすくい上げで作られていない。** apex 0.425 m = 上昇 0.315 m ⇔ 打ち出し
   vz ≈ 2.49 m/s に対して ``foot_vz`` は 0.81 m/s。差分を作っているのは接触法線の
   向き、つまり「ボール中心より下に速い水平速度で当てている」ことの方。
   足裏高さ h での法線仰角は asin((0.11 − h)/0.11) なので、h = 0.083 なら 14°、
   実測の射出仰角 25° は 14° に foot_vz 分が乗った値として整合する。
   **仰角を 45-60° まで持っていくには h を 0.03 m 台まで下げる必要がある。**

3 段が通った後の実測 (2026-08-18、k1_walk_lob_hist/2026-08-18_09-07-17)
-----------------------------------------------------------------------
3 段構成で **立ち上がりは成功した** (kick_rate 0.998 / base_height 終了 0.003 /
``kick_finished`` で正常終了)。ただし apex は it1900 で頭打ちになった::

                  1000-1300  1300-1600  1600-1900  1900-2200  2200-2500  2500-2780
    apex_height      0.256      0.281      0.319      0.360      0.361      0.352
    elevation_deg    27.2       27.7       27.8       28.3       28.2       28.2
    kick_vel_ratio   0.447      0.464      0.509      0.534      0.529      0.513
    plant_lon       -0.381     -0.374     -0.378     -0.372     -0.368     -0.371

狙った 3 点のうち **効いたのは kick_foot_lift だけ** (foot_vz −0.55 → +0.34)。
``kick_plant_foot`` は目標側が 0.24 動いたのに実測は 0.01 しか動かず、f_lon が
0.996 → 0.433 と剥がれていくだけだった。``kick_contact_height`` は
sole_height 0.053 → 0.065 と逆行した。

**原因は ``r_stance`` = 0.25 との構造的衝突** (:data:`_R_STANCE_LOB` のコメント参照)。
歩行目標 G がボール後方 0.25 m より近づかないので、軸足は −0.37 前後で平衡する。
``kick_plant_foot`` (weight 0.6) は、``walk_speed`` (weight 1.5) が作ったこの壁を
より小さい重みで押し返そうとしていた。そして軸足が動かない限り「すくい上げ」と
「低い当たり」は運動学的にトレードオフになるので、重みで勝った方へ倒れるだけになる。

→ 2026-08-18 の修正: **壁の方を動かす** (r_stance 0.25 → 0.15)。あわせて軸足目標を
届く範囲へ緩め (−0.03 → −0.15、σ 0.10 → 0.15)、重みを当たり所側へ寄せた
(foot_lift 4.0 → 2.0、contact_height 2.0 → 3.0)。

walk_lob からの変更点
---------------------
1. **キック段 (stage 2) を挟む。** 上記のとおり。

2. **観測履歴 100 フレーム + HistoryCNN**
   (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)。全段。

3. **センサ遅延 DR** (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_delay`)。
   IMU / エンコーダは stage 1 から (``sources=("body",)``)、ボール観測 (視覚) は
   ボールが存在する stage 2 以降。

4. **観測スロット 3 を左足裏 → ボール 3D 位置に差し替える**。B-Human 原典の観測表では
   スロット 3 は Current Ball 3D Position で、walk_kick 系の ``sole_pos`` は
   評価表キャプション "Left Sole" の誤読だった。``sole_pos`` は joint_pos 12 次元から
   FK で完全に決まる冗長情報なので、差し替えは論文準拠と情報量の両面で正味プラス。

   **両足キック化 (位相オフセット {0, π}) と mirror loss は入れない。** 目的は apex
   高さであって両足で蹴ることではない (ユーザー判断 2026-08-18)。したがって
   ``kick_foot_right_frac`` は 1.0 付近に張り付いたままになる想定。

5. **``kick_plant_foot`` の目標と σ をカリキュラムで動かす** (stage 3 のみ)。
   固定目標では届かないことが実測で分かったので、**実測値の側から始めて目標へ引っ張る**。

6. **``kick_contact_height`` (新規) を足す** (stage 3 のみ)。接触時の足裏高さを直接
   下げさせる項。5 と表裏の関係で、あちらが原因側 (構え)、こちらが結果側 (当たり所)。
   線形ランプなので勾配が死なず、カリキュラム不要。

7. **重みを当たり所側へ寄せる** (stage 3 のみ)。``kick_contact_height`` 2.0 → 3.0、
   ``kick_foot_lift`` は walk_lob と同じ 2.0 のまま (一度 4.0 にして失敗した)。

8. **``r_stance`` を 0.25 → 0.15 に下げる** (stage 3 のみ、:func:`_set_r_stance`)。
   5-7 が動くための前提。上の実測セクション参照。

継承したまま変えないもの
------------------------
* ロブの報酬設計 (``kick_velocity_scaled`` 撤去 / ``vz_sat`` 5.0 / ``phi_sat`` 60° /
  σ_direction 0.6)。**ただし撤去が stage 3 でだけ効くようになった** のが今回の要点。
* ``disable_landing_shaping`` / ``rebalance_gait_vs_kick`` は **呼ばない**
  (dual 系 = fewa 由来のレシピで、walk_lob の歩容設計とは別系統)。
* ボール物性の DR。

  .. note::
     ボールの反発係数について。``soccer_ball`` の spawn material は
     restitution 0.6 / combine_mode ``average`` だが、地面とロボットの material は
     restitution 0.0 / combine_mode ``multiply`` で、``sim.physics_material`` も
     terrain のものが使われる。PhysX は 2 材質のうち **優先度の高い combine mode**
     (average < min < multiply < max) を採るので実効は ``multiply`` になり、
     ボール↔地面・ボール↔足はいずれも 0.6 × 0.0 = **0.0** になるはず。
     つまり ``ball_physics_material`` の DR (restitution 0.0-0.7) は実質効いておらず、
     すでに実機と同じ e≈0 で学習していることになる。上の実測 3 とも整合する。
     **sim を動かして確かめてはいないので、物理側は今回変更していない。**

学習の進め方
------------
既存の ``k1_walk_lob`` / ``k1_walk_lob_walk_phase`` の checkpoint は観測スロット 3 の
意味も actor の形も違うので流用できない。stage 1 から通すこと::

    ./scripts/rsl_rl/train_walk_lob_hist.sh              # 平坦 3 段 (まずこちら)
    TERRAIN=rough ./scripts/rsl_rl/train_walk_lob_hist.sh

凹凸版の stage 1 (``k1_walk_lob_rough_walk_phase``) は 2026-08-17 に 8000 iteration
学習済みで健全なので、凹凸で通すときはそれを ``WALK_CKPT`` に渡せば再学習は不要。

効果の見方
----------
まず ``Train/mean_episode_length`` と ``Episode_Termination/base_height``。
段の切り替え後 100-200 iteration で eplen が 400 以上へ戻らなければ、その段の
報酬集合が前段からブートストラップできていないということ (今回の失敗と同じ形)。

立ち上がった後は **まず ``plant_lon`` だけ見る**。r_stance を下げた効果が出ていれば
最初の 1000 iteration で −0.37 から手前 (0 側) へ動き始めるはず。**動かなければ
r_stance は原因ではなかった** ということなので、そこで止めて別を疑うこと
(この 1 本だけで切り分けが付くように 8 を独立した変更にしてある)。

そのうえで ``Metrics/kick_direction/`` の 4 つ:

* ``plant_lon``          : −0.37 から −0.15 側へ動くか (変更 5・8)
* ``sole_height_at_kick``: 0.083 から 0.03 台へ下がるか (変更 6)
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
    K1WalkKickBothFeetEnvCfg,
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
# ``step = common_step_counter // steps_per_iteration`` を使う。``common_step_counter``
# は env ステップごとに 1 増えるので、**1 iteration あたり ``num_steps_per_env``
# だけ進む**。この系列の RunnerCfg は ``num_steps_per_env = 48``。
#
# walk_kick 系のカリキュラムは全て ``steps_per_iteration = 24`` を渡しているので、
# **カリキュラムの 1 step = 0.5 iteration** になっている (48 // 24 = 2)。
# 既存の「end_step: 500」は iteration 250 で完了する、ということ。
#
# ここは既存と揃えて 24 のままにし (段の間で fade-in の速さが変わる方が害が大きい)、
# **iteration で書きたい定数は :func:`_iter` で変換する**。
#
# NOTE: 2026-08-18 の run で実測確認済み。it413 での
#       ``Curriculum/kick_plant_foot_lon_target/lon_target`` は −0.384 で、
#       (413·48//24 = 826) から計算した −0.3837 と一致する。
# --------------------------------------------------------------------------- #
_STEPS_PER_ITERATION = 24
_NUM_STEPS_PER_ENV = 48


def _iter(n: int) -> int:
    """iteration 数をカリキュラムの step 単位へ変換する。"""
    return n * (_NUM_STEPS_PER_ENV // _STEPS_PER_ITERATION)


# キック報酬のフェードイン (weight 0 → 最終値) が終わる iteration。
# 既存の walk_lob / walk_kick は end_step=500 (= iteration 250) なので、
# それと同じ時刻になるよう iteration で 250 と書く。
_KICK_FADE_IN_END_ITER = 250

# --------------------------------------------------------------------------- #
# 軸足配置 (kick_plant_foot) のカリキュラム (stage 3 のみ)
#
# **なぜ必要か**: 固定目標 (_PLANT_LON_TARGET = −0.03, _PLANT_SIGMA_LON = 0.10) では
# 実測 plant_lon = −0.42 に対して f_lon ≈ 5e-4 で、報酬も勾配もゼロだった
# (モジュール docstring の実測 1)。ガウスは「正解の位置に立ったら褒める」形なので、
# そこへ至る坂が無いと一切動かない。**目標の側を実測値から出発させて引っ張る。**
#
#   _PLANT_LON_TARGET_START = -0.42 : 2026-08-16 run の収束値。ここなら f_lon ≈ 1。
#   _PLANT_SIGMA_LON_START  = 0.25  : 出発時点の許容幅。stage 2 から引き継いだ直後の
#       plant_lon は −0.42 ちょうどではないので、σ を広めに取って出発点のばらつきを
#       覆う。0.25 なら −0.42 ± 0.25 = [−0.67, −0.17] で半値以上。
#   _PLANT_LON_TARGET_END   = -0.15 : **walk_lob の −0.03 から緩めた** (2026-08-18)。
#       −0.03 は「軸足の足箱中心をボール真横」という幾何から出た値だが、r_stance を
#       0.15 に下げてもそこまでは届かない見込み。**到達不能な目標を追わせると、
#       目標だけが逃げて f_lon が単調に剥がれ、policy から見て「何をしても損」の
#       信号になる** (実測 f_lon 0.996 → 0.433、このまま行くと 0.003)。まず届く範囲に
#       置いて、実際に追随したら次の run でさらに詰める。
#   _PLANT_SIGMA_LON_END    = 0.15  : 同じ理由で 0.10 から緩めた。終点で gap が残っても
#       報酬が完全には消えないようにする。
#
#   アニール区間 [500, 4000] iteration: 開始は「キック報酬のフェードインが終わり、
#   前段から引き継いだ蹴り方が stage 3 の報酬で一度落ち着く」ぶんの余裕を見た値。
#   終了 4000 は stage 3 を 15000 iteration 回す想定に対して前半で締め切る。
#
# NOTE: 目標と σ を **同時に** 動かす。目標だけ動かすと σ=0.25 のまま緩い採点が
#       残り、σ だけ絞ると届かないまま裾の外に落ちる (元の失敗の再現)。
# NOTE: 効いているかは Metrics/kick_direction/plant_lon が −0.42 から離れて
#       いくかで見る。Curriculum/kick_plant_foot/lon_target に目標側の値が出るので、
#       2 本を重ねると「目標に追随できているか」がそのまま読める。
# --------------------------------------------------------------------------- #
_PLANT_LON_TARGET_START = -0.42
_PLANT_LON_TARGET_END = -0.15
_PLANT_SIGMA_LON_START = 0.25
_PLANT_SIGMA_LON_END = 0.15
_PLANT_ANNEAL_START_ITER = 500
_PLANT_ANNEAL_END_ITER = 4000

# --------------------------------------------------------------------------- #
# キック立ち位置の半径 r_stance [m] (stage 3 のみ 0.25 → 0.15)
#
# **2026-08-18 の run (k1_walk_lob_hist/2026-08-18_09-07-17) で分かった本丸。**
#
# 歩行目標 G は :mod:`..walk_kick.mdp.kick_state` で
#
#     reach = clamp(alpha * dist_robot_ball, min=r_stance, max=0.5)
#     G     = ball_pos - reach * kick_dir
#
# と定義されており、**ボール後方 r_stance より近づかない**。``walk_speed`` (weight 1.5)
# がそこへ引き、到達すると p_walk が飽和して前進の圧力が消える。base がボール後方
# 0.25 m に落ち着くと軸足 (足首) はさらに後ろに来るので、plant_lon は −0.37 前後で
# 平衡する。実測がまさにこの値だった。
#
# つまり ``kick_plant_foot`` (weight 0.6) は **歩行報酬が作った立ち位置の壁を、
# より小さい重みで押し返そうとしていた**。カリキュラムで目標を動かしても実測は
# 2780 iteration で 0.01 しか動かず (目標側は 0.24 動いた)、f_lon が 0.996 → 0.433 と
# 剥がれていくだけだった。**目標ではなく壁の方を動かす必要がある。**
#
# 0.15 は「軸足をボールの真横に置く」姿勢から逆算した値ではなく、**壁を 10 cm 手前へ
# 動かす**という増分の指定。r_stance を下げすぎるとキック前にボールへ足が触れてしまう
# (``reset_ball`` の dist_range 下限と同じ制約) ので、まず 1 段だけ動かして
# plant_lon が追随するかを見る。
#
# .. warning::
#    ``r_stance`` は **報酬 9 項・終了条件 (kick_finished)・コマンド (base_velocity)**
#    の 3 か所に配られている。``kick_state`` は同一ステップ内で最初の呼び出しだけが
#    状態を更新するので、**1 か所でも古い値が残っていると、その値が P_kick と G を
#    決めてしまう**。IsaacLab の step 順では termination が reward より先に走るため、
#    報酬側だけ変えても無意味になる。:func:`_set_r_stance` が 3 か所すべてを揃え、
#    取りこぼしがあれば例外を投げる。
# --------------------------------------------------------------------------- #
_R_STANCE_LOB = 0.15

# --------------------------------------------------------------------------- #
# 接触高さ報酬 (kick_contact_height) の定数 (stage 3 のみ)
#
# f_low = clamp((ball_radius − h) / (ball_radius − h_sat), 0, 1)。導出と根拠は
# :func:`~..walk_kick.mdp.rewards.kick_contact_height` の docstring 参照。
#
#   _CONTACT_HEIGHT_BALL_RADIUS = 0.11 : ボール半径。ここに当てると法線が水平 = 0 点。
#   _CONTACT_HEIGHT_SAT         = 0.03 : 満点になる足裏高さ。法線仰角 45° 相当
#       (asin((0.11−0.03)/0.11) = 47°)。**これより下げても得をしない** ようにして、
#       つま先を地面へ掻き込むだけの解に動機を与えない。
#   _CONTACT_HEIGHT_WEIGHT      = 3.0  : **2.0 から上げた** (2026-08-18)。
#       direction (6.0) / loft (5.0) / elevation (5.0) より下、という原則は保つが、
#       kick_foot_lift (2.0) より **上** に置く。実測で仰角を作っているのは接触法線
#       (= 当たり所の高さ) であって足の鉛直速度ではなく、foot_lift を 4.0 にした run
#       では sole_height が 0.053 → 0.065 と逆行して apex がむしろ下がったため。
#
# NOTE: 線形ランプなので実測 h = 0.083 でも f_low = 0.34 が付き、勾配も生きている。
#       kick_plant_foot と違ってカリキュラムは要らない。
# --------------------------------------------------------------------------- #
_CONTACT_HEIGHT_BALL_RADIUS = 0.11
_CONTACT_HEIGHT_SAT = 0.03
_CONTACT_HEIGHT_WEIGHT = 3.0

# --------------------------------------------------------------------------- #
# すくい上げ (kick_foot_lift) の重み。walk_lob と同じ 2.0 に戻す (stage 3 のみ)。
#
# 一度 4.0 まで上げた (2026-08-18 の run) が、**上げすぎだった**。foot_vz は狙いどおり
# −0.55 → +0.34 と上向きに転じた一方で、``sole_height_at_kick`` が 0.053 → 0.065 と
# **逆行**し、apex は 0.36 で頭打ち・旧 flat run (0.426) より低くなった。
#
# 原因は **すくい上げと低い当たりが運動学的にトレードオフ** であること。軸足がボールの
# 37 cm 後ろにある姿勢ではつま先をボールの下へ入れられないので、「足を上向きに動かす」と
# 「接触点を下げる」を同時には満たせない。重みで勝っている方に倒れるだけで、4.0 は
# foot_lift 側に倒しただけだった。
#
# トレードオフを解くのは軸足の位置 (:data:`_R_STANCE_LOB`) であって重み配分ではない。
# ここは 2.0 に戻し、**当たり所側 (_CONTACT_HEIGHT_WEIGHT = 3.0) をやや上に置く**。
#
# NOTE: それでも direction (6.0) は超えない。方向ゲートを最上位に保つのは
#       walk_lob から一貫した設計 (踏みつけ / かすらせ exploit を塞ぐ構造)。
# --------------------------------------------------------------------------- #
_FOOT_LIFT_WEIGHT = 2.0


# --------------------------------------------------------------------------- #
# 共通ヘルパー
# --------------------------------------------------------------------------- #
def _disable_phase_offset(cfg) -> None:
    """両足キック化 (歩行位相の初期オフセット) を外す。

    stage 2 の継承元 :class:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg.K1WalkKickBothFeetEnvCfg`
    は ``randomize_phase_offset`` (mode="reset"、{0, π} の二値) を入れる。あちらの
    目的は「両足で蹴れるようにする」ことだが、この系列の目的は apex 高さなので
    入れない (ユーザー判断 2026-08-18)。両足を学ばせるぶん学習が重くなるのを避ける。

    stage 1 (walk phase) は :class:`~..walk_lob.walk_lob_env_cfg.K1WalkLobWalkPhaseEnvCfg`
    を継承しており位相オフセットを入れないので、ここを呼ぶ必要はない。段をまたいで
    「常にオフセット無し」で揃うことになる。
    """
    cfg.events.randomize_phase_offset = None


def _walk_phase_ball_pos_slot(cfg) -> None:
    """walk phase 用にスロット 3 (ボール 3D 位置) を歩行コマンドへ差し替える。

    ボールはこの段でシーンごと消えているため、実体を読む関数を残すと落ちる。
    ゼロ埋めではなく歩行コマンド (vx, vy, 0) にするのは、継承元が ``prev_ball_pos``
    に対してやっているのと同じ理由:「スロットが指す方へ歩く」という入力→挙動の
    対応を次段と共通にしておくと歩容がそのまま転移する。
    """
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.ball_pos.func = mdp.walk_command_xyz
        group.ball_pos.params = {"command_name": "base_velocity"}


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

    **stage 3 (lob) だけで必要**。stage 2 の継承元 (both_feet) はガウスパイプラインを
    入れないので呼ばなくてよいが、呼んでも冪等なので害はない。

    ガウス側に戻すなら、この関数を呼ばず ``enable_obs_delay`` を
    ``sources=("body",)`` にすること (そうすればボール 3 項には触らない)。
    """
    policy = cfg.observations.policy
    policy.ball_pos.func = mdp.delayed_ball_pos_b
    policy.ball_pos.params = {"delay_steps": _BALL_POS_DELAY, "dim": 3}
    policy.ball_pos.noise = Unoise(n_min=-0.02, n_max=0.02)
    policy.prev_ball_pos.func = mdp.delayed_ball_pos_b
    policy.prev_ball_pos.params = {"delay_steps": _BALL_POS_PREV_DELAY, "dim": 2}
    policy.prev_ball_pos.noise = Unoise(n_min=-0.02, n_max=0.02)


def _set_r_stance(cfg, value: float) -> None:
    """キック立ち位置の半径 ``r_stance`` を **配られている全箇所** で揃える。

    ``r_stance`` は「理想キック立ち位置 P_kick をボール後方どれだけに置くか」で、
    :func:`~..walk_kick.mdp.kick_state.kick_state` が P_kick と歩行目標 G を作るのに
    使う。値の意味と 0.15 を選んだ理由は :data:`_R_STANCE_LOB` のコメント参照。

    **なぜ「全箇所」でなければならないか**: ``kick_state`` は ``common_step_counter``
    でステップ境界を見て **同一ステップ内では最初の呼び出しだけが状態を更新する**。
    2 番目以降の呼び出しは引数を無視してキャッシュを返す。したがって値がばらつくと
    「そのステップで最初に評価されたマネージャの値」が P_kick と G を決める。
    IsaacLab の step 順は termination → reward → command なので、**報酬側だけ書き換えても
    ``kick_finished`` の古い値が毎ステップ勝つ** ことになり、まったく効かない。

    配られている先は 3 種類:

    1. 報酬項 (``_KICK_STATE_PARAMS`` を展開している全項)
    2. 終了条件 ``kick_finished``
    3. コマンド ``base_velocity`` (:class:`~..walk_kick.mdp.commands.BallFollowVelocityCommandCfg`
       のフィールド。params ではなく cfg 直下の属性)

    取りこぼしを黙って通さないよう、書き換えた箇所を数えて検算する。将来キック報酬を
    足したときにここを通らないと例外で気づける。

    NOTE: **アニールにはしていない。** 3 つのマネージャに跨る値を毎ステップ書き換える
          カリキュラムは、上の「最初の呼び出しが勝つ」性質と噛み合わせると壊れやすい
          (書き換えの順序とマネージャの評価順序の両方に依存する)。段の頭で 1 度だけ
          決め打つ方が、env.yaml に残って検証もできる。
    """
    n = 0
    for term_name in dir(cfg.rewards):
        if term_name.startswith("_"):
            continue
        term = getattr(cfg.rewards, term_name, None)
        params = getattr(term, "params", None)
        if isinstance(params, dict) and "r_stance" in params:
            params["r_stance"] = value
            n += 1
    if cfg.terminations.kick_finished is not None:
        cfg.terminations.kick_finished.params["r_stance"] = value
        n += 1
    cfg.commands.base_velocity.r_stance = value
    n += 1

    # 2026-08-18 時点の内訳: 報酬 9 項 + kick_finished + base_velocity = 11。
    # 報酬項の増減で変わりうるので下限だけ見る (0 や 1 で通り抜けるのを防ぐのが目的)。
    if n < 5:
        raise RuntimeError(
            f"_set_r_stance: 書き換え先が {n} 箇所しかない。kick_state を共有する"
            " 報酬 / 終了条件 / コマンドの構成が想定と違う (揃っていないと"
            " P_kick と G が古い値で決まる)。"
        )


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

    for param_name, start, end in (
        ("lon_target", _PLANT_LON_TARGET_START, _PLANT_LON_TARGET_END),
        ("sigma_lon", _PLANT_SIGMA_LON_START, _PLANT_SIGMA_LON_END),
    ):
        setattr(
            cfg.curriculum,
            f"kick_plant_foot_{param_name}",
            CurrTerm(
                func=mdp.linear_reward_param,
                params={
                    "term_name": "kick_plant_foot",
                    "param_name": param_name,
                    "start_value": start,
                    "end_value": end,
                    "start_step": _iter(_PLANT_ANNEAL_START_ITER),
                    "end_step": _iter(_PLANT_ANNEAL_END_ITER),
                    "steps_per_iteration": _STEPS_PER_ITERATION,
                },
            ),
        )


def _add_contact_height_reward(cfg) -> None:
    """``kick_contact_height`` (接触時の足裏高さ) を足す。

    既存のキック報酬とは **加算** で並べる (``kick_loft`` に掛けない)。項の内部は
    ``r_direction`` への乗算なので、方向ゲート・kick_done ゲート・胴体の正対を
    通過した蹴りにしか払われない。``sigma_direction`` は他のキック報酬と揃える
    (:data:`~..walk_lob.walk_lob_env_cfg._LOB_SIGMA_DIRECTION`)。
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
            "end_step": _iter(_KICK_FADE_IN_END_ITER),
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
    """PLAY 共通の間引き。

    ``_disable_ball_obs_jitter`` は **呼ばない**。この系列のボール観測はガウスジッタ
    ではなく一様ノイズ + 連続遅延 (:func:`_restore_vision_ball_obs` と
    :func:`enable_obs_delay`) なので、``enable_corruption = False`` だけでノイズが
    落ちる。遅延は観測パイプラインの構造なので PLAY でも残る。
    """
    cfg.scene.num_envs = 20
    cfg.scene.env_spacing = 4
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


# --------------------------------------------------------------------------- #
# Stage 1: 歩行のみ (履歴入力)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobHistWalkPhaseEnvCfg(K1WalkLobWalkPhaseEnvCfg):
    """Stage 1 (平坦): ボール無しで歩行だけを学習する。履歴入力版。

    観測グループを both_feet 版 (スロット 3 = ボール 3D 位置) に差し替えているので、
    継承元がボール由来スロットを歩行コマンドへ読み替える処理に **スロット 3 のぶんを
    足す** 必要がある (:func:`_walk_phase_ball_pos_slot`)。

    センサ遅延 DR は ``sources=("body",)`` で **IMU / エンコーダだけ**。ボール 3 項は
    歩行コマンドに化けているので触らない。
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        _walk_phase_ball_pos_slot(self)
        enable_obs_history(self)
        enable_obs_delay(self, _OBS_DELAY_MAX_S, sources=("body",))


@configclass
class K1WalkLobHistWalkPhaseEnvCfg_PLAY(K1WalkLobHistWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)


@configclass
class K1WalkLobRoughWalkPhaseEnvCfg(K1WalkLobHistWalkPhaseEnvCfg):
    """Stage 1 (凹凸): 平坦版との差は地形だけ。

    2026-08-17 に 8000 iteration 学習済みで健全 (eplen 962/1000、base_height 終了
    6.4%)。凹凸で通すときはその checkpoint を使い回せる。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_rough_terrain(self)


@configclass
class K1WalkLobRoughWalkPhaseEnvCfg_PLAY(K1WalkLobRoughWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)


# --------------------------------------------------------------------------- #
# Stage 2: キック (ブートストラップ段)
#
# walk_kick の報酬集合 (kick_velocity_scaled を **含む**) で「ボールに当てる」ことを
# 先に覚えさせる段。lob の報酬集合は当たった後の飛び方しか見ないので、この段が無いと
# 歩行ポリシーから立ち上がらない (モジュール docstring 参照)。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobHistKickEnvCfg(K1WalkKickBothFeetEnvCfg):
    """Stage 2 (平坦): 限定レンジのキックを学習する。履歴入力版。

    継承元を both_feet 側にしてあるので観測スロット 3 は既にボール 3D 位置。
    **位相オフセット (両足キック化) だけ外す** (:func:`_disable_phase_offset`)。
    この構成は both_feet / dual の stage 2 で 2 回とも立ち上がりが確認できている
    (100-200 iteration で eplen が 400 以上へ戻る)。

    引き継ぎ元は Stage 1 (この系列) の checkpoint。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _disable_phase_offset(self)
        enable_obs_history(self)
        enable_obs_delay(self, _OBS_DELAY_MAX_S)


@configclass
class K1WalkLobHistKickEnvCfg_PLAY(K1WalkLobHistKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)


@configclass
class K1WalkLobRoughKickEnvCfg(K1WalkLobHistKickEnvCfg):
    """Stage 2 (凹凸): 平坦版との差は地形だけ。

    .. warning::
       凹凸地形 + ボールの組み合わせはこのリポジトリで一度も学習を通していない
       (``k1_walk_kick_rough`` 系の log が存在しない)。**まず平坦で 3 段通すこと。**
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_rough_terrain(self)


@configclass
class K1WalkLobRoughKickEnvCfg_PLAY(K1WalkLobRoughKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)


# --------------------------------------------------------------------------- #
# Stage 3: ロブ (最終段)
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLobHistEnvCfg(K1WalkLobEnvCfg):
    """Stage 3 (平坦): 高さ特化のロブキック。履歴入力・観測 DR・当たり所の誘導つき。

    引き継ぎ元は Stage 2 (この系列) の checkpoint。**Stage 1 から直接は繋がない**
    (それが 2026-08-18 の失敗。モジュール docstring 参照)。

    ``__post_init__`` の順序に意味がある:

    1. ``super()``                     — walk_lob のロブ報酬一式 + ガウスボール観測
    2. :func:`_restore_vision_ball_obs` — ガウスパイプラインを外す (1 の後始末)
    3. 報酬の変更 (当たり所を下げる 3 点)
    4. :func:`enable_obs_history`      — 観測グループの構成が固まった後
    5. :func:`enable_obs_delay`        — 2 でパイプラインを外してあるので全項に掛かる
    """

    observations: K1WalkKickBothFeetObservationsCfg = K1WalkKickBothFeetObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- walk_lob が入れたガウス認識パイプラインを外す (5 と二重掛けになるため)
        _restore_vision_ball_obs(self)

        # -- 立ち位置の壁を 10 cm 手前へ動かす (これが効かないと下の 3 点も動かない)
        _set_r_stance(self, _R_STANCE_LOB)

        # -- 当たり所を下げるための 3 点
        _apply_plant_foot_curriculum(self)
        _add_contact_height_reward(self)
        _raise_foot_lift_weight(self)

        # -- 履歴入力とセンサ遅延 DR (内界センサ + 視覚)
        enable_obs_history(self)
        enable_obs_delay(self, _OBS_DELAY_MAX_S)


@configclass
class K1WalkLobHistEnvCfg_PLAY(K1WalkLobHistEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)


@configclass
class K1WalkLobRoughEnvCfg(K1WalkLobHistEnvCfg):
    """Stage 3 (凹凸): 平坦版との差は地形だけ。

    平坦 3 段が通ってから、その checkpoint を ``--load_pretrained`` して
    **この段だけ fine-tune する** 使い方を想定している。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_rough_terrain(self)


@configclass
class K1WalkLobRoughEnvCfg_PLAY(K1WalkLobRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)
