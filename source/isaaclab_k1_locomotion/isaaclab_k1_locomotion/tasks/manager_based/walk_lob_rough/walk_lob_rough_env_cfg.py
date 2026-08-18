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
**本線は凹凸 (Rough-*)。** sim2real が目的なので凹凸で通す
(2026-08-18 のユーザー指示「roughかつhistoryで歩行から学習したい」)。
``train_walk_lob_hist.sh`` の ``TERRAIN`` 既定も ``rough``。

平坦版 (Flat-*) も同じ 3 段を登録してあるが、これは **切り分け専用**。凹凸地形 +
ボールの組み合わせはこのリポジトリで学習を通した実績が無い (``k1_walk_kick_rough``
系の log が存在しない) ので、凹凸で立ち上がらなかったときに「地形のせいか、報酬設計の
せいか」を分けるために平坦で同じ段を回す、という使い方をする。**既定にはしない。**

歩行だけなら凹凸でも問題なく学習できることは確認済み
(``k1_walk_lob_rough_walk_phase``: eplen 962/1000、base_height 終了 6.4%)。

観測・行動・ネットワークは地形に依存しないので、**平坦と凹凸の checkpoint は
相互にそのまま載る**。平坦で作った段を凹凸のウォームスタートに使ってよい。

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
立ち上がりには成功した (kick_rate 0.998 / base_height 終了 0.003 /
``kick_finished`` で正常終了)。

.. warning::
   **この run では「段の追加」と「地形 rough → flat」を同時に変えてしまったので、
   立ち上がった原因がどちらかは分離できていない。** 失敗した 2 段 run
   (``k1_walk_lob_rough/2026-08-18_04-42-43``) は rough、この 3 段 run は flat。
   「3 段構成が効いた」とは言えず、**rough + ボールの組み合わせが立ち上がりを
   壊していた可能性が同じだけ残っている** (この組み合わせはリポジトリで学習を
   通した実績が無い)。切り分けは「3 段 × rough」を 1 本回せば付く: 立ち上がれば
   効いたのは段、立ち上がらなければ地形。

ただし apex は it1900 で頭打ちになった::

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

5. **``kick_foot_lift`` の重みを 2.0 → 6.0 に上げる** (stage 3 のみ)。
   **stage 3 で apex を伸ばす施策はこれ 1 本に絞ってある。** 理由は下の「反証」節。

6. **``kick_plant_foot`` を項ごと撤去する** (stage 3 のみ、:func:`_remove_plant_foot_reward`)。
   カリキュラムと対で消す (項だけ消すと ``linear_reward_weight`` が存在しない term を
   引きにいって落ちる)。

反証: apex を伸ばす 3 つの仮説はすべて外れた (2026-08-18)
---------------------------------------------------------
軸足を前へ・当たり所を下げる・立ち位置の壁を動かす、の 3 つを試して **3 つとも
apex を伸ばせなかった**。同 iteration (1700-2000) で並べるとこうなる::

    run                        apex   上昇   仰角  v_ratio  foot_vz  sole_h  plant_lon
    旧 flat lob (it11500)     0.426  0.316  24.4°   0.697   +0.814   0.085    -0.422
    hist 前回                 0.340  0.230  27.9°   0.525   +0.168   0.062    -0.378
    hist 今回                 0.234  0.124  27.1°   0.401   -0.317   0.050    -0.365

**当たり所 (sole_h) を下げた run ほど apex が低い**という逆相関になっている。
仰角は確かに 24.4° → 27.1° と上がったが、ボール速度がそれ以上に落ちた。

3 点は次の関係にほぼ完全に乗る (正規化誤差 1% 以内)::

    apex 上昇 ∝ (kick_vel_ratio · sin φ)²

    (v·sinφ)²  正規化   rise 正規化
      0.0829    1.000       1.000
      0.0602    0.726       0.726
      0.0335    0.404       0.393

仰角の変動幅 (24-28°) よりボール速度の変動幅 (0.40-0.70) の方がずっと大きいので、
**apex は実質ボール速度が支配している**。低く当てるには立ち位置を詰めるしかなく、
それがスイング長 = 速度を削るので、当たり所を狙う施策は構造的に自滅する。

速度と競合せずに仰角を稼げる唯一の量が ``foot_vz`` (接触の瞬間の足の鉛直速度) で、
スイングを短くせず足首・膝の軌道で上向き成分を足す。apex 最高の run が ``foot_vz``
最大 (+0.814) だったことともつじつまが合う。だから stage 3 は
``kick_foot_lift`` 一本に絞ってある。個々の反証の詳細と、再挑戦するときの
出発点・落とし穴は :data:`_FOOT_LIFT_WEIGHT` の手前のコメントブロック参照。

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

    ./scripts/rsl_rl/train_walk_lob_hist.sh              # 凹凸 3 段 (既定・本線)
    TERRAIN=flat ./scripts/rsl_rl/train_walk_lob_hist.sh  # 平坦 (切り分け用)

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

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import _apply_rough_terrain, _KICK_W_SCALE
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
from ..walk_lob.walk_lob_env_cfg import K1WalkLobEnvCfg, K1WalkLobWalkPhaseEnvCfg

# --------------------------------------------------------------------------- #
# カリキュラムの時間単位についての注意 (この段では今のところ使っていない)
#
# ``linear_reward_weight`` / ``linear_reward_param`` は
# ``step = common_step_counter // steps_per_iteration`` を使う。``common_step_counter``
# は env ステップごとに 1 増えるので **1 iteration あたり ``num_steps_per_env`` だけ
# 進む**。この系列の RunnerCfg は ``num_steps_per_env = 48`` である一方、walk_kick 系の
# カリキュラムは全て ``steps_per_iteration = 24`` を渡しているので、
# **カリキュラムの 1 step = 0.5 iteration** になっている (48 // 24 = 2)。
# 継承している ``end_step: 500`` は **iteration 250** で完了する、ということ。
#
# 2026-08-18 の run で実測確認済み: it413 での
# ``Curriculum/kick_plant_foot_lon_target/lon_target`` が −0.384 で、
# (413·48//24 = 826) から計算した −0.3837 と一致した。
#
# **この段に新しくカリキュラムを足すときは、iteration で書きたい値を 2 倍すること。**
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 反証済みの 3 施策 (2026-08-18)。**定数だけ残して、どれも使っていない。**
#
# stage 3 で apex を伸ばすために 3 つ試して、**3 つとも外れた**。同じ道を二度通らない
# よう、値と反証の内容をここに残す。
#
# 1. **軸足を前へ (kick_plant_foot + そのカリキュラム)**
#    目標を実測 −0.42 から −0.15 へ動かすカリキュラムを入れたが、``plant_lon`` は
#    2 run 合わせて 4500 iteration でほぼ動かなかった (−0.378 → −0.365)。
#    カリキュラムが実測を追い越した後は **むしろ後退** した (−0.345 → −0.364)。
#    → ``plant_lon`` は位置報酬で動く量ではない。決めているのは歩容 (歩幅と接触が
#      起きる位相) の方で、報酬で押しても戻される。動かしたいなら歩容側の話になる。
#
# 2. **立ち位置の壁を動かす (r_stance 0.25 → 0.15)**
#    1 の原因を「歩行目標 G がボール後方 r_stance より近づかないこと」と読んで
#    10 cm 動かしたが、``plant_lon`` は **1.3 cm しか動かなかった** (−0.378 → −0.365)。
#    代わりにスイングが短くなって ``kick_vel_ratio`` が 0.525 → 0.401 へ落ちた。
#    → 壁ではなかった。定数は残してあるが :func:`_set_r_stance` を呼んでいない。
#
# 3. **当たり所を下げる (kick_contact_height)**
#    ``sole_height_at_kick`` は狙いどおり 0.062 → 0.050 に下がった。**しかし apex は
#    0.340 → 0.234 に下がった。** 低く当てるには立ち位置を詰めるしかなく、それが
#    スイング長 = ボール速度を削る。3 run を並べると当たり所が低い run ほど apex が
#    低いという逆相関になっている。
#    → 「当たり所を下げれば仰角が上がって apex が上がる」という仮説の反証。
#
# 3 run から出た定量則 (モジュール docstring の表を参照)::
#
#     apex 上昇 ∝ (kick_vel_ratio · sin φ)²      ← 3 点で正規化誤差 1% 以内
#
# 仰角の変動幅 (24-28°) よりボール速度の変動幅 (0.40-0.70) の方がずっと大きいので、
# **apex は実質ボール速度が支配している**。速度を削る施策は全て逆効果になる。
# 速度と競合せずに仰角を稼げるのは ``kick_foot_lift`` (すくい上げ) だけで、
# スイングを短くせず足首・膝の軌道で上向き成分を足すため。
#
# 試した値 (再挑戦するときの出発点として):
#
#   1. lon_target −0.42 → −0.15 / sigma_lon 0.25 → 0.15、iteration 500-4000 で線形
#      (``mdp.linear_reward_param``)。項の weight は 2.0 × _KICK_W_SCALE。
#   2. r_stance 0.25 → 0.15。
#   3. kick_contact_height: ball_radius 0.11 / h_sat 0.03 / weight 2.0-3.0。
#      報酬関数 ``mdp.kick_contact_height`` は残してある (他タスクから使える)。
#
# .. warning::
#    2 を再挑戦するときの落とし穴。``r_stance`` は **報酬 9 項だけでなく
#    ``terminations.kick_finished.params`` と ``commands.base_velocity``
#    (params ではなく cfg 直下の属性) にも配られていて、合計 11 箇所ある**。
#    ``kick_state`` は ``common_step_counter`` でステップ境界を見て
#    **同一ステップ内では最初の呼び出しだけが状態を更新する**ので、1 箇所でも古い値が
#    残るとその値が P_kick と G を決める。IsaacLab の step 順は
#    termination → reward → command なので、**報酬側だけ書き換えても
#    ``kick_finished`` の値が毎ステップ勝ち、まったく効かない**。
#    2026-08-18 の run では逆に、報酬項を足す前に一括設定したせいで
#    後から作られた ``kick_contact_height`` だけ 0.25 が残った (実害は無かったが、
#    終了条件を外していたら壊れていた)。**書き換えるなら全項を足し終えた後に、
#    11 箇所すべてを一度に。**
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# すくい上げ (kick_foot_lift) の重み。walk_lob の 2.0 から引き上げる (stage 3 のみ)。
#
# **stage 3 で唯一残っている当たりの施策。** 上の「反証済みの 3 施策」のとおり、
# 軸足も当たり所も apex を伸ばせなかった一方で、apex は
#
#     apex 上昇 ∝ (kick_vel_ratio · sin φ)²
#
# にほぼ完全に従い、ボール速度が支配していることが 3 run で分かった。速度を削らずに
# 仰角を上げられるのは **接触の瞬間に足自身が上へ動いていること** だけで、これは
# スイングを短くしないので速度と競合しない。実際 apex 最高の run
# (旧 flat lob、0.426) が ``foot_vz`` +0.814 で最大だった。
#
#   6.0 : loft (5.0) / elevation (5.0) を **超える** 水準。ここまで上げるのは、
#       この項が「浮かせ方の指定」ではなく **唯一の浮かせる手段** に格上げされた
#       ため。過去 2 run の実測は 4.0 で foot_vz +0.17、2.0 で −0.32 だったので、
#       正へ十分振らせるには 4.0 でも足りていない。
#   それでも direction (6.0) と同着まで。方向ゲートより上には置かない
#       (踏みつけ / かすらせ exploit を塞ぐ構造なので、ここは崩さない)。
#
# NOTE: ``vz_foot_sat`` は 2.0 のまま (walk_lob 既定)。実測 foot_vz が飽和に近づいたら
#       上げること。今は負値なので飽和の心配は無い。
# --------------------------------------------------------------------------- #
_FOOT_LIFT_WEIGHT = 6.0


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


def _remove_plant_foot_reward(cfg) -> None:
    """``kick_plant_foot`` を項ごと撤去する (カリキュラムも対で消す)。

    walk_lob が ``__post_init__`` で報酬項とカリキュラム
    (``kick_plant_foot_weight``) の両方を登録するので、**両方消すこと**。
    項だけ消してカリキュラムを残すと ``linear_reward_weight`` が存在しない term を
    ``reward_manager.get_term_cfg`` で引きにいって落ちる (walk_lob が
    ``kick_velocity_scaled`` で同じ注意書きを残している)。

    撤去する理由は「反証済みの 3 施策」の 1 番。``plant_lon`` は 2 run 合わせて
    4500 iteration 動かず、カリキュラムが実測を追い越した後は後退した。歩容が決めて
    いる量なので位置報酬では押せない。**報われない項を置いておくと、他の項との
    トレードオフで純粋に足を引っ張る** (実際 f_lon が剥がれていく過程は policy から
    見て「何をしても損」の信号になる)。

    NOTE: ``plant_lat`` (横方向) の方は目標 0.19 に対して実測 0.17 前後で最初から
          合っており、そもそも押す必要が無かった。前後方向だけが問題だった。
    """
    cfg.rewards.kick_plant_foot = None
    cfg.curriculum.kick_plant_foot_weight = None


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

        # -- 効かなかった軸足誘導を撤去し、すくい上げに一本化する
        #
        # ここで **やっていないこと** を明示しておく (「反証済みの 3 施策」参照):
        #   * kick_contact_height は足さない (当たり所を下げると速度が落ちて apex が下がる)
        #   * _set_r_stance は呼ばない        (r_stance は 0.25 のまま = walk_lob 既定)
        _remove_plant_foot_reward(self)
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
