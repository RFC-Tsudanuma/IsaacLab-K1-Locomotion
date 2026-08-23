# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ロブキック (軸足の踏み込みつき) — 観測履歴で通す 3 段構成。

    Stage 1  歩行のみ (flat, 履歴 actor)            k1_walk_lob_plant_walk_phase
    Stage 2  flat で「浮かせる蹴り」を発見・収束     k1_walk_lob_plant
    Stage 3  rough + ボール物性 DR の拡大            k1_walk_lob_plant_rough

    ./scripts/rsl_rl/train_walk_lob_plant.sh

目的の優先順位 (ユーザー指示 2026-08-23)
----------------------------------------
1. **浮かせることが最優先。** ``kick_loft`` / ``kick_elevation`` を「壊れない範囲で
   最強」に置く (5.0 → **10.0** × :data:`~..walk_kick.walk_kick_env_cfg._KICK_W_SCALE`)。
2. 軸足の扱いは **インサイドキックで実証済みの流儀**をそのまま持ち込む
   (線形テント / 重みは objective 段 / span カリキュラム / 軸足のヨー)。
   ガウスの ``kick_plant_foot`` は使わない。
3. 段は「歩行 → flat で浮かせる → rough + ボール DR」の 3 つ。

なぜ walk_lob_rough を直さずに新しい系列を立てるのか
----------------------------------------------------
1. **観測レイアウトが違う。** :mod:`..walk_lob_rough` は
   :class:`~..walk_kick_both_feet.walk_kick_both_feet_env_cfg.K1WalkKickBothFeetObservationsCfg`
   (スロット 3 = 左足裏 → ボール 3D 位置、critic 58 次元) に差し替えている。
   こちらは **walk_kick 素のレイアウト** (policy 55 次元・``sole_pos`` スロットのまま /
   critic 61 次元) を使う。理由は下の「観測レイアウト」節。同じモジュールに
   両方のレイアウトを同居させると、どの checkpoint がどちらに載るのかが
   クラス名からは読めなくなる。
2. **段の物語が違う。** あちらは 3 段目の lob を立ち上げるために **キック段**
   (walk_kick の報酬集合) を stage 2 として挟んでいる。こちらは段を足さず、
   stage 2 の中で ``kick_velocity_strong`` を折れ線で「立ち上げてから退場させる」
   ことでブートストラップする (下の「呼び水」節)。
3. **過去 run の帰属を壊さない。** ``k1_walk_lob`` / ``k1_walk_lob_hist`` /
   ``k1_walk_lob_rough`` の log は既に反証の材料として docstring から参照されている。
   同じ experiment 名のまま報酬設計を入れ替えると、あとから TB を見た人が
   どの設計の run なのか判別できなくなる。

引き継いでいる失敗の記録 (walk_lob / walk_lob_rough の docstring より)
---------------------------------------------------------------------
======================================  ==================================================
記録                                     このタスクでの扱い
======================================  ==================================================
**apex 0.425 m でプラトー**              目標 0.9 m に対して半分以下。loft/elevation の
(flat lob it11500、elevation 23.7°)      重みを 5.0 → 10.0 へ上げる (第 4 節)。
**plant_lon −0.42 で完全に不動**         ガウス (σ_lon 0.10) は実測位置で f ≈ 5e-4 =
(5 run、``kick_plant_foot`` は 0.0002)   真っ平ら。**線形テント** ``kick_plant_lon`` に
                                         置き換える (第 2 節)。
**「地面との間で弾ませる」exploit**      Isaac の反発 e≈0.6 でだけ成立し MuJoCo・実機
(walk_lob の変更点 7)                    (e≈0) で消える解。歯止めは φ_sat 60° +
                                         方向ゲートで、どちらも外さない (第 4 節)。
**walk phase → lob が直行しない**        ``kick_velocity_scaled`` を撤去したので「まず
(2026-08-18、eplen 25 のまま 400 iter)   ボールに触りにいく」動機が無い。``strong`` の
                                         折れ線で呼び水を作る (第 3 節)。
**``r_stance`` を動かしても効かない**    0.25 → 0.15 で plant_lon は 1.3 cm しか動かず
(2026-08-18)                             vel_ratio が 0.525 → 0.401 に落ちた。
                                         **触らない** (walk_lob 既定 0.25 のまま)。
**``kick_contact_height`` は反証済み**   sole_height は下がったが apex 0.340 → 0.234。
(2026-08-18)                             入れない。``kick_foot_ceiling`` も入れない
                                         (下の「入れないもの」)。
======================================  ==================================================

.. warning::
   **軸足の項は「威力」とトレードしうる。** 3 run から出た定量則は

       apex 上昇 ∝ (kick_vel_ratio · sin φ)²

   で、仰角の変動幅 (24-28°) よりボール速度の変動幅 (0.40-0.70) の方がずっと大きい。
   つまり **apex は実質ボール速度が支配している**。``kick_plant_lon`` /
   ``kick_plant_yaw`` は「構え」を指定する項なので、構えを取るためにスイングを
   短くすれば apex は下がりうる。インサイドでは vel_ratio 0.887 を保ったまま
   plant_lon が −0.23 → −0.107 まで動いた実績があるが、**あれは強い蹴りを目的に
   含むタスクでの話**で、ロブでは目的が高さに寄っているぶん逃げ道が違う。
   副作用の監視は ``Metrics/kick_direction/kick_vel_ratio`` と
   ``Metrics/kick_direction/foot_vz`` の 2 つ。どちらかが落ちながら plant_lon だけが
   動いているなら、軸足の項が威力を食っている。そのときは weight ではなく
   **span を緩める側** (第 2 節の折れ線の後段を後ろへずらす) で調整すること。

観測レイアウト — walk_kick 素の 55 / 61 を使う
----------------------------------------------
policy 55 次元 (スロット 3 = ``sole_pos`` のまま) / critic 61 次元。
:mod:`..walk_lob_rough` の both_feet レイアウト (スロット 3 = ボール 3D 位置) は
**持ち込まない**。理由は 3 つ:

* **共用の歩行 checkpoint がそのまま載る。** stage 1 の既定は
  ``logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt``
  (walk_kick / walk_inside_kick と同じもの)。both_feet レイアウトにすると
  次元は 55 のままなので ``--load_pretrained`` は **形の上では通ってしまう**が、
  スロット 3 の意味が違うので中身が繋がらない。たちの悪い壊れ方をする。
* **インサイド系と checkpoint を相互に流用できる。** 同じ 55/61 なので、
  ロブ側で歩容が崩れたときにインサイドの収束済み checkpoint から入り直せる。
* **右足専用タスクなので mirror は使わない。** 左右対称化する予定が無い以上、
  「左右の足裏を両方観測に入れる」という both_feet の動機がそもそも無い
  (walk_lob_rough も両足キック化と mirror は入れていない)。

ボール 3D 位置スロットが無くて困らないのは、``prev_ball_pos`` (2D、実機の認識
パイプライン相当のノイズ+遅延つき) と critic 側の ``ball_pos_rel`` (特権、3D) で
必要な情報が揃っているため。ロブで足りないのは「ボールがどこにあるか」ではなく
「どう当てるか」だというのが 3 run の結論。

観測履歴は **全段** に入れる (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)
-------------------------------------------------------------------------------------------------
actor の入力を 100 フレーム (N, 100, 55) にして
:class:`~..locomotion.networks.ActorCriticHistoryCNN` で受ける。critic は 1 フレーム
(61 次元) のまま。段ごとに入れたり外したりしないので、stage 1 → 2 → 3 の
checkpoint は全部そのまま繋がる (履歴 → 履歴)。

stage 1 だけは引き継ぎ元 (共用の歩行 checkpoint) が **1 フレーム観測**なので、
``--warm_start_from_single_frame`` で旧 actor を「最新フレームの列」へ移植する
(:func:`~..locomotion.networks.remap_single_frame_actor`)。
通しスクリプトが checkpoint の中身を見て自動で付ける。

.. note::
   mirror loss は入れない (:func:`~..walk_inside_kick.agents.rsl_rl_ppo_cfg._use_history_cnn_policy`
   を使う)。右足専用タスクで鏡像対称性が成り立たないのはインサイドと同じ。

入れないもの (すべて意図的)
---------------------------
* ``kick_plant_foot`` (ガウス版の軸足) — 反証済み。第 2 節で項ごと外す。
* ``kick_contact_height`` (低く当てるほど得) — 反証済み (apex 0.340 → 0.234)。
* ``kick_foot_ceiling`` (足を上げすぎない天井) — インサイドでは入れているが、
  **ロブでは入れない**。ロブが欲しいのは「ボールの下に潜る」ことで、天井は
  その方向と正面から競合しうる。加えて今回は「軸足 2 項 + loft/elevation の
  重み倍増」を同時に入れており、これ以上変更点を増やすと apex が動いた/動かない
  原因を帰属できなくなる。効かなかったときの次の一手として温存する。
* ``r_stance`` の変更 — 反証済み (0.25 → 0.15 で vel_ratio が落ちただけ)。
* ``kick_velocity_scaled`` の復活 — ロブの設計そのもの。「指令の速さに一致しろ」は
  vz = v·sin φ の最大化と衝突する (:mod:`..walk_lob` の変更点 1)。
  ブートストラップは ``strong`` の折れ線が担う (第 3 節)。
* 回り込み型 G (``apply_orbit_params``) / 拡大ゲート — インサイドのレシピには
  入っているが、こちらは loop_shoot 由来のコマンド設計をそのまま使う。
  変更点を「浮かせる」ことに絞るため。

TensorBoard で見るもの
----------------------
* ``Metrics/kick_direction/kick_rate`` — **最初に見るのはこれ**。インサイドは
  250 iteration で 0.85 を超えた。呼び水 (strong) が効いていれば 500 iteration まで
  には立ち上がる。立ち上がらなければ第 3 節の設計が失敗したということなので、
  walk_lob_rough と同じ「キック段を挟む」へ戻す判断になる。
* ``Metrics/kick_direction/kick_apex_height`` — 本命。0.425 (旧 flat lob の頭打ち)
  を超えて 0.9 へ向かうか。
* ``Metrics/kick_direction/plant_lon`` — −0.42 から 0 側へ動くか
  (``kick_plant_lon`` が効いているかの唯一の判定材料)。
* ``Metrics/kick_direction/plant_yaw_dot`` — ``kick_plant_yaw`` の対象。
  **1 iteration 目の値を必ず記録すること** (ロブ系での実測がまだ無い。
  1 = 蹴り方向、0 = 真横)。素の値が既に 0.9 級ならこの項は効きようが無いので、
  weight ではなく ``yaw_span`` を絞る側で考え直す。
* ``Metrics/kick_direction/kick_vel_ratio`` / ``foot_vz`` — 副作用の監視
  (上の warning)。
* ``Train/mean_episode_length`` と ``Episode_Termination/base_height`` — 段の
  切り替え直後は必ず崩れる。100-200 iteration で eplen が戻らなければ、その段は
  前段からブートストラップできていない (2026-08-18 の失敗と同じ形)。

禁止フラグ
----------
* ``--resume`` — experiment_name が段ごとに違うので前段を検出できないうえ、
  ``common_step_counter`` を同期してしまい、キック報酬のフェードインと strong の
  折れ線が「もう終わった」と判定される。段の引き継ぎは常に ``--load_pretrained``。
* ``--reset_noise_std`` — 歩行 checkpoint のスイングを壊す。
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..locomotion.rough_env_cfg import _apply_play_viewer
from ..walk_inside_kick.walk_inside_kick_env_cfg import (
    _BALL_AVOIDANCE_SIGMA_POSE,
    _BALL_AVOIDANCE_SIGMA_SOLE,
    _INSIDE_STRONG_KNOTS,
    _PLANT_LON_TARGET,
    _PLANT_YAW_SPAN,
    _ROUGH_BALL_DYNAMIC_FRICTION_RANGE,
    _ROUGH_BALL_MASS_SCALE_RANGE,
    _ROUGH_BALL_RESTITUTION_RANGE,
    _ROUGH_BALL_STATIC_FRICTION_RANGE,
)
from ..walk_kick import mdp
from ..walk_kick.curriculum_pin import pin_curricula_at_end
from ..walk_kick.walk_kick_env_cfg import (
    _apply_rough_terrain,
    _disable_ball_obs_jitter,
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
    K1WalkKickWalkPhaseEnvCfg,
)
from ..walk_kick_dual.walk_kick_dual_env_cfg import enable_obs_history
from ..walk_lob.walk_lob_env_cfg import _LOB_SIGMA_DIRECTION, K1WalkLobEnvCfg
from ..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg import (
    _BALL_OBS_DELAY_MAX_S as _FEWA_BALL_OBS_DELAY_MAX_S,
)
from ..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg import (
    _OBS_DELAY_MAX_S as _FEWA_OBS_DELAY_MAX_S,
)
from ..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg import (
    enable_obs_delay as _fewa_enable_obs_delay,
)
from ..walk_weak_kick_orbit.orbit_mods import apply_ball_param_dr

# --------------------------------------------------------------------------- #
# カリキュラムが 1 iteration とみなす env step 数
#
# ``mdp.curriculums`` の start_step / end_step は
# ``env.common_step_counter // steps_per_iteration`` で iteration に換算される。
# ``common_step_counter`` は ``env.step()`` 1 回で 1 増えるので、この値は PPO の
# ``num_steps_per_env`` と **一致していなければ** 書いた数字が iteration にならない。
#
# この系列の RunnerCfg は 48 (:data:`.agents.rsl_rl_ppo_cfg._NUM_STEPS_PER_ENV`。
# 基底 :class:`~..locomotion.agents.rsl_rl_ppo_cfg.K1FlatPPORunnerCfg` の値そのもので、
# あちらが動いてもここが黙って壊れないよう **RunnerCfg 側でも明示的に代入している**)。
#
# なぜ 24 ではなく 48 なのか — 継承したランプも全部 48 に書き換える
# ----------------------------------------------------------------
# walk_kick 系の既存カリキュラムは全て ``steps_per_iteration = 24`` を渡しているが、
# 実際の ``num_steps_per_env`` は 48 なので、**書いてある iteration 数の半分で
# 終わっている** (48 // 24 = 2 → カリキュラムの 1 step = 0.5 iteration)。
# walk_lob_rough が 2026-08-18 の run で実測確認した既知の食い違い
# (:mod:`..walk_lob_rough.walk_lob_rough_env_cfg` の「カリキュラムの時間単位に
# ついての注意」)。
#
# この系列は **新設・既存を問わず全部 48 に揃える** (:func:`_retime_curricula`)。
# 選んだ理由:
#
# * 混在させると「end_step 500」が項によって iteration 250 だったり 500 だったり
#   することになり、フェードインの同時性 (キック報酬 6 項が同じ窓で立ち上がる)
#   という設計意図そのものが崩れる。どちらかに揃えるしかない。
# * 揃えるなら **書いてある数字が本当の iteration になる方**。ここは新しい系列で、
#   合わせるべき既存の収束 run が無い (むしろ walk_lob 系の run は全部反証側)。
# * 影響は「全カリキュラムが一様に 2 倍の壁時計へ引き伸ばされる」だけで、項どうしの
#   相対関係は 1 ミリも変わらない。そして引き伸ばしの向きは **序盤が緩くなる**方 =
#   ``approach_penalty`` (−3.0) と ``kick_pose_overshoot`` (−50.0) がゆっくり効く方。
#   この系列の documented な失敗は「負の圧力が先に効いて立ち上がらない」なので、
#   緩む向きは安全側にある。
# * 代償は壁時計。フェードインが 500 iteration、strong の退場が 1200 iteration、
#   lon_span の仕上げが 5000 iteration になる。既定 ``ITER=8000`` はこれを見込んだ量。
#
# NOTE: 継承元のモジュール (walk_kick / walk_loop_pass / walk_loop_shoot / walk_lob) は
#       **触っていない**。書き換えるのはこのタスクの cfg インスタンスに乗った
#       curriculum 項の params だけなので、既存タスクの挙動は変わらない。
# --------------------------------------------------------------------------- #
_SPI = 48

# --------------------------------------------------------------------------- #
# 「浮かせる」2 項の最終重み (× _KICK_W_SCALE) — このタスクの最優先事項
#
# walk_lob 既定は loft 5.0 / elevation 5.0 (direction 6.0 の下)。**10.0 へ上げる。**
#
# 「壊れない範囲で最強」の根拠
# ----------------------------
# 1. **農作の抜け道が構造で塞がっている。** ``kick_loft`` も ``kick_elevation`` も
#    ``r_direction`` への **乗算**なので、方向ゲート (τ_direction) ・胴体の正対
#    (p_style) ・latch ゲート (kick_done) を通らない蹴りには 1 円も払われない。
#    重みを上げても「ボールが上に飛びさえすればよい」にはならない
#    (:mod:`..walk_lob` の変更点 4 の NOTE — 方向ゲートは絶対に外さない)。
# 2. **残る exploit は 1 つだけで、そこにも歯止めがある。** sim 固有の
#    「足を水平に突っ込んで地面との間でボールを弾ませる」解 (反発 e≈0.6 依存)。
#    歯止めは φ_sat = 60° (これ以上仰角を付けても得しない = 真上へポップする動機が
#    出ない) と方向ゲートの二重で、``kick_foot_lift`` (接触の瞬間の足の鉛直速度)
#    が原因側を直接見る。重みを上げても **この 3 つは 1 つも緩まない**。
# 3. **天井は歩容・安定項との相対で決まる。** キック報酬は latch 後の猶予窓 (2.0 秒)
#    にしか払われず、``_KICK_W_SCALE`` で 1 回あたりの収益を一定に割り戻してある。
#    一方 ``termination_penalty`` (−100)・``base_height``・``flat_orientation_l2``
#    などはエピソード全体に効く。10.0 = direction (6.0) の 1.7 倍、
#    loft + elevation の合計で 20.0 は「1 回の蹴りの価値」としては大きいが、
#    転倒 1 回 (−100 × dt の一撃 + 残りの歩行報酬の喪失) を下回る。
#
# **さらに上げるのが最初のレバー。** apex が頭打ちで、かつ歩容が健全
# (``Train/mean_episode_length`` が高い / ``Episode_Termination/base_contact`` が
# 低い) なら、次は 12.0 → 15.0 と上げる。逆に eplen が落ちているなら上げすぎ。
#
# NOTE: **カリキュラム側の ``end_weight`` を書き換えること。** 2 項とも
#       ``linear_reward_weight`` で 0 → 終値へランプする構成なので、報酬項の
#       ``weight`` を直接上げても毎ステップ上書きされて効かない
#       (:func:`~..walk_lob_rough.walk_lob_rough_env_cfg._raise_foot_lift_weight` の
#        docstring にある同じ注意)。
# --------------------------------------------------------------------------- #
_LOFT_WEIGHT = 10.0
_ELEVATION_WEIGHT = 10.0

# --------------------------------------------------------------------------- #
# すくい上げ (kick_foot_lift) の最終重み (× _KICK_W_SCALE)
#
# walk_lob 既定は 2.0。**6.0 へ上げる** = walk_lob_rough が stage 3 で採った値
# (:data:`~..walk_lob_rough.walk_lob_rough_env_cfg._FOOT_LIFT_WEIGHT`) と同じ。
#
# あちらがそこまで上げた理由をそのまま引き継ぐ: 3 run の実測で
# ``apex 上昇 ∝ (kick_vel_ratio · sin φ)²`` が成り立ち、**速度を削らずに仰角を稼げる
# 唯一の量が foot_vz** だと分かった (当たり所を下げる施策は立ち位置を詰めさせ、
# スイング長 = 速度を削って自滅した)。過去の実測は weight 4.0 で foot_vz +0.17、
# 2.0 で −0.32 なので、正へ十分振らせるには 4.0 でも足りていない。
#
# direction (6.0) と同着まで。方向ゲートより上には置かない。
#
# NOTE: ``vz_foot_sat`` は walk_lob 既定の 2.0 のまま。実測 foot_vz が飽和に
#       近づいたら上げること (旧 flat lob の実測は +0.81 = 飽和の 40%)。
# --------------------------------------------------------------------------- #
_FOOT_LIFT_WEIGHT = 6.0

# --------------------------------------------------------------------------- #
# 軸足の前後位置 (kick_plant_lon) の最終重み (× _KICK_W_SCALE)
#
# インサイドの :data:`~..walk_inside_kick.walk_inside_kick_env_cfg._PLANT_LON_WEIGHT`
# と同じ 6.0 = direction 同格 = 「形の項が目的の項を出し抜かない」序列の上限。
# あちら側のコメントにある上限の根拠 (r_direction 乗算なので農作の抜け道は構造で
# 塞がれており、残る壊れ方は「威力・精度を削ってでも軸足を置く」トレードだけ) は
# ロブでもそのまま成り立つ。
#
# ただし **ロブでは objective が loft/elevation (10.0 ずつ) の側にある**ので、
# 6.0 は inside と違って「objective より下」。形が高さを出し抜けない配分になっている。
#
# 目標値 (:data:`~..walk_inside_kick.walk_inside_kick_env_cfg._PLANT_LON_TARGET` = 0.0)
# は inside から **import してそのまま使う**。span の折れ線だけはロブ専用
# (:data:`_LOB_PLANT_LON_SPAN_KNOTS`、下)。
# --------------------------------------------------------------------------- #
_PLANT_LON_WEIGHT = 6.0

# --------------------------------------------------------------------------- #
# span の折れ線 (ロブ専用)。inside の ``_PLANT_LON_SPAN_KNOTS`` (0.45 → 0.25 → 0.15)
# を **そのまま使ってはいけない**。
#
# ロブの実測 plant_lon は **−0.42** (走り出しの最悪値) で、inside の出発点 −0.23 より
# ずっと後ろ。inside の初期半幅 0.45 だと f = 1 − 0.42/0.45 = 0.07 と勾配が薄く、
# 第 2 段 (1500 → 3000 で 0.25 まで絞る) に入った時点で −0.42 の位置は f = 0 に潰れる。
# 「ポリシーの居る場所で報酬が真っ平ら」はこの系列で kick_plant_foot を殺した
# 死因そのものなので、出発点に合わせて広く始め、絞りも遅らせる:
#
#   iteration    0     1500    3000    5000
#   span        0.60   0.60    0.35    0.25
#   f(−0.42)    0.30   0.30    0.0*    —        * ここまで一歩も動いていなければ別の問題
#   f(−0.30)    0.50   0.50    0.14    —
#   f(−0.15)    0.75   0.75    0.57    0.40
#   勾配 W/span 10     10      17      24
#
# 終値 0.25 は inside の第 2 段と同じ値 (inside がそこから −0.11 まで踏み込めた実績)。
# inside の第 3 段 (0.15) までは絞らない — ロブは apex 優先で、軸足の圧を上げ切る
# より「威力と軸足のトレード」が apex に出ないことを先に確かめる。
# 絞りの開始 1500 は strong の退場 (1200) の後 (トーキック期の居場所を潰さない)。
#
# .. warning::
#    ``Metrics/kick_direction/plant_lon`` が 3000 iteration までに −0.30 側へ
#    動いていなければ、第 2 段で f が 0 に潰れる。そのときは折れ線を後ろへずらすのでは
#    なく、**軸足が報酬で動かない別の原因** (歩幅と接触位相、walk_lob_rough の記録) を
#    疑うこと。折れ線の形は 1 本の piecewise で書くこと (2 本並べると同じ param に
#    書き手が 2 人になる。:func:`~..walk_kick.mdp.curriculums.piecewise_reward_param`)。
# --------------------------------------------------------------------------- #
_LOB_PLANT_LON_SPAN_KNOTS = [(0, 0.60), (1500, 0.60), (3000, 0.35), (5000, 0.25)]

# --------------------------------------------------------------------------- #
# 軸足の向き (kick_plant_yaw) の最終重み (× _KICK_W_SCALE)
#
# インサイドの :data:`~..walk_inside_kick.walk_inside_kick_env_cfg._PLANT_YAW_WEIGHT`
# と同じ 3.0 = 「形の項は objective の半分から入れる」。
# span は :data:`~..walk_inside_kick.walk_inside_kick_env_cfg._PLANT_YAW_SPAN`
# (π/2 = 90°) を import。
#
# ロブでこの項に期待する働きは inside と少し違う。あちらは「軸足が斜めだと骨盤も
# 斜めになり、振り足のインサイド面がキック線に正対しない = 当たりが薄い」だった。
# こちらは **軸足が蹴り方向を向いていれば骨盤が開き、振り足をボールの下へ潜らせる
# 余地が増える** という仮説。どちらも「軸足の向きが振り足の通り道を決める」という
# 同じ幾何の話なので、実証済みの形をそのまま流用する。
#
# NOTE: ``Metrics/kick_direction/plant_yaw_dot`` の **ロブ系での実測がまだ無い**。
#       1 iteration 目の値を必ず記録すること (素の値が既に 0.9 級ならこの項は
#       効きようが無いので、weight ではなく yaw_span を絞る側で考え直す)。
# --------------------------------------------------------------------------- #
_PLANT_YAW_WEIGHT = 3.0

# キック報酬のフェードイン窓 [iteration]。基底 walk_kick の _phase2 と同じ 0 → 500 で、
# 新設 3 項も既存項と同時に立ち上げる (発見期には既に満額で乗っている状態にする)。
_FADE_IN_END_ITER = 500


def _retime_curricula(cfg) -> list[str]:
    """cfg に載っている **全ての** curriculum 項の ``steps_per_iteration`` を
    :data:`_SPI` に揃える。

    継承元 (walk_kick / walk_loop_pass / walk_loop_shoot / walk_lob) のランプは全て
    24 を渡しているが、実際の ``num_steps_per_env`` は 48 なので、書いてある
    iteration 数の **半分** で終わっている。新設分だけ 48 にすると同じ「end_step 500」が
    項によって別の壁時計を指すことになるので、この系列は全部 48 に揃える。
    判断の理由は :data:`_SPI` のコメント。

    ``steps_per_iteration`` を **持っている項だけ**触る。locomotion 側の 3 項
    (``modify_command_resampling_time_range`` / ``lin_vel_command_curriculum`` /
    ``modify_push_robot``) は生の ``common_step_counter`` を見る作りでこのキーを
    持たないので、自動的に対象外になる (キーを足すと未知の引数として渡って落ちる)。

    **報酬・カリキュラムの登録が全部済んだ後に呼ぶこと。** 後から足した項は
    当然ながら書き換えられない。

    Returns:
        書き換えた curriculum 項の名前 (呼び出し側の検証・表示用)。
    """
    retimed: list[str] = []
    for name in sorted(dir(cfg.curriculum)):
        if name.startswith("_"):
            continue
        term = getattr(cfg.curriculum, name, None)
        # configclass のメソッド (to_dict / replace など) も dir() に出るので型で確かめる。
        if not isinstance(term, CurrTerm):
            continue
        if "steps_per_iteration" not in term.params:
            continue
        if term.params["steps_per_iteration"] == _SPI:
            continue
        term.params["steps_per_iteration"] = _SPI
        retimed.append(name)
    return retimed


def _apply_lob_plant_recipe(cfg: "K1WalkLobEnvCfg") -> None:
    """walk_lob のロブ報酬一式に「軸足の踏み込み」と「呼び水」を足す。

    ``__post_init__`` の中で ``super()`` の後・:func:`enable_obs_history` の前に
    呼ぶこと。観測の次元・並びには一切触らない (55 / 61 のまま) ので、共用の歩行
    checkpoint がそのまま載る。

    やること (番号は下のコメントと対応):

    1. ``kick_plant_foot`` (ガウス) を項ごと撤去
    2. ``kick_plant_lon`` (線形テント) + ``kick_plant_yaw`` を追加
    3. ``kick_velocity_strong`` を折れ線で復活 = 発見の呼び水
    4. ``kick_loft`` / ``kick_elevation`` / ``kick_foot_lift`` の終値を引き上げ
    5. 接触幾何のメトリクスを出す
    6. 全カリキュラムの時間単位を :data:`_SPI` へ統一
    """
    # -- 1. ガウス版の軸足 (kick_plant_foot) を項ごと撤去 -------------------- #
    #
    # walk_lob が ``__post_init__`` で報酬項とカリキュラム (``kick_plant_foot_weight``)
    # の両方を登録するので **両方消すこと**。項だけ消してカリキュラムを残すと
    # ``linear_reward_weight`` が存在しない term を ``reward_manager.get_term_cfg`` で
    # 引きにいって落ちる (:func:`~..walk_lob_rough.walk_lob_rough_env_cfg._remove_plant_foot_reward`
    # と同じ手当て)。
    #
    # 撤去の理由: walk_lob 系 5 run の実測で ``plant_lon`` は −0.36〜−0.43 に居座り、
    # 一度も目標 (−0.03) に漸近しなかった。σ_lon = 0.10 のガウスは実測 −0.42 で
    # f = exp(−0.39²/(2·0.10²)) ≈ 5e-4 = **裾の完全な外側で真っ平ら**。
    # ``Episode_Reward/kick_plant_foot`` の実測 0.0002 がその帰結。
    # 「軸足は動かない」のではなく「動かす勾配が無かった」というのが 2026-08-23 の
    # 読み直しで、下の第 2 節がそこを線形テントで置き換える。
    #
    # walk_lob_rough が試した「目標を動かすカリキュラム」(kick_plant_foot_lon_target /
    # _sigma_lon) はこの系列には存在しないが、将来 walk_lob 側に足されたときに
    # 取りこぼさないよう getattr で見てから消す。
    cfg.rewards.kick_plant_foot = None
    for _curr in ("kick_plant_foot_weight", "kick_plant_foot_lon_target", "kick_plant_foot_sigma_lon"):
        if getattr(cfg.curriculum, _curr, None) is not None:
            setattr(cfg.curriculum, _curr, None)

    # -- 2a. 軸足の前後位置を線形テントで誘導する (kick_plant_lon) ----------- #
    #
    # インサイドキックで実証済みの形をそのまま持ち込む。
    # :func:`~..walk_kick.mdp.rewards.kick_plant_lon` は半幅 ``lon_span`` の線形テントで、
    # −0.42 でも f = 0.07、−0.23 で f = 0.51 と **ポリシーが実際に居る場所に傾きが残る**。
    # これが撤去したガウス版との唯一にして決定的な違い。
    # lat (横) は掛けない — plant_lat は walk_lob の実測で目標 0.19 に対して 0.17 と
    # 最初から合っており、掛けると「横が外れているあいだ lon の勾配も死ぬ」という
    # kick_plant_foot の失敗を作り直すことになる。
    #
    # 目標 (0.0) は inside から import、span の折れ線はロブ専用
    # (:data:`_LOB_PLANT_LON_SPAN_KNOTS`: 出発点 −0.42 に合わせて広く始める)。
    # 初期 span は折れ線の先頭 knot からそのまま読む。
    #
    # sigma_direction は **このタスクの他のキック項と必ず同じ値** (0.6)。
    # 項ごとに違うと方位を外したときの損得が食い違って何を最適化しているのか読めなくなる
    # (:mod:`..walk_lob` の変更点 4)。
    cfg.rewards.kick_plant_lon = RewTerm(
        func=mdp.kick_plant_lon,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _LOB_SIGMA_DIRECTION,
            "lon_target": _PLANT_LON_TARGET,
            "lon_span": _LOB_PLANT_LON_SPAN_KNOTS[0][1],
        },
    )
    cfg.curriculum.kick_plant_lon_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_plant_lon",
            "start_weight": 0.0,
            "end_weight": _PLANT_LON_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": _FADE_IN_END_ITER,
            "steps_per_iteration": _SPI,
        },
    )

    # span の折れ線 (0.60 → 0.35 → 0.25、:data:`_LOB_PLANT_LON_SPAN_KNOTS`)。
    # テントの勾配は W/span なので、weight を上限で止めたあとの増強は **span を絞る側**
    # で行う、という inside の設計を引き継ぐ。ただし出発点が −0.42 と後ろなので
    # inside の数列は使わない (定数のコメント参照)。3 段を **1 本の折れ線** で書くこと
    # (linear_reward_param を 2 本並べると同じ param に書き手が 2 人になり、最終値が
    # CurriculumManager の実行順で決まってしまう。
    # :func:`~..walk_kick.mdp.curriculums.piecewise_reward_param` の docstring)。
    cfg.curriculum.kick_plant_lon_span = CurrTerm(
        func=mdp.piecewise_reward_param,
        params={
            "term_name": "kick_plant_lon",
            "param_name": "lon_span",
            "knots": _LOB_PLANT_LON_SPAN_KNOTS,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 2b. 軸足の向きを誘導する (kick_plant_yaw) -------------------------- #
    #
    # 軸足のつま先を蹴り方向へ向かせる項 (角度に対する線形テント、半幅 90°)。
    # 2a とは **加算** で並べる。掛け算にすると「片方が外れているあいだ両方の勾配が
    # 死ぬ」という kick_plant_foot の失敗を作り直す。位置が合っていても向きは独立に
    # 外れるので、項を分けるのが正しい。
    cfg.rewards.kick_plant_yaw = RewTerm(
        func=mdp.kick_plant_yaw,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _LOB_SIGMA_DIRECTION,
            "yaw_span": _PLANT_YAW_SPAN,
        },
    )
    cfg.curriculum.kick_plant_yaw_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_plant_yaw",
            "start_weight": 0.0,
            "end_weight": _PLANT_YAW_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": _FADE_IN_END_ITER,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 3. 発見の呼び水: kick_velocity_strong を折れ線で復活させる ---------- #
    #
    # **walk_lob_rough の「キック段 (stage 2) を挟む」を置き換えるのがこの節。**
    #
    # ロブの報酬集合は ``kick_velocity_scaled`` を撤去してあるため、「まずボールに
    # 触りにいく」動機を作る密な項が 1 つも無い。``kick_loft`` / ``kick_elevation`` は
    # **当たった後の飛び方**しか見ないので、一度も当てられない段階では全部 0 で
    # 勾配が立たない。残る密な信号は ``approach_penalty`` (−3.0) と
    # ``kick_pose_overshoot`` (−50.0) という **負の圧力**で、実際 2026-08-18 の
    # 2 段 run は eplen 25 ステップのまま 400 iteration 改善しなかった
    # (「蹴らない」のではなく **蹴る前に立てていない**)。
    #
    # ``kick_velocity_strong`` = r_direction × v_ball (青天井) は、まさに
    # 「ボールを速く飛ばせば得」という **発見用の密な項**。loop_pass が項ごと
    # 撤去しているのでここで作り直す (基底 walk_kick の定義と同じ形)。
    #
    # 折れ線はインサイドの
    # :data:`~..walk_inside_kick.walk_inside_kick_env_cfg._INSIDE_STRONG_KNOTS`
    # をそのまま import する: ``[(0, 0), (500, _STRONG_W), (1200, 0)]``。
    # 0 → 500 で立ち上げ、**500 → 1200 で 0 へ落として退場**させる。
    # weak/middle の既定 [(0,0),(500,W),(1500,W),(3000,0)] ではなく inside の
    # 前倒し版を使うのは、strong が「いちばん強く振れる蹴り方」= 低い弾道の
    # トーキックを名指しで要求する項だから。inside では発見が 250 iteration で
    # 済んでいるのに 0-1500 の間 strong の払いが当たり所の項の 43〜69 倍あり、
    # 型の決定期をまるごと支配していた。ロブでも同じことが起きると **高さを
    # 捨てて速さを取る**型で固まる。
    #
    # 終値が 0 であることは変えないこと。少しでも残すと「速いほど得」が残り、
    # vz = v·sinφ の φ 側を伸ばす動機と最後まで競合する。
    #
    # NOTE: ``_STRONG_W`` は :data:`~..walk_weak_kick.walk_weak_kick_env_cfg._STRONG_W`
    #       で、既に ``_KICK_W_SCALE`` を掛けた値 (= 0.9)。折れ線の knot にはそのまま
    #       入っているので、ここで重ねて掛けないこと。
    # NOTE: sigma_direction はこのタスクの値 (0.6)。基底の定義は 0.35 なので、
    #       作り直すときに必ず渡す。
    cfg.rewards.kick_velocity_strong = RewTerm(
        func=mdp.kick_velocity_strong,
        weight=0.0,
        params={**_KICK_STATE_PARAMS, "sigma_direction": _LOB_SIGMA_DIRECTION},
    )
    cfg.curriculum.kick_velocity_strong_weight = CurrTerm(
        func=mdp.piecewise_reward_weight,
        params={
            "term_name": "kick_velocity_strong",
            "knots": _INSIDE_STRONG_KNOTS,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 4. 「浮かせる」項の終値を引き上げる -------------------------------- #
    #
    # loft 5.0 → 10.0 / elevation 5.0 → 10.0 / foot_lift 2.0 → 6.0。
    # 根拠は :data:`_LOFT_WEIGHT` と :data:`_FOOT_LIFT_WEIGHT` のコメント。
    #
    # **カリキュラムの end_weight を書き換える** (報酬項の weight ではない)。
    # 3 項とも ``linear_reward_weight`` で 0 → 終値へランプする構成なので、項側の
    # weight を直接上げてもカリキュラムに毎ステップ上書きされて効かない。
    cfg.curriculum.kick_loft_weight.params["end_weight"] = _LOFT_WEIGHT * _KICK_W_SCALE
    cfg.curriculum.kick_elevation_weight.params["end_weight"] = _ELEVATION_WEIGHT * _KICK_W_SCALE
    cfg.curriculum.kick_foot_lift_weight.params["end_weight"] = _FOOT_LIFT_WEIGHT * _KICK_W_SCALE

    # NOTE: ``kick_direction`` は基底の 6.0 のまま **触らない**。方向ゲートは
    #       踏みつけ / かすらせ exploit を塞ぐ構造そのもので、ここを相対的に
    #       弱めると loft/elevation を上げた分だけ exploit 側が有利になる。
    #       (loft/elevation が direction を数値で超えるのは、あの 2 項が
    #        r_direction への **乗算**で、direction を経由してしか払われないため。
    #        乗算の係数と加算の項を同じ土俵で比べる必要はない。)

    # -- 5. 接触の幾何をメトリクスに出す ------------------------------------ #
    #
    # ``Metrics/kick_direction/plant_yaw_dot`` (第 2b 節の対象) と foot_kick_dot /
    # ball_side が出るようになる。既定 False = 他タスクの TB タグ集合を変えないため
    # (:class:`~..walk_kick.mdp.commands.KickDirectionCommandCfg` の同名フラグ)。
    #
    # ``plant_lon`` / ``plant_lat`` / ``sole_height_at_kick`` / ``foot_vz`` は
    # このフラグとは無関係に常時出る。
    cfg.commands.kick_direction.log_contact_geometry = True

    # -- 6. カリキュラムの時間単位を _SPI へ統一 ---------------------------- #
    #
    # **必ず最後**。ここまでで登録した全ての項 (継承分 + 新設分) が対象。
    # 理由は :data:`_SPI` のコメント。
    _retime_curricula(cfg)


def _apply_play_tweaks(cfg) -> None:
    """PLAY 共通の間引き (env 数・外乱・観測ノイズ)。"""
    cfg.scene.num_envs = 20
    cfg.scene.env_spacing = 4
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


# =========================================================================== #
# Stage 1: 歩行のみ (flat, 履歴 actor)
# =========================================================================== #
@configclass
class K1WalkLobPlantWalkPhaseEnvCfg(K1WalkKickWalkPhaseEnvCfg):
    """Stage 1: ボール無しで歩行だけを学習する。観測は 100 フレームの履歴。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickWalkPhaseEnvCfg` との差は
    **:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history` を掛ける
    ことだけ**。報酬・地形・コマンドは共用の walk phase と完全に同一なので、この段の
    仕事は「既に収束している歩容を、履歴 actor という別のネットワークで再現し直す」
    ことに尽きる。

    :class:`~..walk_lob.walk_lob_env_cfg.K1WalkLobWalkPhaseEnvCfg` ではなく walk_kick
    の walk phase を直接継承しているのは、あちらが「experiment 名を分けるためだけの
    空サブクラス」だから (T-N カーブ付きアクチュエータは撤回済みで実体は同一)。
    継承の鎖を 1 段短くしておく。

    既定の引き継ぎ元 (通しスクリプト)::

        logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt

    これは **1 フレーム観測** (素の ``ActorCritic``) なので、そのままでは履歴 actor
    (``ActorCriticHistoryCNN``) に 1 本も載らない。``--warm_start_from_single_frame``
    で旧 actor を「最新フレームの列」へ移植すると、学習開始時点の出力が元のポリシーと
    一致する (:func:`~..locomotion.networks.remap_single_frame_actor`)。
    通しスクリプトが checkpoint の中身 (``actor.0.weight`` の有無) を見て自動で付ける。

    ``WALK_ITER`` の既定が 2000 と短いのはこのため
    ---------------------------------------------
    歩容そのものは既に収束しており、この段で獲得し直すのは「履歴を入力に取る actor の
    重み」だけ。ゼロから歩行を学習する場合 (``WALK_CKPT="" `` で警告を無視して回す
    場合) は 8000-20000 iteration 必要なので、``WALK_ITER`` も一緒に上げること。

    カリキュラムについて
    --------------------
    walk phase はキック報酬とそのカリキュラムを全て ``None`` にしているので、残るのは
    locomotion 側の 3 項 (コマンド再サンプリング間隔 / 線速度コマンド範囲 / 外乱プッシュ)
    だけ。3 つとも ``steps_per_iteration`` を取らず生の ``common_step_counter`` を
    見るので、:func:`_retime_curricula` の対象外 = **この段では時間単位の問題が
    そもそも起きない**。したがってここでは呼ばない。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # 必ず最後。policy 観測グループの構成が固まってから (N, H, 55) に変える。
        enable_obs_history(self)


@configclass
class K1WalkLobPlantWalkPhaseEnvCfg_PLAY(K1WalkLobPlantWalkPhaseEnvCfg):
    """Stage 1 の PLAY。

    :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter` は **呼ばない**。
    walk phase では ``prev_ball_pos`` が歩行コマンド (``walk_command_xy``) に
    差し替わっており、``jitter_std`` を書き込むと未知のキーワード引数として
    渡って落ちる。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)


# =========================================================================== #
# Stage 2: flat で「浮かせる蹴り」を発見・収束させる (本命)
# =========================================================================== #
@configclass
class K1WalkLobPlantEnvCfg(K1WalkLobEnvCfg):
    """Stage 2: 平坦でロブを学習する。軸足の踏み込み + 呼び水 + 高さ重視の重み。

    :class:`~..walk_lob.walk_lob_env_cfg.K1WalkLobEnvCfg` (ロブの報酬設計一式 +
    実機の認識パイプライン相当のボール観測ノイズ) を土台に、
    :func:`_apply_lob_plant_recipe` の 6 点を足したもの。

    ``__post_init__`` の順序に意味がある::

        super()                    … ロブの報酬一式 + _apply_noisy_ball_obs
        _apply_lob_plant_recipe()  … 報酬・カリキュラムの差分 (観測には触らない)
        enable_obs_history()       … **最後**。観測グループの構成が固まってから履歴化

    ``super()`` が最後に :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs`
    を呼んでいるが、レシピは報酬とカリキュラムしか触らないので順序の衝突は無い。
    :mod:`..walk_lob_rough` のような :func:`_restore_vision_ball_obs` 相当の後始末も
    不要 — こちらは ``enable_obs_delay`` (連続遅延 + 一様ノイズ) を **使わない**ので、
    パイプラインが二重に載ることが無い。

    引き継ぎ元は stage 1 の checkpoint (履歴 → 履歴なので warm start 不要)。

    .. note::
       ``ITER`` の既定は 8000 と長い。カリキュラムの終点が lon_span の第 3 段
       (5000 iteration) にあることに加えて、**ロブは apex がなかなか飽和しない** —
       loop_shoot 系では 10000 iteration を超えても apex が上がり続けていた。
       途中で止めた値を「頭打ち」と読まないこと。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_lob_plant_recipe(self)

        # 必ず最後。policy グループの構成が固まってから (N, H, 55) に変える。
        enable_obs_history(self)


@configclass
class K1WalkLobPlantEnvCfg_PLAY(K1WalkLobPlantEnvCfg):
    """Stage 2 の PLAY。

    ``enable_corruption = False`` は ObsTerm の ``noise`` しか切らないので、
    :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` が関数側へ移した
    ジッタは別途 0 にする。遅延とサンプル&ホールドは観測パイプラインの構造
    (= PLAY で見たいもの) なので残す。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)
        _disable_ball_obs_jitter(self)


# =========================================================================== #
# Stage 3: rough + ボール物性 DR の拡大 (実機へ寄せる最終段)
# =========================================================================== #
@configclass
class K1WalkLobPlantRoughEnvCfg(K1WalkLobPlantEnvCfg):
    """Stage 3: 凹凸地形 (±1-4 cm) とボール DR の 4 点セットを載せる fine-tune 段。

    ``__post_init__`` の順序::

        super()                  … stage 2 一式 (履歴化まで済んでいる)
        pin_curricula_at_end()   … 全カリキュラムを終値へ固定して項ごと None に
        _apply_rough_terrain()   … 地形だけ凹凸へ
        apply_ball_param_dr()    … ボール DR の 4 点セット (帯は loop_shoot 相当)

    ``super()`` が既に :func:`enable_obs_history` を掛け終わっているが問題ない。
    :func:`~..walk_kick.walk_kick_env_cfg._apply_rough_terrain` が触るのは
    ``scene.terrain`` の 3 属性と ``events.reset_ball.params["spawn_clearance"]`` だけ、
    :func:`~..walk_weak_kick_orbit.orbit_mods.apply_ball_param_dr` が触るのは
    events と ``soccer_ball.spawn.rigid_props`` だけで、どちらも観測グループには
    一切触らない。

    なぜカリキュラムを固定するのか
    ------------------------------
    この段は ``--load_pretrained`` で **収束済み** checkpoint (stage 2) から始める。
    ``--load_pretrained`` は ``--resume`` と違って ``common_step_counter`` を
    引き継がず 0 から数え直すので、カリキュラムを生かしたままだと全ランプが巻き戻る:

    * キック報酬 5 項 (direction / loft / elevation / foot_lift / plant_lon /
      plant_yaw) が weight 0 からフェードインし直す。この間 ``kick_finished`` は
      「残りの歩行報酬を捨てるコスト」だけを課すので **最初の 500 iteration は
      蹴らない方が得**が明示的に成立し、蹴らなくなった後に weight が戻ってきても
      払われる先が無い。
    * ``kick_plant_lon`` の ``lon_span`` が 0.25 → 0.60 に戻り、勾配が 24 → 10 に鈍る。
    * **``kick_velocity_strong`` が満額で復活する。** これがいちばん危ない。
      あの項は「速いほど得」= 低い弾道のトーキックを名指しで要求する呼び水で、
      退場させたのが stage 2 の肝 (:func:`_apply_lob_plant_recipe` の第 3 節)。
      折れ線の最終 knot は (1200, 0.0) なので、固定すると正しく 0 になる。

    :func:`~..walk_kick.curriculum_pin.pin_curricula_at_end` は終値を対象へ直接
    書き込んでから項ごと ``None`` にする。固定できない ``func`` が 1 つでも残って
    いれば ``NotImplementedError`` で落ちるので、**新しいカリキュラムを足したのに
    固定の仕方を書き忘れる**という事故は起動時に捕まる。

    ``expansion_alpha`` は渡さない (このタスクは
    :func:`~..walk_kick.mdp.curriculums.kick_rate_gated_expansion` を使っていない)。

    ボール DR について — 何が上書きされるか
    ---------------------------------------
    lob の系列は **既に** ボール物性 DR を持っている。
    :class:`~..walk_loop_shoot.walk_loop_shoot_env_cfg.K1WalkLoopShootEnvCfg` が
    ``events.ball_physics_material`` / ``events.ball_mass`` を startup イベントとして
    入れており、範囲は静摩擦 0.3-1.0 / 動摩擦 0.2-0.8 / 反発 0.0-0.7 / 質量 ×0.9-1.15。

    :func:`~..walk_weak_kick_orbit.orbit_mods.apply_ball_param_dr` はこの 2 つを
    **EventTerm ごと作り直して上書き**し、さらに 3 点を足す:

    ======================================  ==================================================
    足されるもの                              内容
    ======================================  ==================================================
    足の反発係数 DR                          ``events.physics_material`` の
                                             ``restitution_range`` を (0.3, 0.7) に。
                                             足↔地面は combine_mode multiply × 0 で無効なので、
                                             **足↔ボールの実効反発だけ**が振れる。
    ボールの初期回転                          ``reset_ball`` に ``spin_from_speed`` と
                                             ``rand_spin_range`` (0-5 rad/s) を足す。
    転がりの減速                              ``soccer_ball.spawn.rigid_props`` を
                                             ``angular_damping = 0.5`` に差し替える。
    ======================================  ==================================================

    渡している 4 つの範囲は inside の stage 3 と同じ定数
    (:data:`~..walk_inside_kick.walk_inside_kick_env_cfg._ROUGH_BALL_STATIC_FRICTION_RANGE`
    ほか) を import したもので、その値は元をたどれば walk_loop_shoot の同名定数。
    **したがって物性の帯は lob が元から持っていたものと数値上同一** で、この呼び出しの
    正味の効果は「上の 3 点が足されること」。範囲を明示的に渡しているのは、
    ``apply_ball_param_dr`` の既定が orbit の narrow な帯 (静摩擦 0.3-0.7 / 動摩擦
    0.2-0.5 / 反発 0.2-0.7) で、**既定のまま呼ぶと lob の帯が狭められてしまう**ため。

    .. warning::
       **凹凸地形 + ボールの組み合わせはこのリポジトリで学習を通した実績が無い**
       (``k1_walk_kick_rough`` / ``k1_walk_lob_rough`` 系の完走 log が存在しない)。
       必ず平坦の stage 2 を先に通し、その checkpoint から入ること。立ち上がりで
       ``kick_rate`` が落ちるのは想定内だが、数百 iteration で戻ってこないなら
       地形が厳しすぎる。地形の振幅を下げる前に、まず stage 2 の checkpoint が
       本当に繋がっているかを起動ログの "Skipped N tensors" で確認すること。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. カリキュラムを終値へ固定して項ごと外す ---------------------- #
        pin_curricula_at_end(self)

        # -- 2. 地形だけ凹凸へ --------------------------------------------- #
        _apply_rough_terrain(self)

        # -- 3. ボール DR の 4 点セット (帯は loop_shoot 相当を明示) --------- #
        apply_ball_param_dr(
            self,
            static_friction_range=_ROUGH_BALL_STATIC_FRICTION_RANGE,
            dynamic_friction_range=_ROUGH_BALL_DYNAMIC_FRICTION_RANGE,
            restitution_range=_ROUGH_BALL_RESTITUTION_RANGE,
            mass_scale_range=_ROUGH_BALL_MASS_SCALE_RANGE,
        )


@configclass
class K1WalkLobPlantRoughEnvCfg_PLAY(K1WalkLobPlantRoughEnvCfg):
    """Stage 3 の PLAY。stage 2 の PLAY + generator 地形用のカメラ設定。

    :func:`~..locomotion.rough_env_cfg._apply_play_viewer` を足すのは
    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickRoughEnvCfg_PLAY` と同じ理由。
    terrain generator は env origin を地形グリッドに割り当てるので、既定の
    world 固定カメラ (原点を見つめたまま動かない) だと ``play.py --video`` に
    **地形しか映らない**。

    カリキュラムは親で全て ``None`` にしてあるので、PLAY でも
    ``common_step_counter`` 0 から巻き戻る心配は無い (項が 1 つも無い)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)
        _disable_ball_obs_jitter(self)
        _apply_play_viewer(self)


# =========================================================================== #
# 360° 系統 (stage 2b / 3b) — 全方位ロブ + fewa 実績準拠の観測ノイズ
#
# 既存の stage 2 / stage 3 (K1WalkLobPlantEnvCfg / K1WalkLobPlantRoughEnvCfg) は
# **1 行も変えない**。あちらの run
# (k1_walk_lob_plant/2026-08-23_02-19-00, k1_walk_lob_plant_rough/2026-08-23_08-05-22)
# は git 追跡下の checkpoint を持っていて、cfg を書き換えると「その model_*.pt が
# どの設定で出たのか」が読めなくなるため。新しい系統は継承で足す。
#
# 何を変えるのか (2026-08-23 のログ分析より)
# ------------------------------------------
# 1. **全方位化**。stage 2 は heading ±45° / half_angle 60° / dist 0.5-0.8 の
#    限定レンジで 4300 iteration に頭打ち (apex 0.615 → 7700 iteration まで平ら)。
#    残りを範囲の拡大に使う。
# 2. **観測ノイズを fewa 方式へ全面置換**。lob_plant には内界センサ (IMU /
#    エンコーダ) の遅延が 1 つも入っていなかった。fewa の stage 4 は
#    「凹凸 + 360° + フルノイズ」で方向誤差 7.1-7.9° / 追従 0.86 を 3 run 一致で
#    出しており (band6 / band6calm / band6grounded)、この組み合わせが成立する
#    ことの実証になっている。
#
# 拡大を壁時計ではなくゲートで進める理由と、そのゲートに apex を足した理由は
# :data:`_LOB_360_APEX_ADVANCE_ABOVE` のコメント。
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 拡大の始点 (= stage 2 の限定レンジそのもの) と終点 (= 全方位)。
#
# 値は :mod:`..walk_inside_kick` の同名定数と同一。inside から import せずに
# 書き下しているのは、あちらが「インサイドキックの拡大」用に調整を入れたときに
# こちらが黙って追随しないようにするため (段の性格が違う)。
# --------------------------------------------------------------------------- #
_LOB_360_BALL_HALF_ANGLE_RANGE = (1.047, math.pi)
_LOB_360_BALL_DIST_START = (0.5, 0.8)
_LOB_360_BALL_DIST_END = (0.5, 1.5)
_LOB_360_HEADING_HALFWIDTH_RANGE = (math.pi / 4, math.pi)

# --------------------------------------------------------------------------- #
# 拡大ゲートの窓 [iteration]
#
# 始点 200: この段は **収束済みの stage 2 checkpoint から入る** ので、inside の
# 500 (ゼロから型を発見する段) ほど待つ必要がない。ただし --load_pretrained は
# 環境が変わった直後に必ず一度崩れるので、その復帰ぶんだけは置く。
# 終点 3000: α が 0 → 1 に届くまでの最短時間。ゲートが閉じている間は進まないので
#            実際にはこれより長くかかる (窓であって予定ではない)。
# --------------------------------------------------------------------------- #
_LOB_360_EXPANSION_START_ITER = 200
_LOB_360_EXPANSION_END_ITER = 3000

# --------------------------------------------------------------------------- #
# apex 込みゲートの閾値 (ユーザー指示 2026-08-23: 「緩めの apex 込みゲート」)
#
# **なぜ kick_rate だけでは駄目か。** :func:`~..walk_kick.mdp.curriculums.kick_rate_gated_expansion`
# が既定で見る ``kick_rate`` は「蹴れたか」しか測らない。ロブを捨ててトーキックで
# 転がしても 1.0 のままなので、apex が 1 度も立ち上がらないままゲートだけが
# 全方位へ開き切る。ロブ系にそのまま持ち込むといちばん起きやすい失敗。
#
# 閾値の根拠は stage 2 の実測 (run 2026-08-23_02-19-00):
#
#   * 収束値 apex 0.60 (4300 iteration でピーク 0.615、7700 まで平ら)
#   * 前進 0.40 = その 2/3。全方位へ広げれば apex はある程度落ちるので、
#     「stage 2 の 2/3 を保てているなら広げてよい」という緩さにする。
#   * 後退 0.25 = 凹凸段 (k1_walk_lob_plant_rough/2026-08-23_08-05-22) が転移で
#     壊れたときの底 0.24 相当。**そこまで落ちたら明確に壊れている**ので戻す。
#
# 前進は kick_rate と AND、後退はどちらか一方で OR。緩めたいときは前進側
# (0.40) を下げること。後退側 (0.25) を下げると「壊れても戻らない」になる。
# --------------------------------------------------------------------------- #
_LOB_360_APEX_ADVANCE_ABOVE = 0.40
_LOB_360_APEX_RETREAT_BELOW = 0.25

# 接近圧 (approach_penalty) / 回り込み圧 (ball_avoidance) の終値。inside と同じ。
_LOB_360_APPROACH_END_WEIGHT = -3.0
_LOB_360_AVOIDANCE_END_WEIGHT = -3.0

# --------------------------------------------------------------------------- #
# エピソード長 [s]
#
# 15.0 は全方位版の共通値 (K1WalkKick360EnvCfg / walk_inside_kick /
# walk_long_pass_fewa と同じ)。ゲートが開き切ると 1.5 m + 半周の回り込みで移動が
# 2.5-3 m になるので、stage 2 の 10.0 のままだと時間切れが増える。
#
# **段の途中で変えない。** 同じ run の中でエピソード長が変わると「時間切れの
# 起きやすさ」が変わり、拡大の効果と混ざって読めなくなる (inside の第 3 節と同じ判断)。
# --------------------------------------------------------------------------- #
_LOB_360_EPISODE_LENGTH_S = 15.0


def _apply_lob_360_expansion(cfg) -> None:
    """限定レンジ → 全方位を 1 本の α で進める拡大ゲートを載せる。

    **:func:`~..walk_kick.curriculum_pin.pin_curricula_at_end` の後に呼ぶこと。**
    あちらの docstring は「固定した後に新しいランプを足すな」と書いているが、
    ここでは **それが意図** — この段で唯一生かしたいカリキュラムが拡大ゲートで、
    他の全ランプ (キック報酬のフェードイン / strong の折れ線 / σ_velocity /
    lon_span) は収束済み checkpoint に合わせて終値で固定しておきたい。

    inside の :func:`~..walk_inside_kick.walk_inside_kick_env_cfg._apply_inside_kick_recipe`
    の第 3-5 節をロブ用に移したもので、違いは 3 点:

    1. ゲートに ``apex_metric_name="kick_apex_height"`` を渡す (理由は
       :data:`_LOB_360_APEX_ADVANCE_ABOVE` のコメント)。
    2. ``approach_fade_iterations=0``。inside は 0 → 500 iteration で接近圧を
       立ち上げるが、こちらは **既に接近も蹴りもできるポリシー**から始めるので、
       壁時計のフェードインを掛けると `pin_curricula_at_end` が終値 (-3.0) へ
       固定した接近圧が 0 に戻ってから 500 iteration かけて復帰する = 巻き戻る。
    3. ``kick_velocity_strong`` には触らない (stage 2 の呼び水は既に退場済み)。
    """
    # -- 1. エピソード長 ---------------------------------------------------- #
    cfg.episode_length_s = _LOB_360_EPISODE_LENGTH_S

    # -- 2. 拡大の始点を明示する (ここからゲートが動かす) -------------------- #
    cfg.events.reset_ball.params["half_angle"] = _LOB_360_BALL_HALF_ANGLE_RANGE[0]
    cfg.events.reset_ball.params["dist_range"] = _LOB_360_BALL_DIST_START
    cfg.commands.kick_direction.ranges.heading = (
        -_LOB_360_HEADING_HALFWIDTH_RANGE[0],
        _LOB_360_HEADING_HALFWIDTH_RANGE[0],
    )

    # -- 3. ball_avoidance を weight 0 で置く ------------------------------- #
    #
    # 「構えができるまでボールに寄るな」の抑止。全方位になってから効かせるので、
    # 重みはゲートが α に比例して立ち上げる。stage 2 (限定レンジ) には存在しない
    # 項なので、ここで新設する。
    cfg.rewards.ball_avoidance = RewTerm(
        func=mdp.ball_avoidance,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_sole": _BALL_AVOIDANCE_SIGMA_SOLE,
            "sigma_pose": _BALL_AVOIDANCE_SIGMA_POSE,
        },
    )

    # -- 4. 拡大ゲート ------------------------------------------------------ #
    #
    # ボール出現範囲・蹴り方向範囲・approach_penalty / ball_avoidance の重みを
    # 1 本の α で同時に動かす。approach_penalty の weight の書き手はこの関数 1 つに
    # 絞る (pin_curricula_at_end が curriculum.approach_penalty_weight を既に
    # None にしているので、二重書きにはならない)。
    cfg.curriculum.kick_expansion = CurrTerm(
        func=mdp.kick_rate_gated_expansion,
        params={
            "command_name": "kick_direction",
            "start_step": _LOB_360_EXPANSION_START_ITER,
            "end_step": _LOB_360_EXPANSION_END_ITER,
            "steps_per_iteration": _SPI,
            "apex_metric_name": "kick_apex_height",
            "apex_advance_above": _LOB_360_APEX_ADVANCE_ABOVE,
            "apex_retreat_below": _LOB_360_APEX_RETREAT_BELOW,
            "ball_event_name": "reset_ball",
            "half_angle_range": _LOB_360_BALL_HALF_ANGLE_RANGE,
            "dist_range_start": _LOB_360_BALL_DIST_START,
            "dist_range_end": _LOB_360_BALL_DIST_END,
            "heading_halfwidth_range": _LOB_360_HEADING_HALFWIDTH_RANGE,
            "approach_term_name": "approach_penalty",
            "approach_end_weight": _LOB_360_APPROACH_END_WEIGHT,
            "approach_fade_iterations": 0,
            "avoidance_term_name": "ball_avoidance",
            "avoidance_end_weight": _LOB_360_AVOIDANCE_END_WEIGHT,
        },
    )


def _apply_fewa_ball_obs(cfg) -> None:
    """観測ノイズを fewa (Stage 4) 方式へ **全面置換**する。

    入る中身は 4 つ:

    * IMU (``projected_gravity`` / ``base_ang_vel``) の遅延 ``[0, 0.02]`` s。
      group ``imu`` で乱数を共有する (同じセンサ読み出しなので独立には遅れない)。
    * エンコーダ (``joint_pos`` / ``joint_vel``) の遅延 ``[0, 0.02]`` s。group ``encoder``。
    * 視覚 (``ball_vel`` / ``prev_ball_pos``) の遅延 ``0.02 + [0, 0.06]`` = 0.02-0.08 s。
      group ``vision``。
    * ボール観測ノイズを一様 位置 ±0.07 m / 速度 ±0.5 m/s へ広げる。

    **内界センサの遅延はこの系列に 1 つも入っていなかった** (lob_plant / walk_lob /
    walk_loop_shoot のどれも ``enable_obs_delay`` を呼んでいない)。実機の IMU も
    エンコーダも「測ってから policy に届くまで」に遅れがあり、遅延ゼロで学習すると
    遅れた観測に過剰反応する方策になる。とくに ``base_ang_vel`` は歩行の安定化に直結する。

    ガウス認識パイプラインを先に剥がすこと
    --------------------------------------
    :class:`~..walk_lob.walk_lob_env_cfg.K1WalkLobEnvCfg` が ``__post_init__`` の
    最後で :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` を呼び、
    policy の ``prev_ball_pos`` を :func:`~..walk_kick.mdp.observations.noisy_ball_pos_b`
    (エピソードごとランダム遅延 2-6 step + 30Hz サンプル&ホールド + ガウスジッタ
    σ=0.067 / クリップ ±0.2) にしている。これを **素の** :func:`~..walk_kick.mdp.observations.prev_ball_pos_b`
    へ戻してから fewa 側を掛ける。戻さないと 2 つの壊れ方をする:

    1. ``params`` にガウス側のキー (``delay_step_range`` / ``camera_hz`` /
       ``jitter_std`` / ``jitter_clip``) が残る。fewa の ``enable_obs_delay`` は
       ``term.params = {**term.params, ...}`` と **マージ** するので、差し替え後の
       :func:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.delayed_prev_ball_pos_b`
       へ未知のキーワード引数として渡り ``TypeError`` になる。
    2. :mod:`..walk_kick_dual` 側の ``enable_obs_delay`` を使った場合は
       ``_PIPELINE_BALL_POS_FUNCS`` ガードに掛かって ``prev_ball_pos`` だけ黙って
       スキップされ、``ball_vel`` のノイズだけ広がった食い違い状態になる。

    なぜ :mod:`..walk_kick_dual` ではなく :mod:`..walk_long_pass_fewa` の
    ``enable_obs_delay`` を使うのか
    --------------------------------------------------------------------------
    あちらの ``_DELAYED_OBS_TERMS`` は both_feet の 2 スロット構成前提で
    ``ball_pos`` (3 次元) を含む。この系列は walk_kick 素の 55 次元レイアウト
    (``prev_ball_pos`` 1 スロット) なので、その項が無く ``AttributeError`` で落ちる。
    fewa 側はまさにこのレイアウト用のローカルコピーで、6 項ちょうど一致する。

    **観測の次元・並びは変わらない** (func / params / noise だけ差し替え) ので、
    ノイズを入れる前後で checkpoint はそのまま繋がる。
    """
    # -- 1. ガウス認識パイプラインを素へ戻す -------------------------------- #
    #
    # 継承元 (K1WalkKickPolicyCfg) の宣言そのものに戻す: func / params / noise の 3 点。
    policy = cfg.observations.policy
    policy.prev_ball_pos.func = mdp.prev_ball_pos_b
    policy.prev_ball_pos.params = {}
    policy.prev_ball_pos.noise = Unoise(n_min=-0.02, n_max=0.02)

    # -- 2. fewa の遅延 DR + ボールノイズ ----------------------------------- #
    #
    # noise は関数側が ±0.07 / ±0.5 へ上書きするので、上の ±0.02 は通過点。
    _fewa_enable_obs_delay(cfg, _FEWA_OBS_DELAY_MAX_S, _FEWA_BALL_OBS_DELAY_MAX_S)


# =========================================================================== #
# Stage 2b: 平坦 + 全方位 (apex 込みゲートで漸進)
# =========================================================================== #
@configclass
class K1WalkLobPlant360EnvCfg(K1WalkLobPlantEnvCfg):
    """Stage 2b: stage 2 の収束済み checkpoint から、限定レンジ → 全方位へ広げる。

    引き継ぎ元は stage 2 の checkpoint
    (``logs/rsl_rl/k1_walk_lob_plant/2026-08-23_02-19-00/model_7600.pt``)。
    履歴 → 履歴なので ``--warm_start_from_single_frame`` は不要。

    ``__post_init__`` の順序に意味がある::

        super()                    … stage 2 一式 (報酬・カリキュラム・履歴化まで)
        pin_curricula_at_end()     … 全ランプを終値へ固定して項ごと None に
        _apply_lob_360_expansion() … **その後**。唯一生かすカリキュラム = 拡大ゲート

    観測は stage 2 のまま (ガウス認識パイプライン)。ノイズの入れ替えは
    :class:`K1WalkLobPlant360RoughEnvCfg` (stage 3b) の仕事で、ここで一緒に変えると
    「全方位で落ちたのか、ノイズで落ちたのか」が切り分けられなくなる。

    TensorBoard で最初に見るもの
    ----------------------------
    * ``Curriculum/kick_expansion/alpha`` — 0 → 1。**止まっているのが正常な状態**も
      ある (ゲートが閉じている = 今の実力の上限)。3000 iteration 使っても 0.3 程度で
      止まるなら、全方位はこのポリシーには早い。
    * ``Curriculum/kick_expansion/apex_ema`` — 0.40 を割ると拡大が止まる。
      0.25 を割ると戻り始める。
    * ``Metrics/kick_direction/kick_apex_height`` — 基準は stage 2 の 0.60。
      α が進むほど落ちるのは想定内だが、0.25 まで落ちたらゲートが戻すはず。
      **戻らないのにここが 0.25 以下なら、ゲートが機能していない** (apex_ema は
      EMA なので実測より遅れる。100 iteration 単位で見ること)。
    * ``Metrics/kick_direction/kick_dir_error_deg`` — stage 2 の基準は 7.7°。
      全方位化で悪化するが、fewa は 360° + フルノイズで 7.1-7.9° を出している
      ので、15° を超えたまま戻らないなら回り込みが成立していない。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. カリキュラムを終値へ固定して項ごと外す ---------------------- #
        #
        # --load_pretrained は common_step_counter を 0 に戻すので、固定しないと
        # キック報酬のフェードインからやり直しになる (= 最初の 500 iteration は
        # 蹴らない方が得)。詳細は pin_curricula_at_end の docstring。
        pin_curricula_at_end(self)

        # -- 2. 拡大ゲートだけを載せ直す ------------------------------------ #
        _apply_lob_360_expansion(self)


@configclass
class K1WalkLobPlant360EnvCfg_PLAY(K1WalkLobPlant360EnvCfg):
    """Stage 2b の PLAY。

    観測は stage 2 と同じガウス認識パイプラインなので、後始末も stage 2 と同じ
    (``enable_corruption = False`` は ObsTerm の ``noise`` しか切らないため、
    関数側へ移したジッタは :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter`
    で別途 0 にする)。

    .. note::
       拡大ゲート (``curriculum.kick_expansion``) は PLAY でも生きている。
       CurriculumManager は PLAY でも ``common_step_counter`` 0 から走るので、
       **PLAY で見えるのは α = 0 = 限定レンジ**になる。全方位の挙動を見たいときは
       :class:`K1WalkLobPlant360RoughEnvCfg_PLAY` (α を 1 に固定済み) を使うか、
       ``env.curriculum.kick_expansion = None`` にしたうえで
       ``events.reset_ball`` / ``commands.kick_direction.ranges.heading`` を
       手で全方位へ書くこと。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)
        _disable_ball_obs_jitter(self)


# =========================================================================== #
# Stage 3b: 凹凸 + ボール DR + fewa 方式の観測ノイズ (実機へ寄せる最終段)
# =========================================================================== #
@configclass
class K1WalkLobPlant360RoughEnvCfg(K1WalkLobPlant360EnvCfg):
    """Stage 3b: stage 2b の上に、凹凸地形・ボール DR・fewa のノイズを載せる。

    引き継ぎ元は stage 2b の checkpoint (履歴 → 履歴、warm start 不要)。

    ``__post_init__`` の順序::

        super()                  … stage 2b 一式 (拡大ゲートが 1 本だけ生きている)
        pin_curricula_at_end()   … その拡大ゲートを α = 1 (全方位) で固定して外す
        _apply_rough_terrain()   … 地形だけ凹凸へ (±1-4 cm)
        apply_ball_param_dr()    … ボール DR の 4 点セット (足の反発 / 物性 / 初期回転 /
                                   転がり減速)
        _apply_fewa_ball_obs()   … **最後**。観測の差し替えは他が全部済んでから

    2 回目の ``pin_curricula_at_end`` が拡大ゲートを畳む
    ---------------------------------------------------
    1 回目 (stage 2b) は全ランプを固定し、その後に拡大ゲートを 1 本足した。
    ここでもう一度呼ぶと、残っているのはその 1 本だけなので
    :func:`~..walk_kick.curriculum_pin.pin_expansion_gate` が α = 1 の値
    (half_angle π / dist 0.5-1.5 / heading ±π / approach 0 / avoidance -3.0) を
    直接書き込んで項ごと外す。**stage 2b で α が 1 に届いていることが前提** —
    届いていない checkpoint から入ると、実力より広い範囲を「ゲートが戻せない
    状態で」固定することになる。``Curriculum/kick_expansion/alpha`` を必ず確認すること。
    届いていないなら ``expansion_alpha`` にその値を渡す。

    足の反発 DR は残す (ユーザー判断 2026-08-23)
    -------------------------------------------
    :func:`~..walk_weak_kick_orbit.orbit_mods.apply_ball_param_dr` の第 1 項は
    ロボット全 body の反発係数を (0.3, 0.7) にする。足↔ボールは average で効くので
    実効反発が stage 2 の 0.00-0.35 から 0.15-0.70 へ倍増し、既存 stage 3 の
    apex 崩壊 (0.60 → 0.24) の最有力候補だったが、**実機へ寄せる方向としては
    正しい**ため入れたまま進める判断になった。apex がまた落ちるようなら、
    ここを (0.0, 0.0) に戻す ablation が最初の切り分けになる。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 拡大ゲートを α = 1 (全方位) で固定して外す ------------------ #
        pin_curricula_at_end(self, expansion_alpha=1.0)

        # -- 2. 地形だけ凹凸へ --------------------------------------------- #
        _apply_rough_terrain(self)

        # -- 3. ボール DR の 4 点セット (帯は loop_shoot 相当を明示) --------- #
        apply_ball_param_dr(
            self,
            static_friction_range=_ROUGH_BALL_STATIC_FRICTION_RANGE,
            dynamic_friction_range=_ROUGH_BALL_DYNAMIC_FRICTION_RANGE,
            restitution_range=_ROUGH_BALL_RESTITUTION_RANGE,
            mass_scale_range=_ROUGH_BALL_MASS_SCALE_RANGE,
        )

        # -- 4. 観測ノイズを fewa 方式へ全面置換 ---------------------------- #
        #
        # 必ず最後。ガウスパイプラインを剥がしてから掛けるので、他の変更が
        # 観測に触らないことが前提になる。
        _apply_fewa_ball_obs(self)


@configclass
class K1WalkLobPlant360RoughEnvCfg_PLAY(K1WalkLobPlant360RoughEnvCfg):
    """Stage 3b の PLAY。

    観測は fewa 方式 (連続遅延 + 一様ノイズ) に差し替わっているので、
    :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter` は
    **呼んではいけない** — あちらは ``prev_ball_pos.params["jitter_std"] = 0.0`` を
    書き込む関数で、差し替え後の ``delayed_prev_ball_pos_b`` には存在しない
    キーワード引数なので ``TypeError`` になる。一様ノイズは
    ``enable_corruption = False`` だけで落ちる (ObsTerm の ``noise`` そのものなので)。
    遅延は観測パイプラインの構造なので PLAY でも残す。

    generator 地形用のカメラ設定を足すのは
    :class:`K1WalkLobPlantRoughEnvCfg_PLAY` と同じ理由 (world 固定カメラだと
    ``play.py --video`` に地形しか映らない)。

    カリキュラムは親で全て ``None`` になっているので、PLAY でも巻き戻らない
    (拡大ゲートも 2 回目の ``pin_curricula_at_end`` で畳まれている = 全方位が見える)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_play_tweaks(self)
        _apply_play_viewer(self)
