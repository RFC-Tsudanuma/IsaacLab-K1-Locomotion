# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 右足インサイドキック環境 (足の内側の面でボールに当てるキックを学習する)。

なぜ新しいタスクが要るか
------------------------
既存のキックタスクは全て **トーキック** (つま先で突く形) に収束している。
``Metrics/kick_direction/kick_foot_right_frac`` が 9/9 の run で右足に寄っている
のと同じくらいはっきりした偏りで、しかも偶然ではない。原因は報酬の側にある。

* ``p_style`` (胴体の向きが蹴り方向にどれだけ正対しているか) が項1-3 の
  ``r_direction`` に掛かっており、素の定義は ``clamp(forward·kick_dir, 0, 1)``。
  インサイドで当てるには胴体が蹴り方向から 30-45° ずれた向きで構えるのが自然な形
  なので、cos40° = 0.77、つまり **キック報酬が 2 割減る**。報酬側が
  「胴体を蹴り方向へ正対させろ」= つま先で蹴る構えを名指しで要求していた。
* 当たり所を見る項が 1 つも無い。``kick_direction`` / ``kick_velocity_*`` は
  「どこへ・どれだけ速く飛んだか」しか見ないので、つま先で当てても面で当てても
  同じ点が付く。同じ点なら、股関節をひねらずに済むトーキックの方が先に見つかる。

このタスクは (1) その逆風を外し、(2) 接触の幾何を直接採点する項で順風を入れる。

実機でインサイドにしたい理由は接触面の広さ。つま先はボールとの接触が事実上 1 点なので、
ボール位置の認識誤差 (実測 3cm 級) がそのまま当たり所のずれ = 方向誤差になる。
足の側面は前後に 18cm 以上あるので、同じ量ずれても当たり所の法線がほとんど変わらない。

このタスクの 2 つの入口
-----------------------
1. **逆風を外す — p_style の帯** (``style_halfwidth`` = 0.698 rad = 40°)

   ``:func:`~..walk_kick.mdp.kick_state.kick_state``` の p_style を
   「角度差が ±40° 以内なら一律 1、そこを超えた分だけ緩やかに減衰」に変える。
   帯の中では胴体の向きが報酬に一切効かなくなるので、構えを決めるのは当たり所の項
   だけになる。40° はインサイドキックで胴体が蹴り方向からずれる自然な量
   (30-45°) の中央。

   帯の外を崖 (0) にせずガウスで落とすのは、帯の外にいる方策にも
   「今より少しでも正対する」という勾配を残すため。

2. **順風を入れる — kick_inside_contact**

   値 latch を起こした接触 (= キック本体) の幾何を凍結値で採点する
   (:func:`~..walk_kick.mdp.rewards.kick_inside_contact`)::

       r_direction × f_perp × f_side × (右足で蹴った) × (接触が計測済み)
       f_perp = clamp((1 − |足の前方向·kick_dir|) / (1 − 0.34), 0, 1)
       f_side = clamp(ボールの足ローカル y / 0.035, 0, 1)

なぜ f_perp と f_side が線形クランプなのか (Gaussian・階段にしないこと)
----------------------------------------------------------------------
``d_sat = 0.34`` = cos70°。足がキック方向から 70° 以上横を向いた当たりで満点、
そこから連続に落ちて、|dot| = 1 (完全なトーキック) でようやく 0 になる。
**学習の出発点であるトーキックの位置に、値は小さくても勾配が残る**ことが肝。

Gaussian にすると σ の外で勾配が完全に死ぬ。walk_lob 系 5 run の
``kick_plant_foot`` (f_lon が Gaussian) で実際にこれが起きていて、plant_lon は
−0.36〜−0.43 に居座ったまま一度も目標へ寄らず、σ を広げて目標を引っ張った版でも
gap が開いただけだった。同じ失敗を繰り返さないため、この形は変えないこと。

f_side も同じ理由で線形。初回 run (2026-08-21_03-49-24) では f_side を
``ball_side > 0`` の 0/1 にしてしまっており、「当たり所をもう少し内側へ寄せる」
方向に勾配がまったく無かった。結果、内側で当たった割合
(``kick_inside_contact`` を ``kick_direction`` と右足率で割り戻した値) は
iter 320 の 0.073 から iter 997 の 0.036 へ **半減** し、トーキックへ戻っていった。
分母の 0.035 は足箱の半幅で、ボール中心が足のローカル +y へ半幅ぶん入ったら満点。

なぜ kick_plant_foot を外すのか
-------------------------------
middle のレシピ (:func:`~..walk_middle_kick.walk_middle_kick_env_cfg._apply_middle_kick_recipe`)
は「蹴る瞬間に軸足がボールの真横」を狙う ``kick_plant_foot`` を入れているが、
このタスクでは **入れた直後に外す**。

walk_lob 系 5 run の実測で、``Metrics/kick_direction/plant_lon`` は
−0.36〜−0.43 に居座ったまま一度も目標 (−0.03) に漸近しなかった。σ_lon を広げて
目標を引っ張った版でも、目標との gap が開いただけで実測は動かなかった。
**軸足の位置は原因ではなく蹴り方の結果** だと考えるのが素直で、結果を報酬で直接
引っ張っても動かない。インサイドの構えでは軸足の置き所も変わるはずなので、
トーキック前提で決めた目標値 (lon −0.03 / lat 0.19) をそのまま押し付ける理由もない。

外したあとも ``Metrics/kick_direction/plant_lon`` / ``plant_lat`` は出続けるので、
**報酬ではなく指標として** 見ること。インサイドが立ち上がったときに軸足がどこへ
移動したかは、次にどんな項を書くかの材料になる。

(この「指標としてだけ見る」は初版の判断で、実機のフィードバックを受けて次の節の
とおり **形を変えて報酬に戻した**。``kick_plant_foot`` 自体は外したままである。)

軸足をもう一度誘導することになった経緯 (kick_plant_lon の新設)
--------------------------------------------------------------
インサイドの型は run 2026-08-21_05-00-22 で立ち上がった (foot_kick_dot ≈ 0 /
ball_side 0.145 / vel_ratio 0.91)。ところが実機で **「振りが手前すぎてボールを
巻き込んで転ぶ」** 事故が出た。軸足がボールより後ろにあると振り足もボールの手前側を
通るので、ボール認識が 3cm ずれるとボールが進行方向に残り、そこへ踏み込んで転ぶ。
軸足がボールの真横にあれば、同じ量ずれてもボールは体の横を抜けていく。

1 回目の対策は **報酬を足さず指令側だけ動かす**もので、失敗した。
``r_stance`` (P_kick = 胴体の終着指令をボール後方どれだけに置くか) を 0.20 → 0.10 に
詰め、model_4999 から ``--resume`` で 6700 iteration の fine-tune を回した
(run 2026-08-21_10-41-17)。結果 **plant_lon は −0.23 のまま完全に不動**
(他の指標は無傷: vel_ratio 0.94 / dir_error 4.2° / touch 1.03)。

原因は、P_kick が報酬にほとんど流れていないこと:

* キック報酬は **飛翔の凍結値** で採点されるので、胴体がどこに立っていたかを見ない。
* ``p_style`` は帯 (40°) にしてあるので、帯の中では胴体の向きも報酬に効かない。
* P_kick へ体を寄せる圧は ``ball_avoidance`` の pose_match 経由だけで、
  実測 −0.01 / episode しかない。

過去に見えていた「r_stance → plant_lon」の相関 (0.25 → −0.39、0.20 → −0.23) は
**ゼロから学習した run 同士の比較**であって、発見期に接近の誘導が型を形作った結果。
収束済みポリシーを動かすレバーではなかった。

そこで設計原則を分けた: **``r_stance`` は胴体の終着指令であり、その役割は「望む構えを
妨害しないこと」まで。軸足の誘導は足そのものを測る別の項が持つ。** ``r_stance`` は
0.10 のまま触らず、:func:`~..walk_kick.mdp.rewards.kick_plant_lon` を新設した。

これは上で外した ``kick_plant_foot`` の復活ではない。あちらの死因は特定できていて、
**σ_lon = 0.10 の Gaussian が実測 −0.42 で f ≈ 5e-4 = 真っ平ら**だったこと (と、
収束済みポリシーへの後掛けだったこと) である。``kick_plant_lon`` は線形テント
(半幅 0.45) なので、−0.42 で f = 0.13、−0.23 で f = 0.56 と、ポリシーが実際に
居る場所に傾きが残る。このタスクの ``kick_inside_contact`` の f_perp (線形クランプ) が
foot_kick_dot を 0.9 → 0.0 まで動かし切った実績と同じ流儀。
lat (横) は掛けない — plant_lat 0.30 は別の課題であり、掛けると「横が外れている
あいだ lon の勾配も死ぬ」という kick_plant_foot の失敗を作り直すことになる。

実機フィードバック 2 回目 (2026-08-23)
--------------------------------------
1 回目 (「振りが手前すぎてボールを巻き込んで転ぶ」→ ``kick_plant_lon`` の新設) の
あと、実機で残っていた症状は **当たりが薄い / 空振りする** と **軸足がまだ手前**
の 2 つ。3 つの変更で対処する。すべて :func:`_apply_inside_kick_recipe` に入れるので、
``K1WalkInsideKickDualEnvCfg`` (stage 2) / ``K1WalkInsideKickDualRoughEnvCfg``
(stage 3) にも継承で自動的に載る (**history 版と通常版の両方に入れる**)。

1. **軸足の向き** (``kick_plant_yaw``、weight 3.0、第 6c 節)

   軸足のつま先を蹴り方向へ向かせる。軸足が斜めのまま立つと骨盤もそちらを向き、
   振り足のインサイド面がキック線に正対しないので当たりが薄くなる。角度に対する
   線形テント (半幅 90°)。胴体は p_style の帯で 30-45° ずれてよく、**ずれるのは
   胴体で軸足は向かせる** という役割分担 (分ける自由度は Hip_Yaw で、このタスクは
   ``joint_deviation_hip`` から外してある)。

2. **軸足の目標を 0 cm へ + span の第 3 段** (第 6b 節)

   ``lon_target`` −0.03 → 0.0 (足首がボール真横 = 足箱の中心は 2.6 cm 前) と、
   ``lon_span`` の折れ線に 3000 → 4000 で 0.25 → 0.15 の第 3 段を足す。
   **目標を動かしてもテントの勾配 (W/span) は変わらないので、目標だけでは政策は
   ほぼ動かない。** 踏み込ませる実際のレバーは span で、2 つはセットで入れる。

3. **足を上げすぎない上限** (``kick_foot_ceiling``、weight 3.0、第 6d 節)

   足裏高さ 0.09 (足箱の中心がボール中心以下に来る高さ) 以下は一律満点、そこから
   0.08 かけて 0 へ。**天井だけで下向きの圧を持たない**のが肝で、「低いほど得」の
   ``kick_contact_height`` は walk_lob で反証済み (スイング長 = ボール速度を削る)。

TensorBoard で見るもの (この 3 つを入れた run)
---------------------------------------------
* ``Metrics/kick_direction/plant_yaw_dot`` — **新設。まず初期値を記録すること。**
  現行ポリシーが軸足を何度向けているかの記録がまだ無い (1 = 蹴り方向、0 = 真横、
  0.87 = 30°、0.71 = 45°)。1 iteration 目の値が「素の値」で、``kick_plant_yaw`` が
  そこからどれだけ動かせたかの基準になる。素の値が既に 0.9 級なら、この項は
  効きようが無いので weight ではなく ``yaw_span`` を絞る側で考え直す。
* ``Metrics/kick_direction/sole_height_at_kick`` — ``kick_foot_ceiling`` の対象。
  0.09 以下なら満点なので、**下がり続ける必要は無い**。0.09 を割ったあとも下がり
  続けているなら、それはこの項ではなく別の圧 (スイングの都合) で下がっている。
* ``Metrics/kick_direction/plant_lon`` — −0.107 から 0 側へ動くか。第 3 段の窓
  (3000 → 4000) は他のカリキュラムが全部終わったあとなので、そこでの変化は
  span 以外に原因が無い。
* ``Metrics/kick_direction/kick_vel_ratio`` — **威力とのトレード**の監視。0.887 が
  基準。形の項を 3 つ (plant_lon 6.0 / plant_yaw 3.0 / foot_ceiling 3.0) 積んだので、
  「形のためにキックそのものを削る」が起きていないかはここでしか見えない。
  ratio が落ちているなら、いちばん新しく足した項から weight を下げる。

なぜ段を分けないのか
--------------------
weak / middle 系は「限定レンジ → 全方位 → 観測ノイズ」を stage で分け、段ごとに
checkpoint を引き継いでいる。このタスクは **1 段で回す**。

* 全方位への拡大は壁時計ではなく **キック成立率のゲート** で進める
  (:func:`~..walk_kick.mdp.curriculums.kick_rate_gated_expansion`)。段を分けるのは
  「難しくしすぎたら前段からやり直す」ための保険だが、ゲートは崩れた時点で自分で
  止まり、崩れ続ければ蹴れていた範囲まで戻るので、その保険が要らない。
* ボール観測のノイズ+遅延は **最初から入れる**。ノイズは「進む軸」ではないので、
  難しくしすぎて収支が逆転する (キック報酬が消えて「蹴らずに歩く」が最適になる)
  という段分けの本来の理由が当てはまらない。ノイズが乗っていても latch は発火し、
  キック報酬は払われ続ける。むしろ発見期からノイズ込みで型を作った方が、
  「ノイズの無い世界でだけ通用する精密な当て方」に収束しない。

学習手順 (1 段)::

    ./scripts/rsl_rl/train_walk_inside_kick.sh

``--resume`` は使わないこと (common_step_counter が同期され、キック報酬の
フェードインが「もう終わった」と判定されてランプしない)。``--reset_noise_std`` も
使わないこと (歩行 checkpoint のスイングを壊す)。

stage 2 / stage 3 (dual history / rough + DR)
---------------------------------------------
上の「なぜ段を分けないのか」は **型を発見するまで** の話。インサイドの型は
run 2026-08-22_11-56-42 (3600 iteration) で収束した::

    Curriculum/kick_expansion/alpha      1.0   (全方位に到達済み)
    Metrics/kick_direction/kick_rate     0.998
    Metrics/kick_direction/plant_lon     -0.107   (初版 -0.23 から目標 -0.03 側へ)
    Metrics/kick_direction/foot_kick_dot -0.030   (足が真横 = 側面で当てている)
    Metrics/kick_direction/kick_vel_ratio 0.887

ここから先は「同じ型のまま実機へ寄せる」フェーズで、こちらは **fine-tune の段** に
分ける。1 run でまとめて掛けると、崩れたときに履歴のせいなのか地形のせいなのか
DR のせいなのか切り分けられないため。

* **stage 2** = :class:`K1WalkInsideKickDualEnvCfg` (平坦、観測履歴のみ)

  actor の入力を 100 フレームの履歴にする (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)。
  **変えるのはそれだけ。** 報酬・コマンド・地形・ボール DR は stage 1 と同一なので、
  ここで指標が動いたら原因は履歴以外にあり得ない。

* **stage 3** = :class:`K1WalkInsideKickDualRoughEnvCfg` (凹凸 + ボール物性 DR 拡大)

  stage 2 の上に :func:`~..walk_kick.walk_kick_env_cfg._apply_rough_terrain` (±1-4 cm の
  ランダム凹凸) と、ボール物性 DR の帯を walk_loop_shoot 相当まで広げたものを載せる。

カリキュラムは段に入る前に終値へ固定する (:func:`_pin_curricula_at_end`)
--------------------------------------------------------------------------
stage 2/3 は ``--load_pretrained`` で **収束済み** checkpoint から始める。あちらは
``common_step_counter`` を 0 に戻すので、カリキュラムを生かしたままだと全ランプが
巻き戻る (キック報酬は 0 から、拡大ゲートは限定レンジから、σ_velocity は 1.0 から、
``lon_span`` は 0.15 から 0.45 へ、strong は満額から)。その帰結が
:func:`~..walk_kick_dual.walk_kick_dual_env_cfg._freeze_fade_in_curricula` の docstring に
書かれている「最初の 500 iteration は蹴らない方が得」で、蹴らなくなった後に weight が
戻ってきても払われる先が無い。詳細と、なぜ ``_freeze_fade_in_curricula`` ではなく
**項ごと None にする** のかは :func:`_pin_curricula_at_end` の docstring。

checkpoint の橋渡し
-------------------
stage 1 の checkpoint は **1 フレーム観測** (素の ``ActorCritic``) なので、履歴 actor
(``ActorCriticHistoryCNN``) には形が合わず、そのままでは actor が 1 本も引き継がれない。
``--warm_start_from_single_frame`` を付けると旧 actor を履歴 actor の
「最新フレームの列」へ移植するので、学習開始時点の出力が stage 1 のポリシーと一致する
(:func:`~..locomotion.networks.remap_single_frame_actor`)。critic は 1 フレームのまま
(このタスクは 61 次元) なので無加工でそのまま載る。
通しスクリプト :file:`scripts/rsl_rl/train_walk_inside_kick_dual.sh` は checkpoint に
``actor.0.weight`` があるかを見てこのフラグを自動で付ける。
stage 2 → stage 3 は履歴 → 履歴なのでフラグは不要 (付かない)。

TensorBoard で最初に見るもの (stage 2/3 共通)
---------------------------------------------
出発点が収束済みなので、**伸びしろより「壊れていないこと」を先に見る**。
上に挙げた run 2026-08-22_11-56-42 の値が基準:

* ``Metrics/kick_direction/plant_lon`` ≈ −0.11。0.05 以上マイナス側へ戻る
  (= −0.16 より後ろ) なら軸足の踏み込みを失っている。実機の転倒事故の直接原因なので、
  ここが戻ったら他がどれだけ良くても採用しない。
* ``Metrics/kick_direction/foot_kick_dot`` ≈ 0。1 に向かって上がっていたら
  インサイドを捨ててトーキックへ戻っている。
* ``Metrics/kick_direction/kick_vel_ratio`` ≈ 0.88。
* ``Metrics/kick_direction/kick_rate`` ≈ 1.0。stage 3 は地形と DR が乗るので
  最初の数百 iteration は落ちてよいが、戻ってこなければ地形が厳しすぎる。

いずれも 1 iteration 目からほぼ基準値のはずで、**そうなっていなければ
checkpoint が繋がっていない** (起動ログの "Skipped N tensors" を見ること)。

禁止フラグ (stage 1 と同じ)
---------------------------
* ``--resume`` — experiment_name が段ごとに違うので前段を検出できないうえ、
  ``common_step_counter`` を同期してしまう。段の引き継ぎは常に ``--load_pretrained``。
* ``--reset_noise_std`` — 収束済みの std (0.06-0.1 級) を戻すと当たり所の精度が壊れる。
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..locomotion.mdp.curriculums import (
    lin_vel_command_curriculum,
    modify_command_resampling_time_range,
    modify_push_robot,
)
from ..locomotion.rough_env_cfg import _apply_play_viewer
from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _apply_rough_terrain,
    _disable_ball_obs_jitter,
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
    _SIGMA_DIRECTION,
    K1WalkKickEnvCfg,
)
from ..walk_kick_dual.walk_kick_dual_env_cfg import enable_obs_history
from ..walk_middle_kick.walk_middle_kick_env_cfg import _apply_middle_kick_recipe
from ..walk_weak_kick.walk_weak_kick_env_cfg import _STRONG_W
from ..walk_weak_kick_orbit.orbit_mods import (
    _KICK_STATE_REWARD_TERMS,
    apply_ball_param_dr,
    apply_orbit_params,
)

# steps_per_iteration = num_steps_per_env (PPO config)。基底の _spi / middle の _SPI と
# 同じ値だが、あちらは __post_init__ のローカル変数なので参照できずリテラルで持つ。
_SPI = 24

# --------------------------------------------------------------------------- #
# インサイド用に上書きする kick_state のパラメータ
#
# apply_orbit_params と同じ配布先 (base_velocity コマンド属性 / kick_finished
# termination / kick_state を呼ぶ全ての報酬項) へ **同じ値** を配ること。
# kick_state はその step で最初に呼んだ項のパラメータで確定する (first-caller-wins)
# ので、1 つでも取りこぼすと結果が報酬項の評価順に依存する。
#
# lateral_band = (-0.15, 0.0):
#   終端の構えに持たせる横方向のあそび (帯) [m]、正 = ロボットから見て右。
#   orbit の既定は (-0.096, 0.0) = 股関節の横オフセットぶんで、「右足をキック線の
#   真上に乗せる」ところまでを許す帯。インサイドはそこからさらに軸足をキック線へ
#   寄せて立つ (蹴り足を体の内側から外へ振るぶん、base をもっと左へ置きたい) ので、
#   下限を広げる。
#   -0.15 の根拠は衝突限界。軸足がボールに当たるのはボール半径 0.11 + 足箱の半幅
#   0.035 = 0.145 より内側なので、0.15 はそのすぐ外側。これより広げると「軸足で
#   ボールを踏む」構えまで指令が許すことになる。
#
# r_stance = 0.20:
#   P_kick (理想キック立ち位置) をボール後方どれだけの点に置くか [m]。
#   既定 0.25 はつま先で突く前提の距離で、インサイドは足の側面を当てるぶん
#   ボールに近づいて構える。0.20 は run 2026-08-21_05-00-22 でインサイドの型
#   (dot≈0 / ball_side 0.145 / vel_ratio 0.91) を作った実績値。
#
#   NOTE: 一度 0.10 まで詰めたことがあるが、あれは **軸足を前に出す目的** の変更で、
#         6700 iteration 回して plant_lon は -0.23 から動かなかった
#         (run 2026-08-21_10-41-17)。P_kick は報酬にほとんど流れないので、
#         軸足を動かすレバーにはならない。軸足は
#         :func:`~..walk_kick.mdp.rewards.kick_plant_lon` の担当に分離したため、
#         r_stance は「望む構えを妨害しない胴体の終着指令」として実績値へ戻す。
#         0.10 はボール半径 0.11 より内側 = 指令の立ち位置がボールの後縁より内側で、
#         ゼロから学習する場合は接近中に体がボールへ触れる事故を増やすだけになる。
#
# overshoot_margin = 0.30:
#   overshoot 罰 (キック線 R を跨いで反対側へ入ったら 1 エピソードに 1 回だけ罰) の
#   遊び [m]。orbit の既定 0.25 は lateral_band が (-0.096, 0.0) だった頃の値で、
#   「指令が許す立ち位置 (帯の端 0.096) を越えてから、さらに 0.154 までは跨ぎを許す」
#   という配分だった。このタスクは帯の下限を -0.15 へ広げたのに 0.25 を据え置いた
#   ので、跨ぎの余裕が 0.25 − 0.15 = 0.10 しか残っておらず、**指令どおり左へ寄って
#   構えるだけで罰の 2/3 まで使い切る** 状態になっていた。実際、初回 run
#   (2026-08-21_03-49-24) の overshoot 発火率は範囲拡大と右足率の上昇に沿って
#   3.3% → 7.0% へ増えている。
#   0.30 = 0.15 (新しい帯の端) + 0.154 (元の余裕) で、トーキック時代と同じ余裕に戻す。
#   罰の目的である「回り直し」(反対側までぐるっと行き直す動き) は 0.30 でも十分捕まる。
#
# style_halfwidth = 0.698:
#   p_style を帯で採点する半幅 [rad] = 40°。インサイドキックは胴体が蹴り方向から
#   30-45° ずれた向きで構えるのが自然な形なので、その中央を取る。
#   帯の中では胴体の向きが報酬に効かなくなり、構えを決めるのは当たり所の項だけになる。
#   詳細はモジュール docstring の「逆風を外す」節と
#   :func:`~..walk_kick.mdp.kick_state.kick_state` の同名引数。
# --------------------------------------------------------------------------- #
_INSIDE_PARAMS = {
    "lateral_band": (-0.15, 0.0),
    "r_stance": 0.20,
    "overshoot_margin": 0.30,
    "style_halfwidth": 0.698,
}

# --------------------------------------------------------------------------- #
# kick_inside_contact の重み
#
# direction (6.0) / scaled (4.0) / strong (3.0) と同じ土俵で、direction と同格の 6.0。
# middle の kick_plant_foot が 2.0 に抑えられていたのは、あれが「目的そのものでは
# なく蹴り方の指定」だったため。こちらは **当たり所がこのタスクの目的そのもの** なので
# 最上位に置く。
#
# 初版は scaled と同格の 4.0。weight 4.0 の run (2026-08-22_11-22-36) で iter 500 の
# foot_kick_dot が 0.91 = strong の全盛期にトーキックの型が固まりつつあった。
# 設計上は strong の退場 (500 → 1200) 後に f_perp が引き戻す想定だが、型の決定期に
# インサイド側の勾配が細いままなのはリスクなので、キック報酬の最上位 (direction) と
# 同格へ上げる。direction を超えないのは、これも r_direction への乗算項であり、
# 「形の項が目的の項を出し抜かない」という序列 (_PLANT_LON_WEIGHT のコメント参照)
# に従うため。
#
# 項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの割り戻し
# (_KICK_W_SCALE) を掛けること。フェードインの窓は基底のキック報酬と同じ 0 → 500 で、
# 発見期には既に満額で乗っている状態にする (型が決まってから足すと、既に決まった
# トーキックの位置で f_perp が小さく、勾配が細いところから始めることになる)。
# --------------------------------------------------------------------------- #
_INSIDE_CONTACT_WEIGHT = 6.0

# --------------------------------------------------------------------------- #
# kick_plant_lon (軸足の前後位置) のパラメータと重み
#
# _PLANT_LON_TARGET = 0.0:
#   軸足の目標前後位置 [m] (ボール基準、+ が前)。body_pos_w が返すのは足リンク原点
#   (= 足首) で、足箱の中心はそこから前方 +0.026 にある。つまり **足首をボールの真横
#   (0.0) に置くと、足箱の中心は 2.6 cm だけボールより前** に来る。
#
#   初版は −0.03 (足箱の中心をボール真横に合わせる導出。middle の kick_plant_foot の
#   lon_target と同じ値) だったが、実機フィードバック 2 回目でも「手前すぎ」の症状が
#   続いたので、目標そのものを −0.03 より前へ動かす。足箱の中心が少し前に出る位置は
#   「軸足がボールと並ぶかわずかに追い越す」構えで、巻き込みの向きから最も遠い。
#
#   **重要: 目標を動かしても、それだけでは政策はほとんど動かない。** 線形テントの
#   勾配は W/span であって目標位置に依らないので、目標を 0.03 前へずらしても
#   「現在位置での傾き」は 1 ミリも変わらない (変わるのは f の切片だけ)。
#   踏み込ませる実際のレバーは **span を絞ること** で、下の第 3 段
#   (:data:`_PLANT_LON_SPAN_END2`) とセットで初めて意味を持つ。目標だけを動かした
#   変更は「効かなかった」ではなく「そもそも効く仕掛けではない」ので、単独では入れない。
#
# _PLANT_LON_SPAN = 0.45:
#   線形テントの半幅 [m]。目標からこの距離離れると 0 点。
#   Gaussian にしないこと・この広さである理由は
#   :func:`~..walk_kick.mdp.rewards.kick_plant_lon` の docstring に書いてある。
#   要点だけ再掲すると、半幅 0.45 なら
#     現在の収束値 −0.23 で f = 0.56、発見期の最悪値 −0.42 (middle 実測) でも f = 0.13
#   と、ポリシーが実際に居る場所で勾配が生きている。σ = 0.10 の Gaussian だった
#   kick_plant_foot は同じ −0.42 で f ≈ 5e-4 = 真っ平らだったので動かなかった。
#
# _PLANT_LON_WEIGHT = 6.0:
#   direction (6.0) と同格 = キック報酬の最上位タイ。方針は「壊れない範囲で最強」。
#
#   履歴: 2.0 (初版。middle の kick_plant_foot に倣い「型の指定だから一段下」) →
#   4.0 (実機の「手前すぎてカス当たり / 空振りした足にボールを巻き込んで転ぶ」
#   事故対策で scaled と同格へ) → 6.0 (weight 4.0 の run 2026-08-22_11-22-36 の
#   決着を待たず、転倒事故の再発コストを優先して壊れない上限まで上げる判断)。
#   plant_lon −0.23 のままでは振り足がボールの手前側を通る幾何そのものが残る。
#
#   「壊れない」の上限を direction 同格 (6.0) に引く理由:
#   * この項は r_direction への乗算・非負なので、方向ゲート (kick_done /
#     τ_direction / p_style) を通らない蹴りには 1 円も払われない。農作の抜け道は
#     構造側で塞がっている。残る壊れ方は「威力・精度を多少削ってでも軸足を置く」
#     というトレードだけで、威力 (v_ball) は r_direction に入っていないぶん
#     scaled が唯一の対抗馬になる。
#   * direction 同格までなら、どの objective 項 (direction / scaled) も plant に
#     1:1 以上で対抗でき、「軸足のためにキックそのものを崩す」が総額で勝てない。
#     これを超えて積むと形 (軸足) が目的 (方向・威力) を出し抜けるようになるので、
#     6.0 が上限。さらに強くしたくなったら weight ではなく span を絞る (下の
#     _PLANT_LON_SPAN_END)。
#
#   効いているかの判定は Metrics/kick_direction/plant_lon (−0.23 から −0.10 側へ
#   動くか)、副作用の監視は vel_ratio (威力を削っていないか) と
#   kick_inside_contact (当たり所と取り合いになっていないか)。
#
#   項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの
#   割り戻し (_KICK_W_SCALE) を掛けること。
#
# _PLANT_LON_SPAN_END = 0.25 / 窓 1500 → 3000 (kick_plant_lon_span の第 2 段):
#   第 2 のエスカレーション。テントの勾配は W/span なので、weight を上限で止めた
#   あとの増強は span を絞る側で行う (6.0/0.45 = 13.3 → 6.0/0.25 = 24。
#   初版 2.0/0.45 = 4.4 の 5.4 倍)。
#   始点 1500 = strong の退場 (1200) + 型の整定余裕 300。トーキック → インサイドの
#   移行が済む前に絞ると、移行中の居場所が 0 に潰れて kick_plant_foot の死因
#   (ポリシーの居る場所で勾配ゼロ) を作り直すことになる。終点 3000 は σ_velocity
#   アニール・拡大ゲートの公称終点と同じ。
#   終値 0.25 は最悪ケースの生存で決めた: 前 checkpoint の収束値 −0.23 (gap 0.20)
#   から一歩も動かなかったとしても f = 1 − 0.20/0.25 = 0.2 が残る。0.20 まで
#   絞るとちょうどその位置で f = 0 になり、「動かない場所で報酬が真っ平ら」を
#   自分で作ってしまう。
#
# _PLANT_LON_SPAN_END2 = 0.15 / 窓 3000 → 4000 (第 3 段。実機フィードバック 2 回目):
#   第 2 段の想定 (「動かなくても f = 0.2 が残る」) は保守的すぎた。実際には
#   run 2026-08-22_11-56-42 で plant_lon は −0.23 → **−0.107** まで動いており、
#   目標 0.0 との gap は 0.11 しかない。span 0.25 ではその位置の f が
#   1 − 0.11/0.25 = 0.56 もあり、「もう十分もらえている」状態になっている。
#   0.15 まで絞ると同じ位置で f = 1 − 0.11/0.15 = 0.27 で、**まだ潰れていないのに
#   伸びしろが大きい** ところへ移る。勾配は 6.0/0.25 = 24 → 6.0/0.15 = 40。
#   0.11 より狭くしないのは、そこで現在位置が f = 0 に潰れるため (kick_plant_foot の
#   死因そのもの)。0.15 は現在位置 −0.107 の外側に 0.04 の余裕を残す最小限の値。
#
#   窓 3000 → 4000 は **他の全カリキュラムが終わったあとの仕上げ窓**。
#   σ_velocity アニール・拡大ゲート・overshoot 罰・span 第 2 段はすべて 3000 で
#   終点に着くので、この窓の中で動いているランプはこれだけになる。plant_lon が
#   動いたか / 何かを壊したかを他の変化と混ぜずに読める。
#
#   NOTE: 3 段は **1 本の折れ線** (piecewise_reward_param) で書くこと。
#         linear_reward_param を 2 本並べると、1 本目が 3000 以降ずっと 0.25 を、
#         2 本目が 3000 まで 0.25 を書き続ける「同じ param に書き手が 2 人」状態に
#         なり、最終値が CurriculumManager の実行順で決まってしまう
#         (:func:`~..walk_kick.mdp.curriculums.piecewise_reward_param` の docstring)。
# --------------------------------------------------------------------------- #
_PLANT_LON_TARGET = 0.0
_PLANT_LON_SPAN = 0.45
_PLANT_LON_SPAN_END = 0.25
_PLANT_LON_SPAN_START_ITER = 1500
_PLANT_LON_SPAN_END_ITER = 3000
_PLANT_LON_SPAN_END2 = 0.15
_PLANT_LON_SPAN2_START_ITER = 3000
_PLANT_LON_SPAN2_END_ITER = 4000
_PLANT_LON_WEIGHT = 6.0

# lon_span の折れ線。書き手を 1 つに保つため 3 段を 1 本にまとめる (上の NOTE)。
# 先頭の (0, _PLANT_LON_SPAN) は「1500 まで 0.45 で据え置き」を明示するための平坦部で、
# これが無いと 0 → 1500 の間に勝手に補間が始まってしまう。
_PLANT_LON_SPAN_KNOTS = [
    (0, _PLANT_LON_SPAN),
    (_PLANT_LON_SPAN_START_ITER, _PLANT_LON_SPAN),
    (_PLANT_LON_SPAN_END_ITER, _PLANT_LON_SPAN_END),
    (_PLANT_LON_SPAN2_END_ITER, _PLANT_LON_SPAN_END2),
]

# 第 2 段の終点 (3000) と第 3 段の始点 (3000) は折れ線では **同じ 1 つの knot** に
# 潰れる。したがって _PLANT_LON_SPAN2_START_ITER は上の knots に現れず、ずれても
# 折れ線は黙って「3000 → 4000 で 0.25 → 0.15」のまま通ってしまう (定数だけが嘘に
# なる)。第 3 段を後ろへずらしたくなったときに 2 つの定数が食い違ったまま放置される
# のを防ぐため、ここで一致を強制する。
assert _PLANT_LON_SPAN_END_ITER == _PLANT_LON_SPAN2_START_ITER, (
    "lon_span の第 2 段の終点と第 3 段の始点は同じ iteration である必要があります "
    f"({_PLANT_LON_SPAN_END_ITER} != {_PLANT_LON_SPAN2_START_ITER})"
)

# --------------------------------------------------------------------------- #
# kick_plant_yaw (軸足の向き) のパラメータと重み — 実機フィードバック 2 回目
#
# _PLANT_YAW_SPAN = math.pi / 2 (90°):
#   線形テントの半幅 [rad]。軸足のつま先が蹴り方向から 90° (真横) ずれると 0 点で、
#   そこから蹴り方向へ向くほど **角度に対して線形に** 増える。
#   cos ではなく角度に対して線形にする理由 (0° 付近で cos が真っ平らになるため) は
#   :func:`~..walk_kick.mdp.rewards.kick_plant_yaw` の docstring。
#   90° より外 = かかとを蹴り方向へ向ける側は誘導したい構えの反対なので 0 で飽和させる。
#
#   NOTE: 絞るのは **現行ポリシーの plant_yaw_dot の実測を見てから**。この項は
#         初導入なので、いまポリシーが何度を向いているかの記録がまだ無い
#         (だからこそ同時に Metrics/kick_direction/plant_yaw_dot を出す)。
#         span を実測より狭くすると、その位置で f = 0 = 勾配ゼロになり
#         kick_plant_foot の死因を作り直す。
#
# _PLANT_YAW_WEIGHT = 3.0:
#   **形の項は objective (direction 6.0) の半分** から入れる、という序列に従う。
#   キック報酬の上限を direction 同格の 6.0 に置く理由は _PLANT_LON_WEIGHT の
#   コメント (形が目的を出し抜けるようになる境目) にあり、この項もその上限の下で
#   運用する。3.0 はその半分 = 「効くかどうかがまず分かる大きさで、かつ direction /
#   scaled のどちらか一方に対しても単独では勝てない」量。
#   kick_plant_lon が 2.0 → 4.0 → 6.0 と上げていったのと同じ道筋を辿る想定で、
#   効果が出て副作用 (vel_ratio の低下) が無ければ上げる。
#
#   項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの
#   割り戻し (_KICK_W_SCALE) を掛けること。
# --------------------------------------------------------------------------- #
_PLANT_YAW_WEIGHT = 3.0
_PLANT_YAW_SPAN = math.pi / 2

# --------------------------------------------------------------------------- #
# kick_foot_ceiling (足を上げすぎない上限) のパラメータと重み — 実機フィードバック 2 回目
#
# _FOOT_CEILING_H_CAP = 0.09:
#   これ以下なら満点になる足裏高さ [m]。足コライダーは足リンク原点から
#   z = −0.038 (足裏) 〜 −0.002 (上面) の厚み 0.036 の箱なので、足裏高さ h のとき
#   足箱の中心は h + 0.018。**足箱の中心がボール中心 0.11 以下に来る** のは
#   h ≤ 0.092 で、0.09 はそのすぐ内側。これより上で当てると足の重心がボール中心より
#   高いところを通るので、上から押さえる / 空振りして足がボールの上を越える形になる。
#
# _FOOT_CEILING_H_SPAN = 0.08:
#   天井を越えてから 0 点になるまでの幅 [m] (h = 0.17 で 0)。ボール直径 0.22 の
#   おおよそ 3/4 で、「ボールの上半分にしか当たっていない」領域を 0 に落とす。
#   狭くするほど天井の壁は急になるが、急にすると天井の外に居るあいだ勾配が死ぬ。
#
# _FOOT_CEILING_WEIGHT = 3.0:
#   _PLANT_YAW_WEIGHT と同じ「形の項は objective の半分から」。上限 6.0 の
#   ルールも同じ (_PLANT_LON_WEIGHT のコメント)。
#
#   **kick_contact_height (低いほど得) は入れないこと。** あちらは walk_lob で
#   反証されている (sole_height は下がったが apex が 0.340 → 0.234。低く当てるには
#   立ち位置を詰めるしかなく、それがスイング長 = ボール速度を削る)。この項は
#   h_cap 以下を一律満点にして **下向きの圧を持たない** ので、その失敗を踏まない。
#   両方入れると h_cap 以下で f_low の下向きの圧だけが残り、結局あちらと同じ形になる。
#
#   項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの
#   割り戻し (_KICK_W_SCALE) を掛けること。
# --------------------------------------------------------------------------- #
_FOOT_CEILING_WEIGHT = 3.0
_FOOT_CEILING_H_CAP = 0.09
_FOOT_CEILING_H_SPAN = 0.08

# --------------------------------------------------------------------------- #
# kick_velocity_strong の折れ線 (このタスク用に「下り」を前倒しした版)
#
# weak/middle のレシピが入れる既定は [(0,0), (500,W), (1500,W), (3000,0)]
# (:data:`~..walk_weak_kick.walk_weak_kick_env_cfg._STRONG_KNOTS`)。
# **上り (0 → 500) はそのまま、満額の維持を打ち切って 500 → 1200 で 0 へ落とす。**
#
# strong は「r_dir × v_ball」の青天井項で、速く蹴るほど得。これは 0 から学習し直す
# タスクで **キックという行動そのものを発見させる** ための項であって、当たり所の
# 型を決める項ではない。そして「いちばん強く振れる蹴り方」はつま先で突く形
# (トーキック) なので、満額で置き続けるかぎり報酬は名指しでトーキックを要求する。
#
# 初回 run (2026-08-21_03-49-24) の実測:
#   * kick_rate は 250 iteration で 0.85 を超えている = 発見は 250 で済んでいる。
#   * それでも 0-1500 の間、strong の払いは kick_inside_contact の 43〜69 倍あった。
#     当たり所の型が決まるのはまさにこの時期なので、インサイドの側の勾配は
#     桁違いに小さい圧としてしか働けなかった。
# 発見が済み次第 strong を退場させ、型の決定権を kick_inside_contact と
# kick_velocity_scaled へ渡す。500 は上りが終わる点 (満額に達した直後から落とす)、
# 1200 は拡大ゲートが動き出す 500 から 700 iteration の移行期間を取った点。
#
# σ_velocity のアニール (500 → 3000) と overshoot 罰のフェードイン (1500 → 3000) は
# **触らない**。あちらは「速度を指令帯へ絞り込む」側の仕掛けで、当たり所とは別件。
# 終値が 0 であることも変えない (少しでも残すと「速いほど得」が残る)。
# --------------------------------------------------------------------------- #
_STRONG_FADE_START_ITER = 500
_STRONG_FADE_END_ITER = 1200
_INSIDE_STRONG_KNOTS = [
    (0, 0.0),
    (_STRONG_FADE_START_ITER, _STRONG_W),
    (_STRONG_FADE_END_ITER, 0.0),
]

# --------------------------------------------------------------------------- #
# 全方位への拡大ゲートの窓 [iteration]
#
# start = 500: キック報酬のフェードイン (0 → 500) が終わるまでは範囲を動かさない。
#   報酬の定義がまだ動いている間に難易度も動かすと、kick_rate が落ちた原因が
#   どちらなのか読めなくなる。
# end = 3000: 公称の拡大速度 = 1/(3000−500) / iteration。あくまで「ゲートが開き
#   っぱなしなら何 iteration で全方位に届くか」であって、実際には kick_rate が
#   0.80 を割った時点で止まり、0.50 を割ると 2 倍速で戻る。
#   既定 ITER = 5000 に対して、届いてから 2000 iteration の仕上げが残る配分。
# --------------------------------------------------------------------------- #
_EXPANSION_START_ITER = 500
_EXPANSION_END_ITER = 3000

# 拡大の始点 (= 基底 walk_kick の限定レンジ) と終点 (= 全方位)。
_BALL_HALF_ANGLE_RANGE = (1.047, math.pi)
_BALL_DIST_START = (0.5, 0.8)
_BALL_DIST_END = (0.5, 1.5)
_KICK_HEADING_HALFWIDTH_RANGE = (math.pi / 4, math.pi)

# ball_avoidance の σ_sole。apply_orbit_params が入れるのと同じ値を明示しておく
# (回り込み半径の決定を罰から指令へ移すためのもの。理由は orbit_mods 参照)。
_BALL_AVOIDANCE_SIGMA_SOLE = 0.20
_BALL_AVOIDANCE_SIGMA_POSE = 0.3

# 脚同士の接近ペナルティ (キック中だけ緩める版) のしきい値。
# 通常時は locomotion 版と同じ値、ボールが近い env だけ緩い方へ切り替わる。
_FEET_CLOSE_THRESHOLD = 0.14
_FEET_CLOSE_RELAXED = 0.10
_KNEE_CLOSE_MIN_DIST = 0.13
_KNEE_CLOSE_RELAXED = 0.10
_CLOSE_RELAX_DIST = 0.5
_CLOSE_PENALTY_WEIGHT = -20.0


def _apply_inside_params(cfg: "K1WalkKickEnvCfg") -> None:
    """:data:`_INSIDE_PARAMS` を kick_state を共有する全ての項に配る。

    配布先は :func:`~..walk_weak_kick_orbit.orbit_mods.apply_orbit_params` と同一
    (base_velocity コマンドの属性 / kick_finished termination /
    :data:`~..walk_weak_kick_orbit.orbit_mods._KICK_STATE_REWARD_TERMS` の全項)。
    kick_state は **その step で最初に呼んだ項のパラメータで確定する**
    (first-caller-wins) ので、1 つでも取りこぼすと G・P_kick・p_style が
    報酬項の評価順に依存してしまう。

    **報酬項を追加し終えた後、``apply_orbit_params`` の後に呼ぶこと。**
    あちらは ``lateral_band`` に orbit の既定 (-0.096, 0.0) を入れるので、
    先に呼ぶとこちらの (-0.15, 0.0) が上書きされてしまう。
    """
    base_velocity = getattr(cfg.commands, "base_velocity", None)
    if base_velocity is not None:
        for _key, _val in _INSIDE_PARAMS.items():
            setattr(base_velocity, _key, _val)

    kick_finished = getattr(cfg.terminations, "kick_finished", None)
    if kick_finished is not None:
        kick_finished.params.update(_INSIDE_PARAMS)

    for _name in _KICK_STATE_REWARD_TERMS:
        _term = getattr(cfg.rewards, _name, None)
        if _term is not None:
            _term.params.update(_INSIDE_PARAMS)


def _apply_inside_kick_recipe(cfg: "K1WalkKickEnvCfg") -> None:
    """middle のレシピを土台に、右足インサイドキック用の差分を全部入れる。

    ``__post_init__`` の最後 (基底クラスの設定が全部済んだ後) に呼ぶこと。
    観測の次元・並びには一切触らないので、歩行 checkpoint をそのまま
    ``--load_pretrained`` できる (観測 55 次元・履歴なし)。
    """
    # -- 1. middle のレシピ ------------------------------------------------ #
    #
    # 帯 (3.2, 4.5) 固定・σ_velocity 終点 0.5・weak の 3 点セット (latch 閾値の
    # 指令追従 / strong の折れ線 / σ アニール + overshoot 罰)・ボール物性 DR が入る。
    # 「5-10 m 飛ぶキックを指令どおりに出す」という戦略側の要求はインサイドでも
    # 変わらないので、実績のあるレシピをそのまま土台にする。
    _apply_middle_kick_recipe(cfg)

    # -- 1b. kick_velocity_strong のフェードアウトを前倒し ------------------ #
    #
    # レシピ関数は触らず、レシピが入れた curriculum 項 (piecewise_reward_weight) の
    # knots だけをこのタスク用の折れ線へ差し替える (:data:`_INSIDE_STRONG_KNOTS`)。
    # 上り (0 → 500) は同じで、下りが 1500 → 3000 から 500 → 1200 に早まる。
    # 理由は :data:`_INSIDE_STRONG_KNOTS` のコメント (発見は 250 iteration で済んで
    # いるのに、当たり所の型が決まる 0-1500 の間 strong が kick_inside_contact の
    # 43〜69 倍の額でトーキックを要求し続けていた)。
    cfg.curriculum.kick_velocity_strong_weight.params["knots"] = _INSIDE_STRONG_KNOTS

    # -- 2. kick_plant_foot を外す ----------------------------------------- #
    #
    # walk_lob 系 5 run の実測で plant_lon は −0.36〜−0.43 に居座り、一度も目標
    # (−0.03) に漸近しなかった。σ_lon を広げて目標を引っ張った版でも gap が開いた
    # だけで実測は動いていない。軸足の位置は原因ではなく **蹴り方の結果** なので、
    # 報酬で直接引っ張っても動かない。加えて目標値 (lon −0.03 / lat 0.19) は
    # トーキック前提で決めたもので、インサイドの構えに当てはまる保証も無い。
    # 報酬からは外し、``Metrics/kick_direction/plant_lon`` を **指標としてだけ** 見る。
    cfg.rewards.kick_plant_foot = None
    cfg.curriculum.kick_plant_foot_weight = None
    cfg.curriculum.kick_plant_foot_sigma_lon = None

    # -- 3. エピソード長と、拡大の始点 -------------------------------------- #
    #
    # 15 秒。ゲートが開き切ると 1.5 m + 半周回り込みで移動 2.5-3 m になるので、
    # 全方位版 (K1WalkKick360EnvCfg) と同じ長さを最初から取る。段を分けない以上、
    # 途中でエピソード長を変えると同じ run の中で「時間切れの起きやすさ」が
    # 変わってしまう。
    cfg.episode_length_s = 15.0

    # 始点は基底 walk_kick の限定レンジそのもの (明示しておく。ここからゲートが動かす)。
    cfg.events.reset_ball.params["half_angle"] = _BALL_HALF_ANGLE_RANGE[0]
    cfg.events.reset_ball.params["dist_range"] = _BALL_DIST_START
    cfg.commands.kick_direction.ranges.heading = (
        -_KICK_HEADING_HALFWIDTH_RANGE[0],
        _KICK_HEADING_HALFWIDTH_RANGE[0],
    )

    # 当たり所の幾何を TensorBoard に出す (Metrics/kick_direction/foot_kick_dot と
    # .../ball_side)。kick_inside_contact の f_perp / f_side の中身そのものなので、
    # 「報酬が伸びない」ときに向きの問題なのか内外の問題なのかを切り分けられる。
    # 既定 False = 他タスクの TB タグ集合を変えないため、ここだけ True にする
    # (:class:`~..walk_kick.mdp.commands.KickDirectionCommandCfg` の同名フラグ)。
    cfg.commands.kick_direction.log_contact_geometry = True

    # -- 4. ball_avoidance を weight 0 で置く ------------------------------- #
    #
    # 全方位になってから効かせる「構えができるまでボールに寄るな」の抑止。
    # 重みはゲート (下の kick_expansion) が α に比例して立ち上げる。
    # 360 版の cfg (K1WalkKick360EnvCfg) を継承しないのは、あちらが
    # approach_penalty を **即座に消す** 作りで、限定レンジから始めるこのタスクとは
    # 噛み合わないため。基底 K1WalkKickEnvCfg から直接組む。
    cfg.rewards.ball_avoidance = RewTerm(
        func=mdp.ball_avoidance,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_sole": _BALL_AVOIDANCE_SIGMA_SOLE,
            "sigma_pose": _BALL_AVOIDANCE_SIGMA_POSE,
        },
    )

    # -- 5. 拡大ゲート ------------------------------------------------------ #
    #
    # ボール出現範囲・蹴り方向範囲・approach_penalty / ball_avoidance の重みを
    # 1 本の α で同時に動かす。α はキック成立率 (EMA) が 0.80 以上なら進み、
    # 0.50 未満なら 2 倍速で戻り、その間は据え置き。
    #
    # 基底が入れている approach_penalty の壁時計フェードイン (0 → 500 iteration で
    # 0 → −3.0) は **この関数の中で再現している** ので、基底のカリキュラム項は
    # None にして書き手を 1 つに絞る。同じ weight を 2 つの curriculum 項が
    # 書き合うと、どちらが最後に走るかで値が決まってしまう。
    cfg.curriculum.approach_penalty_weight = None
    cfg.curriculum.kick_expansion = CurrTerm(
        func=mdp.kick_rate_gated_expansion,
        params={
            "command_name": "kick_direction",
            "start_step": _EXPANSION_START_ITER,
            "end_step": _EXPANSION_END_ITER,
            "steps_per_iteration": _SPI,
            "ball_event_name": "reset_ball",
            "half_angle_range": _BALL_HALF_ANGLE_RANGE,
            "dist_range_start": _BALL_DIST_START,
            "dist_range_end": _BALL_DIST_END,
            "heading_halfwidth_range": _KICK_HEADING_HALFWIDTH_RANGE,
            "approach_term_name": "approach_penalty",
            "approach_end_weight": -3.0,
            "approach_fade_iterations": 500,
            "avoidance_term_name": "ball_avoidance",
            "avoidance_end_weight": -3.0,
        },
    )

    # -- 6. インサイドの当たり所を採点する項 -------------------------------- #
    #
    # このタスクの目的そのもの。r_direction への乗算なので kick_done ゲート・
    # 方向精度・胴体の向き (帯つき p_style) を全て通過した蹴りにしか払われず、
    # 他のキック報酬とは **加算** で並ぶ (乗算にしない。学習初期はインサイドがまず
    # 出ないので、掛けると他項の勾配がゼロ付近で死ぬ)。
    # sigma_direction は他のキック項と必ず同じ値 (_SIGMA_DIRECTION = 0.35)。
    cfg.rewards.kick_inside_contact = RewTerm(
        func=mdp.kick_inside_contact,
        weight=0.0,
        params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION},
    )
    cfg.curriculum.kick_inside_contact_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_inside_contact",
            "start_weight": 0.0,
            "end_weight": _INSIDE_CONTACT_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": 500,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 6b. 軸足の前後位置を誘導する項 ------------------------------------- #
    #
    # 「振りが手前すぎてボールを巻き込んで転ぶ」実機事故 (run 2026-08-21_05-00-22) の
    # 対策。軸足 (蹴っていない方の足) をボールの真横へ寄せる。軸足がボール横にあれば、
    # ボール認識が 3cm ずれて手前に当たってもボールは体の横を抜けていき、踏む位置に
    # 残らない。
    #
    # **上の kick_plant_foot 除去 (セクション 2) を取り消すものではない。** あちらは
    # f_lon が Gaussian (σ_lon = 0.10) で、実測 −0.42 では f ≈ 5e-4 = 勾配ゼロ。
    # こちらは線形テント (半幅 0.45) なので −0.42 でも f = 0.13、−0.23 で f = 0.56 と
    # ポリシーが実際に居る場所で傾きが残る。lat も掛けない (plant_lat 0.30 は別課題で、
    # 掛けると「横が外れているあいだ lon の勾配も死ぬ」を再現してしまう)。
    #
    # r_stance を詰める案 (0.20 → 0.10) は先に試して失敗している
    # (run 2026-08-21_10-41-17: plant_lon −0.23 のまま不動)。r_stance は **胴体の
    # 終着指令**であって軸足を測っていないので、収束済みポリシーを動かすレバーには
    # ならない。r_stance = 0.10 はそのままにして (望む構えを妨害しない役割は果たして
    # いる)、軸足の誘導はこの項が持つ。詳細は
    # :func:`~..walk_kick.mdp.rewards.kick_plant_lon` の docstring。
    #
    # フェードインの窓は他のキック報酬と同じ 0 → 500 (発見期に満額で乗せる)。
    #
    # NOTE: 運用。**--resume での後掛け fine-tune では common_step_counter が既に
    #       進んでいるため、このフェードインは初回 iteration で即座に終値になる。**
    #       それで構わない (線形テントなので、いきなり満額でも現在位置 −0.23 に
    #       ちゃんと勾配がある)。ただし収束済みポリシーへの後掛けが効かなかった
    #       前例が既に 2 つある (lob の kick_plant_foot、r_stance 0.10 の resume) ので、
    #       **本命はゼロから (歩行 checkpoint 起点で) 回すこと**。
    cfg.rewards.kick_plant_lon = RewTerm(
        func=mdp.kick_plant_lon,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _SIGMA_DIRECTION,
            "lon_target": _PLANT_LON_TARGET,
            "lon_span": _PLANT_LON_SPAN,
        },
    )
    cfg.curriculum.kick_plant_lon_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_plant_lon",
            "start_weight": 0.0,
            "end_weight": _PLANT_LON_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": 500,
            "steps_per_iteration": _SPI,
        },
    )

    # 第 2・第 3 のエスカレーション: 後期に span を絞り、目標付近の傾きを立てる
    # (勾配 = W/span)。0.45 → 0.25 (1500 → 3000) → 0.15 (3000 → 4000) の
    # **3 段を 1 本の折れ線で** 書く。各段の根拠は :data:`_PLANT_LON_SPAN_END` と
    # :data:`_PLANT_LON_SPAN_END2` のコメント。strong 退場 (1200) より前に絞ると
    # トーキック期の居場所が 0 に潰れるので、絞り始めは必ずその後に置くこと。
    #
    # linear_reward_param を 2 本並べないこと。同じ param に書き手が 2 人になり、
    # 3000 以降の値が CurriculumManager の実行順で決まってしまう
    # (:func:`~..walk_kick.mdp.curriculums.piecewise_reward_param` の docstring)。
    cfg.curriculum.kick_plant_lon_span = CurrTerm(
        func=mdp.piecewise_reward_param,
        params={
            "term_name": "kick_plant_lon",
            "param_name": "lon_span",
            "knots": _PLANT_LON_SPAN_KNOTS,
            "steps_per_iteration": _SPI,
        },
    )

    # 発見期の呼び水。**weight 0 で置くだけ。カリキュラムも付けない。**
    #
    # kick_inside_contact は「当たった瞬間」しか見ないので、インサイドで当たる接触が
    # 一度も起きなければ勾配が 1 度も立たない。この項は接触していなくても
    # 「ボールの近くで足が横を向いている」ことに払うので、その入口を作れる。
    # ただし **蹴らずにボールの脇で足を横に向けたまま滞在すると貯まる** (農作) ので、
    # 既定では寝かせておく。kick_inside_contact が数千 iteration 立ち上がらず、
    # kick_rate は健全なのにインサイドの当たりが出ない、と確認してから人が有効化する
    # (詳細は :func:`~..walk_kick.mdp.rewards.inside_foot_orient` の warning)。
    cfg.rewards.inside_foot_orient = RewTerm(
        func=mdp.inside_foot_orient,
        weight=0.0,
        params={**_KICK_STATE_PARAMS},
    )

    # -- 6c. 軸足の向きを誘導する項 ---------------------------------------- #
    #
    # 実機フィードバック 2 回目 (2026-08-23) の 1 つ目。軸足 (蹴っていない方の足) の
    # **つま先を蹴り方向へ向かせる**。軸足がキック線に正対して立つと骨盤もそちら側へ
    # 寄るので、振り足のインサイド面がキック線に正対しやすくなる。軸足が斜めのままだと
    # 振り足はその骨盤の向きに引きずられて斜めに入り、**当たりが薄い / 空振り**になる。
    # 実機で残っている失敗がこれ。
    #
    # 6b (kick_plant_lon) との関係: あちらは軸足を **どこに置くか** (前後位置)、
    # こちらは **どちらへ向けるか** (ヨー)。位置が合っていても向きは独立に外れるので
    # 項を分ける。lon に掛け算しないのは kick_plant_foot の失敗 (片方が外れている
    # あいだ両方の勾配が死ぬ) を作り直さないため — 常に加算で並べる。
    #
    # 胴体の帯 (style_halfwidth = 40°) と矛盾しない。**胴体は 30-45° ずれてよいが
    # 軸足は蹴り方向を向かせる**、という役割分担で、両者を分ける自由度が Hip_Yaw。
    # このタスクは joint_deviation_hip の対象から Hip_Yaw を外してある (第 7 節) ので、
    # 軸足のヨーはポリシーが自由に使える。
    #
    # フェードインの窓は他のキック報酬と同じ 0 → 500。
    cfg.rewards.kick_plant_yaw = RewTerm(
        func=mdp.kick_plant_yaw,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _SIGMA_DIRECTION,
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
            "end_step": 500,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 6d. 足を上げすぎない上限 ------------------------------------------- #
    #
    # 実機フィードバック 2 回目 (2026-08-23) の 3 つ目。蹴り足の足裏高さに
    # **天井だけ** を課す (h ≤ 0.09 は一律満点、そこから 0.08 かけて 0 へ)。
    # 高い位置で当てると足の重心がボール中心より上を通るので、上から押さえる形か、
    # 空振りして足がボールの上を越える形になる。
    #
    # **kick_contact_height (低いほど得) とは別物で、あちらは入れない。** walk_lob
    # 2026-08-18 で反証済み: 足裏を下へ押す圧はスイングを短くして apex とボール速度を
    # 削った (0.340 → 0.234)。この項は h_cap 以下で完全に平ら = 下向きの圧が
    # ゼロなので、その失敗を踏まない。採りたいのは「上げすぎない」だけ。
    #
    # フェードインの窓は他のキック報酬と同じ 0 → 500。
    cfg.rewards.kick_foot_ceiling = RewTerm(
        func=mdp.kick_foot_ceiling,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _SIGMA_DIRECTION,
            "h_cap": _FOOT_CEILING_H_CAP,
            "h_span": _FOOT_CEILING_H_SPAN,
        },
    )
    cfg.curriculum.kick_foot_ceiling_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_foot_ceiling",
            "start_weight": 0.0,
            "end_weight": _FOOT_CEILING_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": 500,
            "steps_per_iteration": _SPI,
        },
    )

    # -- 7. joint_deviation_hip から Hip_Yaw を外す ------------------------- #
    #
    # 元は (-0.05, [".*_Hip_Yaw", ".*_Hip_Roll"])。Hip_Yaw を初期値 0 へ引き戻す圧が
    # 常時掛かっていた。インサイドは **Hip_Yaw を ±1 rad まで使って足先を外へ向ける**
    # 動きが主役なので、この項がその構えに常時ブレーキを掛けることになる。
    # Hip_Roll (脚の開き) の側は歩容の安定に効くので残す。
    #
    # SceneEntityCfg は resolve() で joint_ids を書き込む可変オブジェクトなので、
    # 既存インスタンスの joint_names を書き換えるのではなく新しく作って差し替える。
    cfg.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*_Hip_Roll"]
    )

    # -- 8. 脚同士の接近ペナルティを「キック中だけ緩める」版へ --------------- #
    #
    # インサイドは軸足をキック線のすぐ脇まで寄せて蹴り足を体の内側から外へ振るので、
    # 接触の前後で左右の脚が普段の歩行より確実に近づく。しきい値を歩行のまま置くと
    # インサイドの構えそのものが罰される。ボールが 0.5 m 以内の env だけしきい値を
    # 緩め、ボールが飛べば距離が開いて自動で元へ戻る。
    #
    # **緩めるだけで無効化はしない。** ``enabled_self_collisions=False`` なので
    # sim では脚がすり抜け、この項が無いと交差する歩容を獲得して実機で脚がぶつかる。
    # 膝コライダーの直径は 0.09、足箱の幅は 0.07 なので、0.10 はまだ物理接触の手前。
    #
    # 項名を locomotion 版から変えてあるのは、
    # :data:`~..walk_weak_kick_orbit.orbit_mods._KICK_STATE_REWARD_TERMS` へ
    # 入れるため。あちらのリストは「kick_state を呼ぶ項」の名簿で、locomotion 版と
    # 同名にすると他タスクの locomotion 版にも orbit のパラメータが配られてしまう。
    cfg.rewards.feet_close_penalty = None
    cfg.rewards.knee_close_penalty = None
    cfg.rewards.feet_close_penalty_kick_aware = RewTerm(
        func=mdp.feet_close_penalty_kick_aware,
        weight=_CLOSE_PENALTY_WEIGHT,
        params={
            **_KICK_STATE_PARAMS,
            "feet_distance_threshold": _FEET_CLOSE_THRESHOLD,
            "relaxed_threshold": _FEET_CLOSE_RELAXED,
            "relax_dist": _CLOSE_RELAX_DIST,
        },
    )
    cfg.rewards.knee_close_penalty_kick_aware = RewTerm(
        func=mdp.knee_close_penalty_kick_aware,
        weight=_CLOSE_PENALTY_WEIGHT,
        params={
            **_KICK_STATE_PARAMS,
            "min_distance": _KNEE_CLOSE_MIN_DIST,
            "relaxed_min_distance": _KNEE_CLOSE_RELAXED,
            "relax_dist": _CLOSE_RELAX_DIST,
        },
    )

    # -- 9. 回り込み型 G / 跨ぎの遊び / ボール物性 DR ----------------------- #
    #
    # **報酬項を全部足し終えた後に呼ぶこと** (apply_orbit_params の docstring)。
    # 上で追加した 4 項にも r_max / orbit_beta / overshoot_margin / lateral_band が
    # 配られる必要がある。
    apply_orbit_params(cfg)
    apply_ball_param_dr(cfg)

    # -- 10. インサイド用のパラメータで上書き ------------------------------- #
    #
    # apply_orbit_params が入れた lateral_band (-0.096, 0.0) をこのタスクの
    # (-0.15, 0.0) で上書きし、r_stance と style_halfwidth も同じ配布先へ配る。
    # 必ず apply_orbit_params の **後**。
    _apply_inside_params(cfg)


@configclass
class K1WalkInsideKickCleanEnvCfg(K1WalkKickEnvCfg):
    """フォールバック: ボール観測ノイズ+遅延を入れない版。**通常は使わない。**

    :class:`K1WalkInsideKickEnvCfg` との差は
    :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` を呼ばないことだけ。
    観測の次元も並びも同じなので、どちらの checkpoint も相互に載る。

    本命は最初からノイズ込みの :class:`K1WalkInsideKickEnvCfg`。ノイズは「進む軸」では
    ないので段を分ける理由が無い (モジュール docstring の「なぜ段を分けないのか」)。
    それでも **インサイドの発見期がどうしても立ち上がらなかったとき** に、原因が
    観測ノイズなのか報酬設計なのかを切り分けるためだけにこちらを回す。
    切り分けが済んだら本命へ戻ること (ノイズ無しで学習した当て方は実機へ転移しない)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_inside_kick_recipe(self)


@configclass
class K1WalkInsideKickCleanEnvCfg_PLAY(K1WalkInsideKickCleanEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkInsideKickEnvCfg(K1WalkKickEnvCfg):
    """本命: 右足インサイドキックを 1 段で学習する。

    歩行 checkpoint からそのまま始める (観測 55 次元・並びとも walk_kick 系と同一)::

        _labpython2 scripts/rsl_rl/train.py \\
            --task Isaac-Velocity-Flat-K1-Walk-Inside-Kick-v0 \\
            --headless --num_envs 4096 --max_iterations 5000 \\
            --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt

    通しスクリプトは :file:`scripts/rsl_rl/train_walk_inside_kick.sh`。

    学習中に見るもの
    ----------------
    * ``Curriculum/kick_expansion/alpha`` と ``kick_rate_ema``: α が止まったまま
      ema が低いなら、その範囲が今のポリシーの実質的な上限。
    * ``Episode_Reward/kick_inside_contact``: これが 0 に張り付いたままなら
      インサイドの当たりが 1 度も出ていない。``Metrics/kick_direction/kick_rate`` が
      健全 (蹴れてはいる) ならトーキックのまま収束しかけているということなので、
      ``inside_foot_orient`` を小さい weight で有効化することを検討する。
    * ``Metrics/kick_direction/plant_lon``: 軸足の前後位置 [m] (+ = ボールより前)。
      ``kick_plant_lon`` が直接引っ張っている値なので、**この項が効いているかの
      唯一の判定材料**。初版の収束値が −0.23 なので、そこから −0.10 側へ動くかを見る。
      1500 iteration 以降は span が 0.45 → 0.25 (3000) → 0.15 (4000) と 3 段で
      絞られる (``Curriculum/kick_plant_lon/lon_span``)。絞りの途中で plant_lon が
      悪化するようなら、絞りすぎ = 居場所の勾配が細っているサイン。
      **第 3 段 (3000 → 4000) は他の全カリキュラムが終わったあとの仕上げ窓**なので、
      ここで動く指標の変化は span 以外に原因が無い。
      動かないまま ``Episode_Reward/kick_plant_lon`` だけが増えているなら、
      f_lon ではなく r_direction (キックの数と質) の方が伸びているだけ。
      なお ``--resume`` での後掛けでは動かない前例が 2 つあるので、動かなければ
      歩行 checkpoint からゼロで回し直す (モジュール docstring の経緯の節)。
    * ``Metrics/kick_direction/plant_lat``: 軸足の横位置 [m] (絶対値)。報酬からは
      外したままの **観察用**。初版の実測は 0.30 で通常スタンス幅 0.192 よりかなり
      広い。plant_lon を詰めた副作用でここが動くかどうかは見ておくこと。
    * ``Metrics/kick_direction/foot_kick_dot`` / ``ball_side``: 当たり所そのもの。
      前者が 1 付近なら足がキック方向を向いたまま = トーキック、0 付近なら足が真横 =
      側面で当てている。後者は蹴り足のローカル y [m] で、正がインサイド側 (足箱の
      半幅 0.035 付近まで来ていれば面の中央で当たっている)。
      ``kick_inside_contact`` が伸びないときに、向き (f_perp) と内外 (f_side) の
      どちらが足りていないのかはこの 2 つで分かる。
    * ``Metrics/kick_direction/kick_foot_right_frac``: このタスクは右足専用なので、
      1.0 から離れていくようなら ``kick_inside_contact`` の右足ゲートが払われて
      いないことになる (キック報酬全体は左足でも入るため、左足へ逃げる余地はある)。

    NOTE: ``--reset_noise_std`` は使わないこと。歩行 checkpoint からの引き継ぎなので
          スイングは既に精密で、std を戻すとそれを壊す。
    NOTE: ``--resume`` も使わないこと。common_step_counter が同期され、キック報酬の
          フェードインと拡大ゲートの start_step が「もう終わった」と判定される。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_inside_kick_recipe(self)

        # ボール位置観測を実機の認識パイプライン寄りに差し替える (エピソードごとの
        # ランダム遅延 2-6 ステップ + 30Hz サンプル&ホールド + フレーム同期ジッタ)。
        # 段は分けず最初から入れる (モジュール docstring 参照)。
        _apply_noisy_ball_obs(self)


@configclass
class K1WalkInsideKickEnvCfg_PLAY(K1WalkInsideKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # enable_corruption = False は ObsTerm の noise しか切らないので、
        # 関数側に移したジッタは別途切る。遅延とサンプル&ホールドは観測パイプラインの
        # 構造 (= PLAY で見たいもの) なので残す。
        _disable_ball_obs_jitter(self)


# =========================================================================== #
# stage 2 / stage 3 — 収束済み checkpoint からの fine-tune 段
#
# 段の位置づけ・引き継ぎ方・TensorBoard で見るものはモジュール docstring の
# 「stage 2 / stage 3 (dual history / rough + DR)」節にまとめてある。
# =========================================================================== #

# --------------------------------------------------------------------------- #
# stage 3 で広げるボール物性 DR の帯
#
# 基底のレシピ (:func:`_apply_inside_kick_recipe` → :func:`~..walk_weak_kick_orbit.orbit_mods.apply_ball_param_dr`)
# は既に orbit の narrow な帯を入れている:
#     静摩擦 (0.3, 0.7) / 動摩擦 (0.2, 0.5) / 反発 (0.2, 0.7) / 質量 ×(0.9, 1.15)
# stage 3 は **これを walk_loop_shoot と同じ広い帯へ広げるだけ** で、DR の項目
# (足の反発 / ボール物性 / 初期回転 / 転がり減速) は 1 つも増やさない。
#
# 値の出どころは :mod:`..walk_loop_shoot.walk_loop_shoot_env_cfg` の同名定数で、
# 「IsaacLab (摩擦 1.0/0.8, restitution 0.6) と MuJoCo (摩擦 0.4, 反発ほぼ 0) の
# 両方を内包する」という取り方。同じ蹴り方が両方の接触モデルで通るように寄せる。
# 実機がどちら寄りかは不明なので、片方に合わせるのではなく両方をカバーする。
#
# 質量スケールだけは orbit と同じ (0.9, 1.15) = 5 号球の公称 410-450 g のばらつき。
# 広げる理由が無い (実球の質量は物理的にこの範囲を出ない)。明示的に渡しているのは、
# 4 つの範囲がここに並んでいた方が「何を広げて何を据え置いたか」が読めるため。
#
# NOTE: 平坦の stage 2 では **広げない**。stage 2 の趣旨は「観測履歴だけを変えて
#       効果を切り分ける」ことなので、DR を混ぜると変更点が 2 つになる。
# --------------------------------------------------------------------------- #
_ROUGH_BALL_STATIC_FRICTION_RANGE = (0.3, 1.0)
_ROUGH_BALL_DYNAMIC_FRICTION_RANGE = (0.2, 0.8)
_ROUGH_BALL_RESTITUTION_RANGE = (0.0, 0.7)
_ROUGH_BALL_MASS_SCALE_RANGE = (0.9, 1.15)

# --------------------------------------------------------------------------- #
# :func:`_pin_curricula_at_end` が **固定せずそのまま残す** カリキュラム項。
#
# 3 つとも locomotion 側 (:class:`~..locomotion.flat_env_cfg.K1FlatCurriculumCfg`) が
# 全 K1 タスクへ配っている **環境 DR のスケジュール** で、報酬のランプではない:
#
#   * modify_command_resampling_time_range … base_velocity の再サンプリング間隔
#   * lin_vel_command_curriculum           … 線速度コマンド範囲の段階拡大
#   * modify_push_robot                    … 外乱プッシュの強さ / 間隔
#
# 残す理由が 3 つある:
#
# 1. **巻き戻る向きが「易しい方」**。フェードイン系が巻き戻ると「蹴らない方が得」に
#    なるのが問題なのに対し、こちらが巻き戻ると外乱が弱く・コマンド帯が狭くなる
#    だけで、収支が逆転する経路が無い。
# 2. **窓が短い**。num_steps は raw step で 6000-14000 = 250-583 iteration
#    (steps_per_iteration = 24)。fine-tune の既定 3000 iteration のごく序盤で
#    終値に着く。
# 3. ``lin_vel_command_curriculum`` はそもそも壁時計のランプではなく **追従誤差で
#    段が進むゲート**。「終値を書き込む」という操作が意味を持たない。
#
# また、このリポジトリの全 PLAY cfg が既にこの 3 項を生かしたまま回している
# (CurriculumManager は PLAY でも step 0 から走る) ので、ここだけ挙動を変えると
# 他タスクの PLAY と比較できなくなる。
#
# **func の identity で判定する** (名前ではなく)。名前で書くと、将来 cfg 側で項名を
# 変えたときに黙って NotImplementedError 側へ落ちる。
# --------------------------------------------------------------------------- #
_UNPINNED_CURRICULUM_FUNCS = (
    modify_command_resampling_time_range,
    lin_vel_command_curriculum,
    modify_push_robot,
)


def _pin_curricula_at_end(cfg: "K1WalkKickEnvCfg", *, expansion_alpha: float = 1.0) -> list[str]:
    """全カリキュラム項の **終値を対象へ直接書き込み、項そのものを None にする**。

    なぜ必要か
    ----------
    stage 2/3 は ``--load_pretrained`` で **収束済み**の checkpoint から始める
    (基準 run 2026-08-22_11-56-42、3600 iteration、カリキュラムは全て 3000 で終点に
    到達済み)。``--load_pretrained`` は ``--resume`` と違って ``common_step_counter`` を
    引き継がず 0 から数え直すので、カリキュラムを生かしたままだと **全部のランプが
    巻き戻る**:

    * キック報酬 4 項 (direction / scaled / inside_contact / plant_lon) が weight 0 から
      フェードインし直す。この間 ``kick_finished`` は「残りの歩行報酬を捨てるコスト」
      だけを課すので、**最初の 500 iteration は蹴らない方が得**が明示的に成立する
      (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg._freeze_fade_in_curricula` の
      docstring にある実測。500 iteration 後に weight が戻っても、そのときには
      蹴らなくなっているので払われる先が無い)。
    * 拡大ゲートの α が 0 に戻り、ボールが ±60°・0.5-0.8 m の限定レンジに縮む。
      収束済みポリシーには易しすぎるうえ、``ball_avoidance`` が 0 に落ちて
      ``approach_penalty`` が復活するので、回り込みの構えを壊す方向に更新される。
    * ``sigma_velocity`` が 0.5 → 1.0 に戻り、速度の採点が緩む
      (「指令どおりに蹴る」の圧が消える)。
    * ``kick_plant_lon`` の ``lon_span`` が 0.15 → 0.45 に戻り、勾配が 40 → 13.3 に鈍る。
      軸足の踏み込み (plant_lon −0.11) は実機の転倒事故への直接の対策なので、
      ここが緩むのがいちばん困る。
    * ``kick_velocity_strong`` が満額 (=「速く蹴るほど得」) で復活する。この項は
      **トーキックを名指しで要求する**項で、退場させたのがこのタスクの肝
      (:data:`_INSIDE_STRONG_KNOTS`)。

    なぜ ``_freeze_fade_in_curricula`` を使い回さないのか
    ----------------------------------------------------
    あちらは (1) ``func`` が ``linear_reward_weight`` の項しか見ず、(2) ``end_step`` が
    ``before_iter`` 以下のものだけを対象にし、(3) 項は残したまま
    ``start_weight = end_weight`` に潰す、という作り。このタスクで足りないのは
    (1) と (2) の方:

    * ``piecewise_reward_weight`` (strong の折れ線) と ``linear_reward_param``
      (σ_velocity) と ``piecewise_reward_param`` (lon_span の 3 段) と
      ``kick_rate_gated_expansion`` (拡大ゲート) は対象外なので、そのまま巻き戻る。
    * ``kick_velocity_overshoot_weight`` の窓は 1500 → 3000 なので、
      ``before_iter = 500`` では拾えない。基準 run では完走しているので凍結が正しい。

    (3) の「項を残す」は、まだ動く窓が後ろに残っている段では利点 (今いくつなのかが
    ``Curriculum/...`` に出続ける) だが、**全部の窓が既に閉じているこの段では
    ただのノイズ**。項ごと ``None`` にすると:

    * 「カリキュラムはもう 1 本も無い」を関数の最後に検査できる (下の assert)。
      定数化しただけだと、新しい項が足されたときに黙って巻き戻る側へ回る。
    * PLAY cfg が自動的に正しくなる。CurriculumManager は PLAY でも
      ``common_step_counter`` 0 から走るので、項が生きていると PLAY で見る値が
      学習終盤と食い違う (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.hold_sigma_direction`
      の「2. アニールが入っている段の PLAY」と同じ機序)。継承だけで直る。

    Args:
        cfg: ``__post_init__`` を通した後の env cfg。
        expansion_alpha: 拡大ゲートを固定する α [0, 1]。既定 1.0 = 全方位。
            基準 run 2026-08-22_11-56-42 の ``Curriculum/kick_expansion/alpha`` は
            3643 iteration 時点で **1.0** (kick_rate_ema 0.996) なので既定でよい。
            別の checkpoint から始めるときは、その run の同じタグを見て合わせること
            (実力より広い範囲を固定すると、ゲートが本来やる「崩れたら戻る」が
            効かない状態で難易度だけ据え置かれる)。

    Returns:
        固定した curriculum 項の名前 (呼び出し側の検証・表示用)。

    Raises:
        NotImplementedError: 終値の意味が分からない ``func`` の項が残っていたとき。
            **黙って巻き戻らせないため、握り潰さずに落とす。** 新しいカリキュラムを
            足したら、この関数にも固定の仕方を書くこと。
    """
    pinned: list[str] = []

    for name in sorted(dir(cfg.curriculum)):
        if name.startswith("_"):
            continue
        term = getattr(cfg.curriculum, name, None)
        # configclass のメソッド (to_dict / replace など) も dir() に出るので、
        # CurrTerm であることを型で確かめてから触る。
        if not isinstance(term, CurrTerm):
            continue

        func = term.func
        params = term.params

        if func in _UNPINNED_CURRICULUM_FUNCS:
            # 報酬のランプではない環境 DR のスケジュール。理由は
            # :data:`_UNPINNED_CURRICULUM_FUNCS` のコメント。
            continue

        if func is mdp.linear_reward_weight:
            _reward_term(cfg, params["term_name"], name).weight = params["end_weight"]

        elif func is mdp.piecewise_reward_weight:
            # 折れ線は最後の knot の weight で頭打ちになる (piecewise_reward_weight の
            # 実装: step >= knots[-1][0] なら knots[-1][1])。このタスクの strong は
            # :data:`_INSIDE_STRONG_KNOTS` の最終 knot が (1200, 0.0) なので 0.0。
            _reward_term(cfg, params["term_name"], name).weight = params["knots"][-1][1]

        elif func is mdp.linear_reward_param:
            # σ_velocity (1.0 → 0.5)。
            _reward_term(cfg, params["term_name"], name).params[params["param_name"]] = params["end_value"]

        elif func is mdp.piecewise_reward_param:
            # 折れ線の params 版。piecewise_reward_weight と同じく最後の knot で
            # 頭打ちになる (実装: step >= knots[-1][0] なら knots[-1][1])。
            # このタスクでは kick_plant_lon の lon_span
            # (:data:`_PLANT_LON_SPAN_KNOTS`) の最終 knot = (4000, 0.15)。
            _reward_term(cfg, params["term_name"], name).params[params["param_name"]] = params["knots"][-1][1]

        elif func is mdp.window_reward_weight:
            # 「start_step < step <= end_step の間だけ weight、外は 0」。窓の外 =
            # end_step より後が終状態なので 0。**現在このタスクには 1 つも無い**が、
            # weak/middle 側に足されたときに黙って巻き戻らないよう先に書いてある
            # (窓が生きていると fine-tune の序盤だけ罰/報酬が復活する)。
            _reward_term(cfg, params["term_name"], name).weight = 0.0

        elif func is mdp.kick_rate_gated_expansion:
            _pin_expansion_gate(cfg, params, expansion_alpha)

        else:
            raise NotImplementedError(
                f"curriculum.{name} (func={getattr(func, '__name__', func)}) の終値の固定方法が "
                "_pin_curricula_at_end に書かれていません。"
                "stage 2/3 は収束済み checkpoint からの fine-tune なので、"
                "巻き戻るランプが 1 本でも残っていると型が壊れます。"
                "固定の仕方をこの関数に足すか、巻き戻ってよい理由を "
                "_UNPINNED_CURRICULUM_FUNCS に書いて除外してください。"
            )

        setattr(cfg.curriculum, name, None)
        pinned.append(name)

    # -- 検算: ランプが 1 本も残っていないこと ----------------------------- #
    #
    # 「新しいカリキュラム項を足したのにこの関数を直し忘れる」を起動時に落とすための
    # 検査。上のループが NotImplementedError で守っているので通常は到達しないが、
    # 除外リストの誤用 (報酬ランプを間違って入れる) はここでしか捕まらない。
    remaining = [
        n
        for n in sorted(dir(cfg.curriculum))
        if not n.startswith("_")
        and isinstance(getattr(cfg.curriculum, n, None), CurrTerm)
        and getattr(cfg.curriculum, n).func not in _UNPINNED_CURRICULUM_FUNCS
    ]
    if remaining:
        raise AssertionError(f"固定されなかった curriculum 項が残っています: {remaining}")

    return pinned


def _reward_term(cfg: "K1WalkKickEnvCfg", term_name: str, curr_name: str):
    """カリキュラムの ``term_name`` が指す報酬項を取り出す (無ければ落とす)。

    ``None`` の報酬項に weight を書いても ``AttributeError`` になるだけで
    「なぜ壊れたか」が読めないので、curriculum 項の名前を添えて先に落とす。
    """
    term = getattr(cfg.rewards, term_name, None)
    if term is None:
        raise AssertionError(
            f"curriculum.{curr_name} が指す報酬項 rewards.{term_name} がありません "
            "(報酬項だけ None にして curriculum 項を消し忘れている可能性)。"
        )
    return term


def _pin_expansion_gate(cfg: "K1WalkKickEnvCfg", params: dict, alpha: float) -> None:
    """:func:`~..walk_kick.mdp.curriculums.kick_rate_gated_expansion` が α で動かす
    5 つの対象に、α を固定した値を直接書き込む。

    **値は全て term 自身の params から読む** (このモジュールの定数を再参照しない)。
    ゲートの設定を :func:`_apply_inside_kick_recipe` で変えたときに、固定側だけ
    古い値のまま残る事故を構造的に防ぐため。

    α = 1 のとき ``approach_penalty`` の weight は **0** になる (``end_weight`` では
    ない)。あちらは ``approach_end_weight × fade × (1 − α)`` というクロスフェードで、
    全方位に届いた時点で「ボールに寄れ」の圧は ``ball_avoidance`` (寄るな) に
    完全に置き換わるため。``end_weight`` を書き込むと、収束済みポリシーに対して
    **互いに打ち消し合う 2 つの罰を同時に掛ける**ことになる。
    """
    def lerp(a: float, b: float) -> float:
        return a + (b - a) * alpha

    half_angle_range = params["half_angle_range"]
    dist_start, dist_end = params["dist_range_start"], params["dist_range_end"]
    heading_range = params["heading_halfwidth_range"]

    ball_event = getattr(cfg.events, params["ball_event_name"])
    ball_event.params["half_angle"] = lerp(*half_angle_range)
    ball_event.params["dist_range"] = (lerp(dist_start[0], dist_end[0]), lerp(dist_start[1], dist_end[1]))

    heading_half = lerp(*heading_range)
    getattr(cfg.commands, params["command_name"]).ranges.heading = (-heading_half, heading_half)

    # approach (寄れ) → avoidance (寄るな) のクロスフェード。fade は
    # min(now / approach_fade_iterations, 1) で、固定する時点では既に 1。
    # α = 1 では積が -0.0 になる。値は 0.0 と等価だがログ表示が紛らわしいので +0.0 で均す。
    getattr(cfg.rewards, params["approach_term_name"]).weight = params["approach_end_weight"] * (1.0 - alpha) + 0.0
    getattr(cfg.rewards, params["avoidance_term_name"]).weight = params["avoidance_end_weight"] * alpha


@configclass
class K1WalkInsideKickDualEnvCfg(K1WalkInsideKickEnvCfg):
    """stage 2 (平坦): actor に 100 フレームの観測履歴を与える。

    :class:`K1WalkInsideKickEnvCfg` との差は **2 つだけ**:

    1. カリキュラムを全て終値に固定する (:func:`_pin_curricula_at_end`)。
    2. policy 観測グループを 100 フレームの履歴にする
       (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)。

    ``__post_init__`` の順序に意味がある::

        super()                 … インサイドのレシピ + ボール観測ノイズ (継承のまま)
        _pin_curricula_at_end() … 報酬・イベント・コマンドが全部揃った後に固定する
        enable_obs_history()    … **最後**。観測グループの構成が固まってから履歴化する

    ``enable_obs_history`` を最後に置くのは walk_lob_rough の Stage 3
    (:class:`~..walk_lob_rough.walk_lob_rough_env_cfg.K1WalkLobHistEnvCfg`) と同じ流儀。
    ``history_length`` は ObservationGroup 全体に掛かるフラグなので、後から観測項を
    足しても壊れはしないが、「グループの形を変える操作は最後」を守っておけば
    順序依存を考えなくて済む。

    dual 系から **持ち込まないもの** (全て意図的)
    ---------------------------------------------
    :mod:`..walk_kick_dual` は履歴のほかに 4 つの変更を畳み込んでいるが、この段は
    **観測履歴だけを変えて効果を帰属させる**のが目的なので 1 つも入れない:

    * ``K1WalkKickBothFeetObservationsCfg`` (観測スロット 3 を左足裏 → ボール 3D 位置、
      critic 58 次元) — 入れると stage 1 の checkpoint が **意味の上で** 繋がらなくなる
      (55 次元なので ``--load_pretrained`` は形の上では通ってしまうぶん、たちが悪い)。
      基底の policy 55 次元 / critic 61 次元をそのまま使う。
    * ``_apply_phase_offset`` (歩行位相の初期オフセット {0, π}) — 両足で蹴れるように
      するための変更。**このタスクは右足専用** (``kick_inside_contact`` が右足ゲート付き)
      なので、蹴り足を割る意味が無い。
    * ``disable_landing_shaping`` / ``rebalance_gait_vs_kick`` — 報酬の変更。stage 1 が
      その報酬集合で収束しているので、ここで動かすと「履歴の効果」が読めなくなる。
      必要になったら **別の段** として足すこと。
    * ``enable_obs_delay`` — ボール観測にはこのタスク独自の認識パイプライン
      (:func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs`: エピソードごとの
      ランダム遅延 2-6 step + 30 Hz サンプル&ホールド + フレーム同期ジッタ) が
      既に載っており、``enable_obs_delay`` はパイプライン付きの位置スロットを
      **二重掛け防止のため飛ばす** 作りなので、掛けても実質 IMU / エンコーダにしか
      効かない。内界センサの遅延 DR は「履歴の効果を見る」この段の目的と別件なので、
      入れるなら stage 3 以降に単独で足す。
    * mirror loss (``PPOSparseMirror`` / ``_use_mirror_loss``) — 右足専用タスクなので
      鏡像対称性が成り立たない。RunnerCfg 側の話なので
      :mod:`.agents.rsl_rl_ppo_cfg` の ``_use_history_cnn_policy`` を参照。

    引き継ぎ::

        ./scripts/rsl_rl/train_walk_inside_kick_dual.sh          # STAGE=23 が既定

    stage 1 の checkpoint は 1 フレーム観測なので ``--warm_start_from_single_frame``
    が要る (スクリプトが自動で付ける)。詳細はモジュール docstring。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. カリキュラムを終値へ固定する ------------------------------ #
        #
        # --load_pretrained は common_step_counter を 0 に戻すので、生かしたままだと
        # キック報酬のフェードイン・拡大ゲート・σ_velocity・lon_span・strong の
        # 全部が巻き戻る。理由の詳細は :func:`_pin_curricula_at_end`。
        _pin_curricula_at_end(self)

        # -- 2. 観測履歴 (この段で変えるのはここだけ) --------------------- #
        #
        # 必ず最後。policy グループの構成が固まってから (N, H, 55) に変える。
        enable_obs_history(self)


@configclass
class K1WalkInsideKickDualEnvCfg_PLAY(K1WalkInsideKickDualEnvCfg):
    """stage 2 の PLAY。:class:`K1WalkInsideKickEnvCfg_PLAY` と同じ調整。

    カリキュラムは親で全て ``None`` にしてあるので、PLAY でも
    ``common_step_counter`` 0 から巻き戻る心配は無い (項が 1 つも無い)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # enable_corruption = False は ObsTerm の noise しか切らないので、
        # 関数側に移したジッタは別途切る。遅延とサンプル&ホールドは観測パイプラインの
        # 構造 (= PLAY で見たいもの) なので残す。
        _disable_ball_obs_jitter(self)


@configclass
class K1WalkInsideKickDualRoughEnvCfg(K1WalkInsideKickDualEnvCfg):
    """stage 3 (凹凸 + ボール物性 DR の拡大): 実機の床とボールのばらつきへ寄せる最終段。

    stage 2 との差は 2 つだけ:

    1. 地形を ±1-4 cm のランダム凹凸へ
       (:func:`~..walk_kick.walk_kick_env_cfg._apply_rough_terrain`、
       :data:`~..walk_kick.walk_kick_env_cfg.WALK_KICK_ROUGH_TERRAIN_CFG`)。
       ボールは ``spawn_clearance`` = 5 cm 浮かせて落とす (凹凸の振幅より上)。
    2. ボール物性 DR の帯を walk_loop_shoot 相当まで広げる
       (:data:`_ROUGH_BALL_STATIC_FRICTION_RANGE` ほか)。

    ``__post_init__`` の順序について
    --------------------------------
    ``super()`` が既に :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`
    を掛け終わっているが、**問題ない**。:func:`_apply_rough_terrain` が触るのは
    ``scene.terrain`` の 3 属性と ``events.reset_ball.params["spawn_clearance"]`` だけで、
    観測グループにも報酬にもコマンドにも一切触らないため (あちらの docstring と実装を
    確認済み)。同じ理由で :func:`~..walk_weak_kick_orbit.orbit_mods.apply_ball_param_dr`
    の再呼び出しも観測に影響しない。

    ``apply_ball_param_dr`` を 2 回呼ぶことについて
    ----------------------------------------------
    ``super()`` の中 (:func:`_apply_inside_kick_recipe` の第 9 節) で 1 回、ここで
    もう 1 回。あちらは **差分ではなく上書き** なので冪等:
    ``events.ball_physics_material`` / ``events.ball_mass`` は EventTerm を作り直して
    代入、``reset_ball`` の spin 2 キーと ``soccer_ball.spawn.rigid_props`` も同じ値の
    再代入で、累積するものは 1 つも無い。したがって 2 回目の呼び出しは
    「範囲だけ差し替わった 1 回目」と等価になる。

    .. warning::
       **凹凸地形 + ボールの組み合わせはこのリポジトリで学習実績が無い**
       (``k1_walk_kick_rough`` / ``k1_walk_lob_rough`` 系の log が存在しない。
       :class:`~..walk_lob_rough.walk_lob_rough_env_cfg.K1WalkLobRoughKickEnvCfg` の
       同じ警告を参照)。必ず平坦の stage 2 を先に通し、その checkpoint から入ること。
       立ち上がりで ``kick_rate`` が落ちるのは想定内だが、数百 iteration で戻って
       こないなら地形が厳しすぎる (地形の振幅を下げるより先に、まず stage 2 の
       checkpoint が本当に繋がっているかを "Skipped N tensors" で確認すること)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 地形だけ凹凸へ -------------------------------------------- #
        _apply_rough_terrain(self)

        # -- 2. ボール物性 DR の帯を広げる -------------------------------- #
        #
        # 項目は増やさず範囲だけ差し替える (:data:`_ROUGH_BALL_STATIC_FRICTION_RANGE`)。
        # 初期回転・転がり減速・足の反発は基底のレシピと同じ値が再代入される。
        apply_ball_param_dr(
            self,
            static_friction_range=_ROUGH_BALL_STATIC_FRICTION_RANGE,
            dynamic_friction_range=_ROUGH_BALL_DYNAMIC_FRICTION_RANGE,
            restitution_range=_ROUGH_BALL_RESTITUTION_RANGE,
            mass_scale_range=_ROUGH_BALL_MASS_SCALE_RANGE,
        )


@configclass
class K1WalkInsideKickDualRoughEnvCfg_PLAY(K1WalkInsideKickDualRoughEnvCfg):
    """stage 3 の PLAY。stage 2 の PLAY + generator 地形用のカメラ設定。

    :func:`~..locomotion.rough_env_cfg._apply_play_viewer` を足すのは
    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickRoughEnvCfg_PLAY` と同じ理由。
    terrain generator は env origin を地形グリッドに割り当てるので、既定の
    world 固定カメラ (原点を見つめたまま動かない) だと ``play.py --video`` に
    **地形しか映らない**。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)
        _apply_play_viewer(self)
