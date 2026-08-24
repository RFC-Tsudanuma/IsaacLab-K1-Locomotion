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

(この「指標としてだけ見る」は初版の判断。実機のフィードバックを受けて一度は
**形を変えて報酬に戻した** — 次の 2 節の ``kick_plant_lon`` / ``kick_plant_yaw`` —
が、2026-08-24 に **再び報酬から外し、指標だけに戻した**。撤回の理由は各節の末尾。
``kick_plant_foot`` 自体は一貫して外したままである。)

軸足をもう一度誘導することになった経緯 (kick_plant_lon の新設) — 2026-08-24 に撤回
------------------------------------------------------------------------------
.. note::
   **この節は履歴。``kick_plant_lon`` は 2026-08-24 に報酬から外した** (結論は
   節の末尾)。残してあるのは「軸足を報酬で誘導しようとして、動かせはしたが
   当たりの質が改善しなかった」という実測ごと捨てないため。

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

結末 (2026-08-24): 報酬から外した
.................................
線形テントは狙いどおり **動いた** — plant_lon は weight 6.0 で −0.23 → −0.107 まで
寄り、kick_plant_foot の「勾配が無くて動かない」は確かに解消できた。
それでも外したのは、**動かした結果が当たりの質を良くしたという証拠がどこにも無い**
ため:

* 実機で残った症状 (巻き込んで転ぶ) は plant_lon が −0.11 まで来た checkpoint でも
  消えず、原因は別のところ (接触点の高さ) にあることが 2026-08-24 に判明した
  (下の「実機フィードバック 3 回目」)。
* B-Human の観察 (2026-08-23) — **軸足の位置も向きも当たりの質と無関係**。
  軸足は「物理的に無理な場所になければよい」だけで、報酬で誘導する対象ではない。
* ロブ側の実測 (run 2026-08-23_08-05-22) が同じ結論を別経路で裏づけた。
  ``Episode_Reward/kick_plant_lon`` ≈ **0.009** = 実質払われておらず、
  plant_lon はむしろ −0.50 側へ流れていた。

したがって方針は「**軸足は報酬で誘導しない**」に転換した。指標
(``plant_lon`` / ``plant_lat`` / ``plant_yaw_dot``) は出し続けるので、
当たりの質が変わったときに軸足がどう動いたかは後から読める。

実機フィードバック 2 回目 (2026-08-23) — 3 項中 2 項は撤回、1 項は狙いを変えて存続
------------------------------------------------------------------------------
1 回目 (「振りが手前すぎてボールを巻き込んで転ぶ」→ ``kick_plant_lon`` の新設) の
あと、実機で残っていた症状は **当たりが薄い / 空振りする** と **軸足がまだ手前**
の 2 つ。当時は 3 つの変更で対処した。

1. **軸足の向き** (``kick_plant_yaw``、weight 3.0) — **2026-08-24 に撤去**

   狙いは「軸足のつま先を蹴り方向へ向かせれば骨盤もそちらを向き、振り足の
   インサイド面がキック線に正対する」。角度に対する線形テント (半幅 90°) で、
   胴体は p_style の帯で 30-45° ずれてよく **ずれるのは胴体で軸足は向かせる**
   という役割分担だった (分ける自由度は Hip_Yaw)。

   撤回の理由: B-Human の観察 (2026-08-23) では **軸足の向きは当たりの質と無関係**。
   ロブ側の実測 (run 2026-08-23_08-05-22) では ``plant_yaw_dot`` が
   **最初から 0.93 で飽和**しており、この項が動かせる余地がそもそも無かった
   (「素の値が既に 0.9 級ならこの項は効きようが無い」と当時書いた条件が
   そのまま当たった)。

2. **軸足の目標を 0 cm へ + span の第 3 段** — **2026-08-24 に撤去**

   ``lon_target`` −0.03 → 0.0、``lon_span`` の折れ線に 3000 → 4000 で
   0.25 → 0.15 の第 3 段。テントの勾配は W/span なので **目標だけを動かしても
   政策は動かない、踏み込ませるレバーは span** という読みは今も正しい。
   撤回の理由は上の「結末 (2026-08-24)」の節と同じ (動かせはしたが、当たりの質が
   良くなった証拠が無い / B-Human 観察 / ロブでの死亡)。

3. **接触点の高さ** (``kick_foot_ceiling``、weight 3.0、第 6c 節) — **存続。
   ただし 2026-08-24 に狙いを「天井」から「低く当てろ」へ変えた**

   関数の形 (``f = clamp(1 − (h − h_target)/h_span, 0, 1)``) は 1 ビットも変えて
   いない。変えたのは cfg 側の ``h_cap`` 0.09 → ``h_target`` 0.05 だけ。
   0.09 は「足箱の中心がボール中心以下に来る」上限として置いた値だったが、
   **収束値 0.087 のすぐ上に座っていて何も止めていなかった**。詳細は次節。

実機フィードバック 3 回目 (2026-08-24) — 巻き込みの真因は接触点の高さだった
-------------------------------------------------------------------------
実機でインサイドキックが **ボールを巻き込んで転ぶ**。軸足を 2 回いじっても消えな
かったこの症状の原因が、run 2026-08-22_11-56-42 のログで特定できた。

* ``Metrics/kick_direction/sole_height_at_kick`` が学習中に **0.051 → 0.087** へ
  上がっており、その上昇が ``Episode_Reward/kick_velocity_scaled`` の伸びと
  **完全に同期していた**。
* 0.087 のとき足箱の中心は 0.087 + 0.018 = **0.105** = ボール中心 0.11 の
  わずか **5 mm 下**。つまり真芯当たりで、下から当てる余裕がゼロ。
* 理由ははっきりしている: ``kick_velocity_scaled`` が **水平速度**で採点していた
  ため、ボール中心を水平に突く「真芯当たり」が最適解になっていた。少しでも下から
  当てて浮かせると、その鉛直成分は水平ノルムでは **損**にしか写らない。

対策は 2 つ (どちらも :func:`_apply_inside_kick_recipe`、stage 2/3 にも継承で載る)。

1. **``kick_velocity_scaled`` を 3D 速度で採点する** (``use_3d_speed=True``、第 1c 節)

   下から当てて浮いた成分が損にならなくなる。latch の閾値 (``v_thresh``) と
   ``Metrics/kick_direction/kick_vel_ratio`` は水平のまま = **触らない**ので、
   「どれだけ蹴れたか」の読み方は過去 run と変わらない。

2. **接触点を「上限」から「低いほど得」へ** (``_FOOT_LOW_H_TARGET`` 0.05、第 6c 節)

   h ≤ 0.05 で満点、0.13 で 0。現在の収束値 0.087 では f ≈ 0.54 なので、
   **いま居る場所に勾配が生きている**まま「もっと低く」が効く。
   0.05 は学習初期 (iter 455) に自然に居た値でもある。

**軸足の項 (plant_lon / plant_yaw) はこの回で両方とも撤去した。** 方針は
「軸足は物理的に無理な場所になければよく、報酬で誘導しない」(B-Human 観察 +
ロブの実測)。指標としては出し続ける。

TensorBoard で見るもの (2026-08-24 版)
--------------------------------------
* ``Metrics/kick_direction/sole_height_at_kick`` — **本命の判定材料**。
  0.087 から **0.05 側へ動くか**。動かないなら ``_FOOT_LOW_WEIGHT`` を
  3.0 → 6.0 へ上げるのが次の一手 (上限の根拠は同定数のコメント)。
* ``Metrics/kick_direction/kick_vel_ratio`` — **威力とのトレード**の監視。0.887 が
  基準。この比は **水平のまま**なので、3D 化で下から当てるようになると
  仰角ぶん下がって見えるのが正常 (仰角 20° なら cos20° = 0.94 倍)。
  それ以上に落ちているなら、低く当てるために立ち位置を詰めてスイングを削っている
  (walk_lob の ``kick_contact_height`` の失敗と同じ形)。
* ``Metrics/kick_direction/plant_lon`` / ``plant_lat`` / ``plant_yaw_dot`` —
  **観察用 (報酬からは外した)**。当たり方が変わったときに軸足がどう動いたかを
  後から読むための記録で、これらが目標へ寄ったかどうかは判定材料ではない。
  基準値は plant_lon −0.107 / plant_lat 0.30 / plant_yaw_dot 0.9 級。
* ``Metrics/kick_direction/foot_kick_dot`` / ``ball_side`` — 当たり所そのもの。
  接触点を下げる圧でインサイド面から外れていないかを見る。

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
    Metrics/kick_direction/plant_lon     -0.107   (当時の kick_plant_lon の成果。
                                                   いまは報酬から外した = 観察用)
    Metrics/kick_direction/foot_kick_dot -0.030   (足が真横 = 側面で当てている)
    Metrics/kick_direction/kick_vel_ratio 0.887
    Metrics/kick_direction/sole_height_at_kick 0.087  (**高すぎる**。2026-08-24 の
                                                   変更で 0.05 側へ動かす対象)

ここから先は「同じ型のまま実機へ寄せる」フェーズで、こちらは **fine-tune の段** に
分ける。1 run でまとめて掛けると、崩れたときに履歴のせいなのか地形のせいなのか
DR のせいなのか切り分けられないため。

* **stage 2** = :class:`K1WalkInsideKickDualEnvCfg` (平坦、観測履歴 + fewa の束)

  actor の入力を 100 フレームの履歴にする (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)。
  初版は「変えるのはそれだけ」= 履歴の効果に帰属させる段だったが、実機の
  フィードバックを受けて **2026-08-24 に帰属の純度を捨て、fewa/47b8863 の設定を
  丸ごと持ち込む段へ変えた**:

  * 歩容側の 3 点 (着地 shaping 3 項を無効化 / ``feet_phase`` 2.0 → 0.8 /
    地形を NOISY_FLAT)。
  * **観測パイプラインごと fewa 方式へ** (2026-08-24): 継承元の
    :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` (30 Hz サンプル&
    ホールドのガウス系パイプライン) を捨て、fewa の
    :func:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.enable_obs_delay`
    (連続遅延 + 一様ノイズ、IMU / エンコーダにも遅延) に置き換える。
    そのために継承元を :class:`K1WalkInsideKickCleanEnvCfg` へ張り替えてある。

  理由は :class:`K1WalkInsideKickDualEnvCfg` の docstring の
  「観測パイプラインを fewa 方式へ (2026-08-24)」節。ここで指標が動いても
  **原因を 1 つに帰属させることはできない** (それを承知で束ごと合わせる段)。

* **stage 3** = :class:`K1WalkInsideKickDualRoughEnvCfg` (凹凸 + ボール物性 DR 拡大)

  stage 2 の上に :func:`~..walk_kick.walk_kick_env_cfg._apply_rough_terrain` (±1-4 cm の
  ランダム凹凸) と、ボール物性 DR の帯を walk_loop_shoot 相当まで広げたものを載せる。

カリキュラムは段に入る前に終値へ固定する (:func:`_pin_curricula_at_end`)
--------------------------------------------------------------------------
stage 2/3 は ``--load_pretrained`` で **収束済み** checkpoint から始める。あちらは
``common_step_counter`` を 0 に戻すので、カリキュラムを生かしたままだと全ランプが
巻き戻る (キック報酬は 0 から、拡大ゲートは限定レンジから、σ_velocity は 1.0 から、
strong は満額から)。その帰結が
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

* ``Metrics/kick_direction/sole_height_at_kick``。基準 run の 0.087 から
  **0.05 側へ動いているか**が 2026-08-24 以降の本命 (接触点をボール中心より下へ)。
  段を越えて 0.087 へ戻るようなら、履歴なり地形なりが低い当て方を壊している。
* ``Metrics/kick_direction/foot_kick_dot`` ≈ 0。1 に向かって上がっていたら
  インサイドを捨ててトーキックへ戻っている。
* ``Metrics/kick_direction/kick_vel_ratio`` ≈ 0.88。この比は水平成分なので、
  3D 採点で下から当てるようになると仰角ぶん下がって見えるのが正常。
* ``Metrics/kick_direction/plant_lon`` ≈ −0.11 は **観察用** (報酬からは外した)。
  ここが動いていても採否の判断材料にはしない。
* ``Metrics/kick_direction/kick_rate`` ≈ 1.0。stage 3 は地形と DR が乗るので
  最初の数百 iteration は落ちてよいが、戻ってこなければ地形が厳しすぎる。
* ``Metrics/kick_direction/kick_dir_error_deg`` (基準 4.2°) と ``kick_rate`` は、2026-08-24 に
  ボール観測を fewa 方式へ張り替えた影響が **最初に出る場所**。位置ノイズが
  ±0.02 m (+ 関数内ジッタ σ 0.067) → **±0.07 m 一様**、速度ノイズが ±0.4 → ±0.5、
  さらに IMU / エンコーダにも 0-0.02 s の遅延が乗るので、1 iteration 目から
  多少悪化するのは想定内。数百 iteration で戻らないなら
  :func:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.enable_obs_delay` の
  ``_BALL_POS_NOISE`` が今の当て方には広すぎるということ (fewa は ±0.1 で崩れて
  ±0.07 に緩めた経緯があるので、同じ症状を見ている可能性が高い)。

その他は 1 iteration 目からほぼ基準値のはずで、**そうなっていなければ
checkpoint が繋がっていない** (起動ログの "Skipped N tensors" を見ること)。
観測の次元も並びも変わっていない (:func:`enable_obs_delay` は ``func`` と
``params`` を差し替えるだけ) ので、stage 1 → stage 2 の引き継ぎはこれまでどおり
成立する。

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
from ..walk_kick_dual.walk_kick_dual_env_cfg import (
    disable_landing_shaping,
    enable_obs_history,
    rebalance_gait_vs_kick,
)
# stage 2 は観測パイプラインごと fewa (47b8863 = 実機実証済み) に合わせる。
# 遅延の実装も定数も **fewa のものをそのまま** 使う (自前で書き直すと「fewa と同じ」
# が保証できなくなる)。詳細は :class:`K1WalkInsideKickDualEnvCfg` の docstring。
from ..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg import (
    _FEWA_NOISY_FLAT_TERRAIN_CFG,
    _OBS_DELAY_MAX_S as _FEWA_OBS_DELAY_MAX_S,
    enable_obs_delay as fewa_enable_obs_delay,
)
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
#         軸足を動かすレバーにはならない。一時は
#         :func:`~..walk_kick.mdp.rewards.kick_plant_lon` に軸足の誘導を分離したが、
#         それも 2026-08-24 に撤去した (軸足は報酬で誘導しない)。いずれにせよ
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
# 「形の項が目的の項を出し抜かない」という序列 (_FOOT_LOW_WEIGHT のコメント参照)
# に従うため。
#
# 項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの割り戻し
# (_KICK_W_SCALE) を掛けること。フェードインの窓は基底のキック報酬と同じ 0 → 500 で、
# 発見期には既に満額で乗っている状態にする (型が決まってから足すと、既に決まった
# トーキックの位置で f_perp が小さく、勾配が細いところから始めることになる)。
# --------------------------------------------------------------------------- #
_INSIDE_CONTACT_WEIGHT = 6.0

# --------------------------------------------------------------------------- #
# 撤去した軸足 2 項の記録 (kick_plant_lon / kick_plant_yaw) — 2026-08-24
#
# ここには ``_PLANT_LON_TARGET`` / ``_PLANT_LON_SPAN*`` / ``_PLANT_LON_WEIGHT`` /
# ``_PLANT_YAW_WEIGHT`` / ``_PLANT_YAW_SPAN`` と、lon_span の 3 段折れ線
# (``_PLANT_LON_SPAN_KNOTS``) が並んでいた。**定数ごと消した。**
#
# 何をやっていたか (経緯の全文はモジュール docstring の 2 つの節):
#   * kick_plant_lon  = 軸足の前後位置の線形テント (目標 0.0 / 半幅 0.45 → 0.25 →
#     0.15 の 3 段 / weight 6.0 = direction 同格)。
#   * kick_plant_yaw  = 軸足のヨーの線形テント (半幅 90° / weight 3.0)。
#
# なぜ消したか:
#   1. **B-Human の観察 (2026-08-23)** — 軸足の位置も向きも当たりの質と無関係。
#      軸足は「物理的に無理な場所になければよい」だけで、報酬で誘導する対象ではない。
#   2. **ロブでの実測 (run 2026-08-23_08-05-22)** — ``Episode_Reward/kick_plant_lon``
#      ≈ 0.009 = 実質払われず、plant_lon はむしろ −0.50 へ流れた。
#      ``plant_yaw_dot`` は最初から 0.93 で飽和しており、誘導する余地が無かった。
#   3. **inside では動かせた。が、それだけ** — weight 6.0 で plant_lon −0.23 →
#      −0.11 まで寄せた実績はあるが、実機の巻き込み事故はそれでも消えず、
#      真因は接触点の高さ (2026-08-24) だった。**動いた ≠ 当たりが良くなった**。
#
# 指標 (``plant_lon`` / ``plant_lat`` / ``plant_yaw_dot``) は
# :class:`~..walk_kick.mdp.commands.KickDirectionCommandCfg` が出し続けるので、
# 当たり方が変わったときに軸足がどう動いたかは後から読める。
#
# 報酬関数 :func:`~..walk_kick.mdp.rewards.kick_plant_lon` /
# :func:`~..walk_kick.mdp.rewards.kick_plant_yaw` と、
# :data:`~..walk_weak_kick_orbit.orbit_mods._KICK_STATE_REWARD_TERMS` の名簿の
# エントリは **残してある** (名簿は None ガード付きの配布先リストで、載っていない
# 項は完全に no-op。将来また試したくなったときの入口でもある)。
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 接触点を低くする項 (kick_foot_ceiling) のパラメータと重み
#
# 使う報酬関数は :func:`~..walk_kick.mdp.rewards.kick_foot_ceiling` (名前は「天井」の
# ままだが、形は ``f = clamp(1 − (h − h_target)/h_span, 0, 1)`` の一次関数一本で、
# **h_target をどこに置くかで「上限」にも「低いほど得」にもなる**)。
# 2026-08-23 の初版は上限として (h_cap = 0.09)、2026-08-24 から **低く当てさせる圧**
# として (h_target = 0.05) 使っている。関数側は 1 ビットも変えていない。
#
# _FOOT_LOW_H_TARGET = 0.05:
#   これ以下なら満点になる足裏高さ [m]。足コライダーは足リンク原点から
#   z = −0.038 (足裏) 〜 −0.002 (上面) の厚み 0.036 の箱なので、足裏高さ h のとき
#   足箱の中心は h + 0.018。h = 0.05 なら足箱の中心 0.068 = **ボール中心 0.11 の
#   42 mm 下**。
#
#   なぜ 0.09 では駄目だったのか (実測):
#     run 2026-08-22_11-56-42 で ``sole_height_at_kick`` は学習中に
#     **0.051 → 0.087** へ上がり、その上昇は ``kick_velocity_scaled`` の伸びと
#     完全に同期していた (水平速度で採点していたので「真芯当たり」が最適解だった。
#     モジュール docstring の「実機フィードバック 3 回目」)。
#     0.087 のとき足箱の中心は 0.105 = ボール中心の **5 mm 下** = 余裕ゼロ。
#     そして当時の h_cap = 0.09 は **その収束値のすぐ上に座っていた**ので、
#     f = 1 のまま何も止めていなかった。上限として置いた値が上限として働かない、
#     という一番よくない失敗の仕方をしている。
#
#   0.05 の根拠は 2 つ:
#     * **学習初期 (iter 455) に自然に居た値**。物理的に無理な高さではないことが
#       同じ run の実測で分かっている。
#     * **実機側のオフセットに 1-2 cm の余裕がある**。芝の沈み・ボール半径の個体差・
#       立ち高さのずれで数 cm ずれても、足箱の中心 0.068 はボール中心より下に留まる。
#
# _FOOT_LOW_H_SPAN = 0.08:
#   満点から 0 点になるまでの幅 [m] (h = 0.13 で 0)。**0.09 時代から据え置き。**
#   現在の収束値 0.087 では f = 1 − 0.037/0.08 = **0.54** で、
#   「まだ潰れていないのに伸びしろが大きい」位置に乗る = いま居る場所に勾配がある。
#   狭くすると壁が急になるぶん、天井の外に居るあいだ勾配が死ぬ
#   (:func:`~..walk_kick.mdp.rewards.kick_plant_foot` の死因)。
#
# _FOOT_LOW_WEIGHT = 3.0:
#   **形の項は objective (direction 6.0) の半分から入れる**、という序列に従う。
#   上限は direction 同格の 6.0 で、根拠はこの項が r_direction への **乗算・非負**
#   であること: 方向ゲート (kick_done / τ_direction / p_style) を通らない蹴りには
#   1 円も払われないので農作の抜け道は構造で塞がっており、残る壊れ方は
#   「威力・精度を多少削ってでも形を取る」というトレードだけ。direction 同格までなら
#   objective 側 (direction / kick_velocity_scaled) がそれぞれ 1:1 以上で対抗でき、
#   形が目的を出し抜けない。それを超えて積むと出し抜けるようになるので 6.0 が上限。
#
#   **エスカレーションのレバーはこの weight**。``sole_height_at_kick`` が 0.087 から
#   下りてこないなら 3.0 → 6.0 へ上げる (span を絞る手もあるが、0.08 を狭めると
#   現在位置の f が急に落ちるので、まず weight で試すこと)。
#   副作用の監視は ``kick_vel_ratio`` (低く当てるために立ち位置を詰めてスイングを
#   削っていないか) と ``kick_inside_contact`` (当たり所と取り合いになっていないか)。
#
#   **kick_contact_height は入れないこと。** あちらは walk_lob で反証済み
#   (sole_height は下がったが apex 0.340 → 0.234)。同じ「低いほど得」でも、
#   あちらは ``f_low = clamp((R − h)/(R − h_sat), 0, 1)`` で **h_sat = 0.03 まで
#   下げ続ける圧**を掛けるのに対し、この項は h_target = 0.05 で頭打ちになる
#   (それ以下は 1 円も増えない)。両方入れると 0.05 以下でも下向きの圧だけが残り、
#   結局あちらと同じ形になる。
#
#   項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの
#   割り戻し (_KICK_W_SCALE) を掛けること。
# --------------------------------------------------------------------------- #
_FOOT_LOW_WEIGHT = 3.0
_FOOT_LOW_H_TARGET = 0.05
_FOOT_LOW_H_SPAN = 0.08

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

    # -- 1c. kick_velocity_scaled を 3D 速度で採点する ---------------------- #
    #
    # 実機フィードバック 3 回目 (2026-08-24) の 1 つ目。**水平速度で採点していたのが
    # 巻き込みの真因だった。**
    #
    # 実測 (run 2026-08-22_11-56-42): ``Metrics/kick_direction/sole_height_at_kick``
    # が学習中に 0.051 → 0.087 へ上がり、その上昇が ``Episode_Reward/kick_velocity_scaled``
    # の伸びと **完全に同期していた**。水平ノルムで採点すると、ボール中心 (0.11) を
    # 水平に突く「真芯当たり」が最適解になる — 少しでも下から当てて浮かせると、
    # その鉛直成分は水平では **損** にしか写らないため。0.087 のとき足箱の中心は
    # 0.105 = ボール中心のわずか 5 mm 下で、下から当てる余裕がゼロだった。
    #
    # 3D ノルムにすると、下から当てて浮いた成分が損にならない。これで第 6c 節
    # (接触点を低くする項) と ``kick_velocity_scaled`` が同じ向きを向く。
    # long_pass / loop 系 (:mod:`..walk_loop_pass`) が同じ理由で先に採っている設定。
    #
    # **latch の閾値とメトリクスは水平のまま = 触らない。**
    #   * ``v_thresh`` (:func:`~..walk_kick.mdp.kick_state.kick_state` の latch 条件)
    #     は水平速度で判定する。ここを 3D にすると「蹴れた」の定義が過去 run と
    #     変わってしまい、kick_rate の比較ができなくなる。
    #   * ``Metrics/kick_direction/kick_vel_ratio`` も水平のまま。3D 化で下から
    #     当てるようになると、この比は仰角ぶん下がって見えるのが正常
    #     (仰角 20° なら cos20° = 0.94 倍)。
    #   * ``kick_velocity_overshoot`` (負の重み) も水平のまま。あちらの役割は
    #     「指令帯を超えて水平に蹴りすぎるのを罰する」ことなので、水平で測るのが
    #     正しい。3D にすると仰角を付けた分だけ超過と判定されてしまう。
    #   * ``kick_velocity_strong`` も水平のまま。この項は 1200 iteration で退場する
    #     呼び水 (:data:`_INSIDE_STRONG_KNOTS`) なので、当たり所の型が決まる時期には
    #     もう居ない。
    cfg.rewards.kick_velocity_scaled.params["use_3d_speed"] = True

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

    # -- 6b. 発見期の呼び水 (inside_foot_orient) ---------------------------- #
    #
    # NOTE: ここには軸足の前後位置を誘導する ``kick_plant_lon`` (線形テント、
    #       weight 6.0、lon_span の 3 段カリキュラム) が居た。**2026-08-24 に
    #       項ごと撤去した。** 「軸足は物理的に無理な場所になければよく、報酬で
    #       誘導しない」への方針転換で、根拠は B-Human の観察 (2026-08-23) と
    #       ロブの実測 (run 2026-08-23_08-05-22 で Episode_Reward/kick_plant_lon
    #       ≈ 0.009 = 死んでいた)。inside では weight 6.0 で plant_lon を
    #       −0.23 → −0.11 まで動かせた実績があるが、**当たりの質が良くなった証拠は
    #       無く**、実機の巻き込み事故も消えなかった (真因は接触点の高さ = 第 6c 節)。
    #       経緯の全文はモジュール docstring の 2 つの節と、撤去した定数の跡に
    #       残したコメント。

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

    # -- 6c. 接触点を低くする項 --------------------------------------------- #
    #
    # 実機フィードバック 3 回目 (2026-08-24) の 2 つ目。蹴り足の足裏高さを
    # **ボール中心より下へ** 誘導する (h ≤ 0.05 は一律満点、そこから 0.08 かけて
    # 0 へ)。h = 0.05 なら足箱の中心は 0.068 = ボール中心 0.11 の 42 mm 下で、
    # 実機側に 1-2 cm のオフセット (芝の沈み・ボール半径・立ち高さ) があっても
    # 中心より下に当たる余裕が残る。
    #
    # **これが実機の「巻き込んで転ぶ」への本命の対策。** 接触点がボール中心より上を
    # 通ると、ボールを上から押さえる形か、空振りして足がボールの上を越える形になる。
    # run 2026-08-22_11-56-42 の収束値 0.087 では足箱の中心 0.105 = ボール中心の
    # 5 mm 下しかなく、余裕がゼロだった。
    #
    # 2026-08-23 の初版は同じ関数を **天井** として使っていた (h_cap = 0.09)。
    # 関数の形は変えていない — 変えたのは h_target を 0.09 → 0.05 に下げたことだけ。
    # 0.09 は収束値 0.087 のすぐ上に座っていて f = 1 のまま何も止めていなかった
    # (:data:`_FOOT_LOW_H_TARGET` のコメント)。
    #
    # **kick_contact_height (h_sat 0.03 まで下げ続ける圧) は入れない。** walk_lob
    # 2026-08-18 で反証済み (sole_height は下がったが apex 0.340 → 0.234)。
    # この項は h_target = 0.05 で頭打ちになる (それ以下は 1 円も増えない) ので、
    # 「立ち位置を詰めてスイングを削る」ところまでは押さない。
    #
    # フェードインの窓は他のキック報酬と同じ 0 → 500。
    cfg.rewards.kick_foot_ceiling = RewTerm(
        func=mdp.kick_foot_ceiling,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _SIGMA_DIRECTION,
            "h_cap": _FOOT_LOW_H_TARGET,
            "h_span": _FOOT_LOW_H_SPAN,
        },
    )
    cfg.curriculum.kick_foot_ceiling_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_foot_ceiling",
            "start_weight": 0.0,
            "end_weight": _FOOT_LOW_WEIGHT * _KICK_W_SCALE,
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
    # 上で追加した 4 項 (kick_inside_contact / inside_foot_orient /
    # kick_foot_ceiling / ball_avoidance) にも r_max / orbit_beta /
    # overshoot_margin / lateral_band が配られる必要がある。
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
    * ``Metrics/kick_direction/sole_height_at_kick``: 蹴った瞬間の足裏高さ [m]。
      ``kick_foot_ceiling`` (h_target 0.05) が直接引っ張っている値で、
      **2026-08-24 以降の本命の判定材料**。基準は run 2026-08-22_11-56-42 の
      収束値 **0.087** (足箱の中心がボール中心の 5 mm 下 = 余裕ゼロ) で、そこから
      0.05 側へ動くかを見る。下りてこないなら :data:`_FOOT_LOW_WEIGHT` を
      3.0 → 6.0 へ上げる。
    * ``Metrics/kick_direction/kick_vel_ratio``: 威力とのトレードの監視。基準 0.887。
      この比は **水平成分**なので、``use_3d_speed=True`` にした後は下から当てた
      仰角ぶん下がって見えるのが正常 (20° で cos20° = 0.94 倍)。それ以上に落ちて
      いるなら、低く当てるために立ち位置を詰めてスイングを削っている。
    * ``Metrics/kick_direction/plant_lon`` / ``plant_lat`` / ``plant_yaw_dot``:
      軸足の前後位置 [m] / 横位置 [m] / 向き。**すべて観察用 (報酬からは外した)**。
      2026-08-24 に「軸足は物理的に無理な場所になければよく、報酬で誘導しない」へ
      方針転換したので、これらが目標へ寄ったかどうかは採否の判断材料にしない
      (モジュール docstring の経緯の節)。基準値は −0.107 / 0.30 / 0.9 級で、
      当たり方が変わったときに軸足がどう動いたかを後から読むための記録。
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
        # ランダム遅延 0-6 ステップ + 30Hz サンプル&ホールド + フレーム同期ジッタ)。
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
# カリキュラムの終値固定 — 共有モジュールへ切り出した
#
# ``_pin_curricula_at_end`` とその補助 (``_reward_term`` / ``_pin_expansion_gate`` /
# ``_UNPINNED_CURRICULUM_FUNCS``) はこのモジュールが初出だったが、「収束済み
# checkpoint から fine-tune 段を積むときは全カリキュラムを終値へ固定してから積む」
# という手順に inside 固有の要素は 1 つも無い (判定は ``func`` の identity、値は全て
# term 自身の params から読む)。2 つ目の利用者
# (:class:`~..walk_lob_plant.walk_lob_plant_env_cfg.K1WalkLobPlantRoughEnvCfg`) が
# 現れたので :mod:`..walk_kick.curriculum_pin` へ移した。**中身は 1 ビットも
# 変えていない。**
#
# 旧名で別名 import しているのは、このモジュールの docstring とコメントに散っている
# ``:func:`_pin_curricula_at_end``` 参照をそのまま生かすため (と、差分を「import 行
# だけ」に留めるため)。補助 3 つはこのモジュールから直接呼んでいないので import
# しない (必要になったら :mod:`..walk_kick.curriculum_pin` から公開名で取ること)。
# --------------------------------------------------------------------------- #
from ..walk_kick.curriculum_pin import pin_curricula_at_end as _pin_curricula_at_end


@configclass
class K1WalkInsideKickDualEnvCfg(K1WalkInsideKickCleanEnvCfg):
    """stage 2 (平坦): actor に 100 フレームの観測履歴を与え、観測を fewa 方式に揃える。

    **継承元は :class:`K1WalkInsideKickCleanEnvCfg`** (本命の
    :class:`K1WalkInsideKickEnvCfg` ではない)。両者の差は
    :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` を呼ぶかどうかだけで、
    レシピも観測の次元・並びも同一。この段はボール観測を fewa 方式へ張り替えるので、
    **ガウス系パイプラインが載っていない方から始める** 必要がある (理由は下の
    「観測パイプラインを fewa 方式へ」節)。

    stage 1 (:class:`K1WalkInsideKickEnvCfg`) との差:

    1. カリキュラムを全て終値に固定する (:func:`_pin_curricula_at_end`)。
    2. policy 観測グループを 100 フレームの履歴にする
       (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`)。
    3. **fewa の 3 点セット** (2026-08-24 追加): 実機で明確に滑らかだった
       fewa/47b8863 の歩容側設定をそのまま持ち込む。

       * 着地 shaping 3 項 (feet_landing_impact / feet_landing_vel /
         feet_heel_strike) を無効化
         (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.disable_landing_shaping`)。
         接地イベント瞬間だけのスパース項で、蹴りのスイングを硬くする側にしか
         働かないことが 47b8863 のコミット記録で示されている。
       * ``feet_phase`` 2.0 → 0.8
         (:func:`~..walk_kick_dual.walk_kick_dual_env_cfg.rebalance_gait_vs_kick`)。
         位相報酬で歩容を縛る力を fewa と同じまで緩める。
       * 地形を完全平面 → NOISY_FLAT (random_rough 0.7 / plane 0.3、±1-4 cm、
         :data:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg._FEWA_NOISY_FLAT_TERRAIN_CFG`)。
         平面だけで学習した足運びは「地面が完璧」前提で硬くなる。blind
         (height_scanner 無し) なので観測の次元は変わらない。

       3 つとも fewa の実機で滑らかさとして実証済みの組み合わせなので、
       個別の ablation はしない。リスク (着地 shaping を外すと kick_rate が
       停滞した系列もある) は fewa の実機実績で引き受ける。

    4. **観測パイプラインを丸ごと fewa 方式へ** (2026-08-24 追加、下の節)。

    観測パイプラインを fewa 方式へ (2026-08-24)
    -------------------------------------------
    ボール観測を、継承元のガウス系パイプライン
    (:func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs`: ランダム遅延
    0-6 step + 30 Hz サンプル&ホールド + フレーム同期ジッタ σ 0.067、**位置スロット
    だけ**) から、fewa の
    :func:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.enable_obs_delay`
    (連続遅延 + 一様ノイズ、位置と速度の両方、加えて IMU / エンコーダにも遅延) へ
    **束ごと** 張り替える。

    これは 2026-08-24 以前にこの docstring が書いていた
    「``enable_obs_delay`` は入れない」の **明示的な撤回** である。撤回にあたって、
    相反する 2 つの事実を両方とも記録しておく:

    * **実機の知覚は本当に 30 Hz の階段である。** カメラが 30 Hz なら、その間の
      フレームでボール位置は動かない。ガウス系パイプラインが模していたこの構造は
      実機に実在するもので、「間違っていたから捨てた」のではない。
    * **それでも fewa の束の方を採る。** 実機で動作確認できているキックは
      fewa/47b8863 の系統 **だけ** で、その束のどこが効いているのかは分かっていない
      (歩容 3 点なのか、観測の作り方なのか、その組み合わせなのか)。帰属が不明な
      以上、部分的に真似て「良いところ取り」を狙うより、**丸ごと同じにする** 方が
      実機で動く確率が高い。階段構造を戻すのは、fewa の束のまま実機で動くことを
      確認した後に、単独の変更として足せばよい。

    張り替えで直る不整合
    ....................
    継承元のガウス系パイプラインは **``prev_ball_pos`` だけ** を遅延させ、
    ``ball_vel`` は生のまま (遅延 0) にしていた。同じカメラフレームから出る 2 つの量の
    レイテンシが違うのは実機ではあり得ず、policy は「遅れた位置」と「今の速度」を
    突き合わせて動きを先読みできてしまう。fewa の
    ``enable_obs_delay`` は位置と速度を同じ "vision" group に入れて **同じ乱数** を
    引かせ、``ball_vel`` には ``base_delay_s`` = 0.02 s を足して
    ``prev_ball_pos`` の設計上の 1 ステップと実効遅延を揃える (単一カメラの整合性)。

    実効的な遅延とノイズ (この段の最終状態)::

        prev_ball_pos    0.02 + [0, 0.06] s (vision)   一様 ±0.07 m
        ball_vel         0.02 + [0, 0.06] s (vision)   一様 ±0.5 m/s
        projected_gravity      [0, 0.02] s (imu)       継承のまま ±0.05
        base_ang_vel           [0, 0.02] s (imu)       継承のまま ±0.2
        joint_pos              [0, 0.02] s (encoder)   継承のまま ±0.03
        joint_vel              [0, 0.02] s (encoder)   継承のまま ±1.5

    IMU / エンコーダの遅延はこのタスクにこれまで **無かった** もの (fewa には
    Stage 1 から入っている)。

    なぜ継承元を Clean へ張り替える必要があったか
    ..............................................
    fewa の ``enable_obs_delay`` は ObsTerm の ``func`` と ``params`` を **無条件で**
    差し替える (walk_kick_dual 版にある「パイプライン付きの位置スロットは飛ばす」
    ガードを持たない。あちらは移植元 47b8863 に無かった安全弁で、fewa 側は逐語コピーを
    優先して入れていない)。本命の :class:`K1WalkInsideKickEnvCfg` を継承したまま呼ぶと、
    ``_apply_noisy_ball_obs`` が入れた ``delay_step_range`` / ``camera_hz`` /
    ``jitter_std`` / ``jitter_clip`` が ``params`` に残ったまま
    ``delayed_prev_ball_pos_b`` へ渡り、未知のキーワード引数で落ちる。
    ガードのある版を使って「飛ばす」のは今回の趣旨 (ボール観測こそ fewa に合わせる) と
    正反対なので、**ガウス系パイプラインを最初から載せない**
    :class:`K1WalkInsideKickCleanEnvCfg` を継承元に選んだ。

    checkpoint の引き継ぎ
    ....................
    ``enable_obs_delay`` は既存の ObsTerm の ``func`` / ``params`` / ``noise`` を
    差し替えるだけで **項を作り直さない** ので、policy 55 次元 / critic 61 次元と
    その並びは変わらない。stage 1 からの ``--load_pretrained``
    (+ ``--warm_start_from_single_frame``) はこれまでどおり繋がる。

    ``__post_init__`` の順序に意味がある::

        super()                 … インサイドのレシピ (ガウス系パイプラインは無し)
        _pin_curricula_at_end() … 報酬・イベント・コマンドが全部揃った後に固定する
        fewa の 3 点セット      … 報酬 weight と地形。pin の後 (curriculum に上書き
                                  されないことが保証されてから)
        enable_obs_delay()      … 観測項の func/params を差し替える。履歴化の前
        enable_obs_history()    … **最後**。観測グループの構成が固まってから履歴化する

    ``enable_obs_history`` を最後に置くのは walk_lob_rough の Stage 3
    (:class:`~..walk_lob_rough.walk_lob_rough_env_cfg.K1WalkLobHistEnvCfg`) と同じ流儀。
    ``history_length`` は ObservationGroup 全体に掛かるフラグなので、後から観測項を
    足しても壊れはしないが、「グループの形を変える操作は最後」を守っておけば
    順序依存を考えなくて済む。

    dual 系から **持ち込まないもの** (全て意図的)
    ---------------------------------------------
    :mod:`..walk_kick_dual` は履歴のほかに 4 つの変更を畳み込んでいる。初版は
    「観測履歴だけを変えて効果を帰属させる」ため 1 つも入れなかったが、2026-08-24 に
    2 つは方針転換で入れた (下の 2 つの「撤回」)。いま入れていないのは残り 2 つ:

    * ``K1WalkKickBothFeetObservationsCfg`` (観測スロット 3 を左足裏 → ボール 3D 位置、
      critic 58 次元) — 入れると stage 1 の checkpoint が **意味の上で** 繋がらなくなる
      (55 次元なので ``--load_pretrained`` は形の上では通ってしまうぶん、たちが悪い)。
      基底の policy 55 次元 / critic 61 次元をそのまま使う。
    * ``_apply_phase_offset`` (歩行位相の初期オフセット {0, π}) — 両足で蹴れるように
      するための変更。**このタスクは右足専用** (``kick_inside_contact`` が右足ゲート付き)
      なので、蹴り足を割る意味が無い。
    * (2026-08-24 撤回) ``disable_landing_shaping`` / ``rebalance_gait_vs_kick`` は
      当初「履歴の効果に帰属させる」ため持ち込まない方針だったが、**fewa の実機が
      明確に滑らかだった**ため方針転換して入れた (下の「fewa の 3 点セット」)。
      帰属の純度より実機の歩容を優先する。
    * (2026-08-24 撤回) ``enable_obs_delay`` — 当初はこう書いていた:
      「ボール観測にはこのタスク独自の認識パイプライン (``_apply_noisy_ball_obs``:
      ランダム遅延 0-6 step + 30 Hz サンプル&ホールド + フレーム同期ジッタ) が既に
      載っており、``enable_obs_delay`` はパイプライン付きの位置スロットを二重掛け
      防止のため飛ばす作りなので、掛けても実質 IMU / エンコーダにしか効かない。
      内界センサの遅延 DR は『履歴の効果を見る』この段の目的と別件なので、入れるなら
      stage 3 以降に単独で足す。」

      **この判断は撤回した。** 実機の知覚が 30 Hz の階段であること自体は今も正しいが、
      実機で動くと確認できているのは fewa の束 **だけ** で、束のどこが効いているのかが
      分からない。帰属が不明なまま良いところ取りをするより丸ごと合わせる方が確度が
      高いので、ボール観測ごと fewa 方式へ張り替えた (fewa 版の ``enable_obs_delay`` は
      「飛ばす」ガードを持たないので、継承元も Clean へ張り替えてある)。
      経緯と実効値は上の「観測パイプラインを fewa 方式へ (2026-08-24)」節。
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

        # -- 2. fewa の歩容 3 点セット (docstring の差分 3) --------------- #
        #
        # 順序: pin の後 (feet_phase の weight を curriculum が上書きしないことは
        # pin 済みなので保証される)、履歴化の前。
        disable_landing_shaping(self)
        rebalance_gait_vs_kick(self)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _FEWA_NOISY_FLAT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        # generator 地形では env origin の z が「その patch の代表高さ」なので、
        # ボールを凹凸 (±4 cm) にめり込ませないよう上から落とす。rough 段の
        # :func:`~..walk_kick.walk_kick_env_cfg._apply_rough_terrain` と同じ値。
        self.events.reset_ball.params["spawn_clearance"] = 0.05

        # -- 3. 観測パイプラインを fewa 方式へ (docstring の差分 4) -------- #
        #
        # ボール 2 項 (prev_ball_pos / ball_vel) を同じ "vision" group の連続遅延
        # 0.02-0.08 s + 一様ノイズ ±0.07 m / ±0.5 m/s に、IMU / エンコーダ 4 項を
        # 0-0.02 s の遅延にする。呼び方は fewa の Stage 4 と同じ
        # (:class:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.K1WalkLongPassFewaEnvCfg`)。
        #
        # 継承元が Clean = ガウス系パイプラインが載っていないので、ボール位置スロットも
        # 素の prev_ball_pos_b のまま差し替えられる (本命の K1WalkInsideKickEnvCfg を
        # 継承していると params にガウス側のキーが残って落ちる。docstring 参照)。
        # func と params の差し替えだけなので観測の次元も並びも変わらない。
        fewa_enable_obs_delay(self, _FEWA_OBS_DELAY_MAX_S)

        # -- 4. 観測履歴 ------------------------------------------------- #
        #
        # 必ず最後。policy グループの構成が固まってから (N, H, 55) に変える。
        enable_obs_history(self)


@configclass
class K1WalkInsideKickDualEnvCfg_PLAY(K1WalkInsideKickDualEnvCfg):
    """stage 2 の PLAY。

    カリキュラムは親で全て ``None`` にしてあるので、PLAY でも
    ``common_step_counter`` 0 から巻き戻る心配は無い (項が 1 つも無い)。

    :class:`K1WalkInsideKickEnvCfg_PLAY` と違い
    :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter` は **呼ばない**
    (2026-08-24)。この段のボール観測は fewa 方式に張り替わっていて、あちらが書き込む
    ``jitter_std`` を受け取る関数がもう居ない (未知のキーワード引数で落ちる)。

    fewa 方式では **ノイズは全て ObsTerm の ``noise`` (Unoise) 側にある** ので、
    ``enable_corruption = False`` だけで位置 ±0.07 m / 速度 ±0.5 m/s とも切れる。
    遅延 (0.02-0.08 s / 0-0.02 s) は関数側にあるので残る = 観測パイプラインの構造は
    PLAY でも見える。これは fewa の PLAY
    (:class:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.K1WalkLongPassFewaEnvCfg_PLAY`)
    と同じ扱い。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


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

    観測は stage 2 のまま (2026-08-24)
    ---------------------------------
    ボール / IMU / エンコーダの遅延と一様ノイズは stage 2 の
    :func:`~..walk_long_pass_fewa.walk_long_pass_fewa_env_cfg.enable_obs_delay` が
    ``super()`` の中で 1 回だけ掛けている。この段では **触らない** ので二重掛けは
    起きない (仮に 2 回呼んでも ``func`` / ``params`` の上書きなので冪等だが、
    呼ばないのが意図)。地形の DR を広げる段であって知覚を変える段ではない。

    ``__post_init__`` の順序について
    --------------------------------
    ``super()`` が既に :func:`~..walk_kick_dual.walk_kick_dual_env_cfg.enable_obs_history`
    を掛け終わっているが、**問題ない**。:func:`_apply_rough_terrain` が触るのは
    ``scene.terrain`` の 3 属性と ``events.reset_ball.params["spawn_clearance"]`` だけで、
    観測グループにも報酬にもコマンドにも一切触らないため (あちらの docstring と実装を
    確認済み)。stage 2 が入れた NOISY_FLAT 地形と ``spawn_clearance`` = 0.05 は
    ここで凹凸地形と同じ値に上書きされる (どちらも generator、clearance も同値)。
    同じ理由で :func:`~..walk_weak_kick_orbit.orbit_mods.apply_ball_param_dr`
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

    stage 2 の PLAY と同じく :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter`
    は **呼ばない** (2026-08-24)。理由は :class:`K1WalkInsideKickDualEnvCfg_PLAY` の
    docstring。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _apply_play_viewer(self)
