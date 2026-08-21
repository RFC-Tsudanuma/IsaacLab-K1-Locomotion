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
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _disable_ball_obs_jitter,
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
    _SIGMA_DIRECTION,
    K1WalkKickEnvCfg,
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
# r_stance = 0.10:
#   P_kick (理想キック立ち位置) をボール後方どれだけの点に置くか [m]。
#   既定 0.25 はつま先で突く前提の距離。初版は 0.20 で学習し、run 2026-08-21_05-00-22
#   はインサイドの型 (dot≈0 / ball_side 0.145 / vel_ratio 0.91) まで到達したが、
#   実機で「振りが手前すぎてボールを巻き込んで転ぶ」事故がそこそこ出た。
#
#   軸足の前後位置 (plant_lon) は指令した立ち位置にほぼ張り付く実績がある:
#     r_stance 0.25 (middle) -> plant_lon -0.39〜-0.42
#     r_stance 0.20 (inside) -> plant_lon -0.23
#   そこで報酬を足さず、指令側の r_stance を 0.10 へ詰めて軸足をボールの真横へ寄せる
#   (2026-08-21 実機フィードバック)。軸足がボール横にあれば、認識誤差で手前に当たっても
#   ボールは体の横を抜けていき、踏む位置に残らない。
#
#   NOTE: 0.10 はボール半径 0.11 より小さい = 指令の立ち位置がボールの後縁より内側。
#         接近中に体がボールへ触れる事故 (ball_touch_count の増加) と、振りかぶりが
#         窮屈になることによる kick_vel_ratio の低下を必ず監視すること。崩れるなら
#         0.12〜0.15 へ戻して締め直す。
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
    "r_stance": 0.10,
    "overshoot_margin": 0.30,
    "style_halfwidth": 0.698,
}

# --------------------------------------------------------------------------- #
# kick_inside_contact の重み
#
# direction (6.0) / scaled (4.0) / strong (3.0) と同じ土俵で、scaled と同格の 4.0。
# middle の kick_plant_foot が 2.0 に抑えられていたのは、あれが「目的そのものでは
# なく蹴り方の指定」だったため。こちらは **当たり所がこのタスクの目的そのもの** なので、
# その上限を外して同格に置く。
#
# 項1-3 と同じく post-latch に dense で払われるので、猶予窓 (2.0 秒) ぶんの割り戻し
# (_KICK_W_SCALE) を掛けること。フェードインの窓は基底のキック報酬と同じ 0 → 500 で、
# 発見期には既に満額で乗っている状態にする (型が決まってから足すと、既に決まった
# トーキックの位置で f_perp が小さく、勾配が細いところから始めることになる)。
# --------------------------------------------------------------------------- #
_INSIDE_CONTACT_WEIGHT = 4.0

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
    * ``Metrics/kick_direction/plant_lon`` / ``plant_lat``: 報酬からは外してある
      ので、インサイドが立ち上がったときに軸足がどこへ移ったかの **観察用**。
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
