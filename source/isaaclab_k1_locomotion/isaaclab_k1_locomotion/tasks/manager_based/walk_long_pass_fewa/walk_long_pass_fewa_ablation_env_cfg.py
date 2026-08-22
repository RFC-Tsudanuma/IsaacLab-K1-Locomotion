# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_fewa の Stage 4 ablation。

:class:`~.walk_long_pass_fewa_env_cfg.K1WalkLongPassFewaEnvCfg` (実機で動いた構成)
を継承し、**1 変種につき 1 箇所だけ**変える。一晩で並列に回して、翌日に良かったものを
実機へ載せるための比較用なので、基底 (fewa) 側は絶対に触らない。

狙いは 2 つ:

* **もっと強く蹴る** … :class:`K1WalkLongPassFewaBand6EnvCfg`
* **蹴っている最中・蹴った後の跳ねを減らす** … :class:`K1WalkLongPassFewaCalmEnvCfg`

「跳ね」は実機で観測された症状 (蹴る瞬間に軸足が浮く / 蹴った直後にぴょんと跳ねる)。
sim のログには直接の指標が無いので、``Episode_Reward/lin_vel_z_l2`` と
``Episode_Reward/feet_air_time`` の変化、および play 動画で判断すること。

観測・行動空間はどの変種でも 55 次元 × :data:`~.walk_long_pass_fewa_env_cfg.
_OBS_HISTORY_LENGTH` フレームのまま **一切変えていない**ので、どの変種も
同じ Stage 4 checkpoint から ``--load_pretrained`` で始められるし、
学習後の checkpoint は基底と同じ ONNX 形状で書き出せる。

一覧:

======================  ==========================================  =====================================
クラス                    変更点                                       仮説
======================  ==========================================  =====================================
Band6                   帯の終点 (3.2, 5.0) → (3.2, 6.0)             実機は v≈6.0 で 20 m 出た実績があり、
                        σ_v 0.9 → 1.4 (帯幅に合わせて)                関節速度上限 (足先 ≈6.5 m/s) にも
                                                                    まだ余裕がある。上限を上げれば
                                                                    そのぶん強く蹴れる。
Calm                    lin_vel_z_l2 -0.8 → -2.0                    跳ねは「上下動」と「胴体のロール/
                        ang_vel_xy_l2 -0.32 → -1.0                  ピッチ振動」と「滞空を稼ぐ歩容」の
                        feet_air_time +0.2 → 0.0                    3 つが報われている結果。罰を強め、
                                                                    滞空への報酬を切れば収まる。
Band6Calm               Band6 + Calm                                強く蹴りつつ跳ねない、が両立するか。
                                                                    片方ずつと突き合わせて交互作用を見る。
Grounded                報酬項 kick_plant_grounded を追加            跳ねるのは「跳んで蹴っても報酬が
                        (weight 3.0 × _KICK_W_SCALE)                 変わらない」から。接地している蹴りに
                                                                    だけ上乗せで払えば跳ばなくなる。
                                                                    Calm と違って **罰ではない** ので、
                                                                    蹴り自体を減らす方向へは効かない。
======================  ==========================================  =====================================
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import _KICK_STATE_PARAMS, _KICK_W_SCALE, _SIGMA_DIRECTION
from .walk_long_pass_fewa_env_cfg import (
    _LONG_PASS_V_THRESH,
    K1WalkLongPassFewaEnvCfg,
    K1WalkLongPassFewaEnvCfg_PLAY,
    _apply_v_thresh,
)

# --------------------------------------------------------------------------- #
# Ablation A: 帯の終点を 6.0 m/s まで伸ばす
#
# 基底は (3.2, 5.0)。d = v²/2a (a ≈ 1.0 m/s²) で 5.0 → 12.5 m 相当。
# 実機では walk_kick_360 系のポリシーが v ≈ 6.0 の指令で 20 m を出しており、
# ハード壁 (関節速度上限から出る足先速度 ≈ 6.5 m/s、loop_shoot 実測) にも
# まだ 0.5 m/s の余裕がある。つまり 6.0 は「届かない指令」ではない。
#
# **開始点 (2.0, 3.0) とゲートは基底のまま。** ここを一段で張り替えると
# 「蹴らずに 15 秒歩き回る」に収束する (基底のモジュール docstring の失敗記録)。
# 帯は kick_rate_gated_speed_range が「蹴れている間だけ」進めるので、6.0 に届かず
# 途中で止まったら、その speed_max がこのスイングの実質的な上限という読み方になる。
#
# NOTE: 公称のランプ区間 (500 → 3000 iteration) は基底のまま。動かす幅が
#       3.0 → 5.0 (2.0 m/s) から 3.0 → 6.0 (3.0 m/s) へ 1.5 倍になるので、
#       1 iteration あたりの上がり方も 1.5 倍になる。ゲートが閉じて公称より
#       遅れることを見込んで ITER は 5000 以上を取ること。終了時に
#       Curriculum/kick_speed_range/alpha が 1.0 かを必ず確認する。
# --------------------------------------------------------------------------- #
_BAND6_SPEED_RANGE = (3.2, 6.0)

# --------------------------------------------------------------------------- #
# 帯幅に合わせた kick_velocity_scaled の速度シェイピング係数 [m/s]
#
# 47b8863 は帯幅 1.8 (3.2-5.0) に対して 0.9 を選んでいる。これは
# 「帯の中心から両端までがちょうど 1σ = 帯の両端が 2σ 離れる」という取り方で、
# 帯の端の指令でも勾配が届き、かつ指令の識別 (3.2 と 5.0 を区別して蹴り分ける)
# は保たれる、という釣り合いになっている。
#
# 帯幅が 2.8 (3.2-6.0) になるので、同じ取り方なら σ = 2.8 / 2 = 1.4。
# 0.9 のままだと帯の端が 3.1σ 離れて上限側の勾配が消え、
# 「蹴らない方が得」の収支逆転 (基底 docstring の失敗記録) が上限側だけで起きる。
# --------------------------------------------------------------------------- #
_BAND6_SIGMA_VELOCITY = 1.4


def apply_band6(cfg) -> None:
    """帯の終点を :data:`_BAND6_SPEED_RANGE` に伸ばし、σ_v を帯幅に合わせる。

    カリキュラム項がある学習用 cfg では **終点だけ** を差し替える (開始点と
    ゲートのしきい値は基底のまま)。PLAY 側は基底が ``kick_speed_range`` を
    None にして帯を終点で固定しているので、そちらは直接 target_speed_range を書く。
    """
    cfg.rewards.kick_velocity_scaled.params["sigma_velocity"] = _BAND6_SIGMA_VELOCITY

    term = getattr(cfg.curriculum, "kick_speed_range", None)
    if term is not None:
        term.params["end_range"] = _BAND6_SPEED_RANGE
    else:
        # PLAY: カリキュラムが外れていて帯は終点で固定されている。
        cfg.commands.kick_direction.target_speed_range = _BAND6_SPEED_RANGE


# --------------------------------------------------------------------------- #
# Ablation B: 跳ねを抑える (config だけ / 報酬の重みのみ)
#
# 実機の症状: 蹴る瞬間に軸足ごと浮く、蹴った直後にぴょんと跳ねる。
# 跳ねを「報われている行動」として見ると、この構成では 3 つの経路がある:
#
#   1. lin_vel_z_l2 (基底 -0.8) … 胴体の上下速度への罰。値は K1FlatEnvCfg で
#      歩行用に決めたもので、キックのような大きな上下動は想定していない。
#   2. ang_vel_xy_l2 (基底 -0.32) … 胴体のロール/ピッチ角速度への罰。跳ねは
#      たいてい前後の煽りを伴うので、上下だけ罰すると煽りへ逃げる。
#   3. feet_air_time (基底 +0.2) … **滞空時間を報酬にしている**。歩行では
#      すり足を防ぐための項だが、「跳ねて両足が浮く」時間もそのまま加点される。
#      跳ね対策としては符号が逆向きなので 0 にする (罰にはしない。罰にすると
#      今度はすり足へ倒れて歩容そのものが崩れる)。
#
# 1 と 2 の値は「跳ねが割に合わなくなる」ところまで上げる。2.5 倍と 3 倍は、
# キック 1 回の収益 (≈ +5) に対して跳ね 1 回のコストが無視できなくなる水準として
# 選んだ見当で、実測での裏付けはまだ無い (これがこの ablation の検証対象)。
#
# 着地 shaping 3 項 (feet_landing_impact / feet_landing_vel / feet_heel_strike) は
# **基底のまま無効のままにする**。「着地を丁寧にする」項なので跳ね対策として
# 復活させたくなるが、47b8863 でこれを入れたまま回した Stage 2 は
# kick_rate が 0.19-0.28 で 4000 iteration 完全に停滞した (蹴るほど接地が強く速く
# なるので、蹴らない方が得になる。基底の disable_landing_shaping のコメント参照)。
# 跳ねを直すために蹴りを失うのでは本末転倒なので、ここでは触らない。
#
# NOTE: 重みは基底 (K1FlatEnvCfg) 側の現在値を確認したうえで置き換えている。
#       基底が動いたらここも見直すこと。
# --------------------------------------------------------------------------- #
_CALM_LIN_VEL_Z_WEIGHT = -2.0     # 基底 -0.8 (locomotion/flat_env_cfg.py)
_CALM_ANG_VEL_XY_WEIGHT = -1.0    # 基底 -0.32 (locomotion/flat_env_cfg.py)
_CALM_FEET_AIR_TIME_WEIGHT = 0.0  # 基底 +0.2 (locomotion/flat_env_cfg.py)


def apply_calm(cfg) -> None:
    """跳ねに効く 3 項の weight を差し替える (項の増減はしない)。

    報酬項の weight を変えるだけなので観測にも行動にも影響せず、checkpoint は
    そのまま繋がる (actor / critic の形は変わらない)。

    項が存在しない構成でも落ちないよう getattr でガードする (基底の
    disable_landing_shaping と同じ扱い)。
    """
    for name, weight in (
        ("lin_vel_z_l2", _CALM_LIN_VEL_Z_WEIGHT),
        ("ang_vel_xy_l2", _CALM_ANG_VEL_XY_WEIGHT),
        ("feet_air_time", _CALM_FEET_AIR_TIME_WEIGHT),
    ):
        term = getattr(cfg.rewards, name, None)
        if term is not None:
            term.weight = weight


# --------------------------------------------------------------------------- #
# A: 帯 6.0
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFewaBand6EnvCfg(K1WalkLongPassFewaEnvCfg):
    """Stage 4 (帯 6.0)。基底との差は帯の終点と σ_v だけ。

    仮説: 実機は v ≈ 6.0 の指令で 20 m を出しており、関節速度上限 (足先 ≈ 6.5 m/s)
    にも余裕がある。帯の上限を 5.0 → 6.0 に伸ばせば、そのぶん強い蹴りが指令できる。

    見るところ: ``Curriculum/kick_speed_range/{speed_max,alpha}`` が 6.0 / 1.0 に
    届くか、その間 ``kick_rate`` が 0.8 を割らずに済むか。alpha が途中で止まったら、
    その speed_max がこのスイングの実質的な上限。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_band6(self)


@configclass
class K1WalkLongPassFewaBand6EnvCfg_PLAY(K1WalkLongPassFewaEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        apply_band6(self)


# --------------------------------------------------------------------------- #
# B: 跳ね抑制
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFewaCalmEnvCfg(K1WalkLongPassFewaEnvCfg):
    """Stage 4 (跳ね抑制)。基底との差は報酬 3 項の weight だけ。

    仮説: 実機で見えた「蹴りながら跳ぶ・蹴った後に跳ねる」は、上下動と胴体の
    煽りへの罰が歩行用の値のままで弱く、そのうえ滞空時間が加点されているため。
    罰を強めて滞空への加点を切れば収まる。

    見るところ: ``Episode_Reward/lin_vel_z_l2`` の絶対値と
    ``Episode_Reward/feet_air_time`` が下がるか、その代償に ``kick_rate`` と
    ``kick_vel_ratio`` がどれだけ落ちるか。落ち幅が大きいなら跳ねと威力が
    トレードオフになっているということで、重みを中間に取り直す。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_calm(self)


@configclass
class K1WalkLongPassFewaCalmEnvCfg_PLAY(K1WalkLongPassFewaEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        apply_calm(self)


# --------------------------------------------------------------------------- #
# C: A + B
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFewaBand6CalmEnvCfg(K1WalkLongPassFewaBand6EnvCfg):
    """Stage 4 (帯 6.0 + 跳ね抑制)。A と B を両方入れたもの。

    仮説: 強い蹴りと跳ねの少なさは両立する。両立しないなら A・B 単独と比べて
    どちらかの指標だけが崩れるはずで、その崩れ方から交互作用が読める。

    NOTE: A を継承したうえで B を足しているので、A と完全に同じ帯設定になる
          (定数を二重に持たない)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_calm(self)


@configclass
class K1WalkLongPassFewaBand6CalmEnvCfg_PLAY(K1WalkLongPassFewaBand6EnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        apply_calm(self)


# --------------------------------------------------------------------------- #
# Ablation D: 蹴る瞬間に軸足が接地していることを報酬にする (跳ね対策・実装あり)
#
# B (Calm) が「跳ねを罰する」なら、こちらは **接地している蹴りに上乗せで払う**。
# 罰と報酬では抜け道が違うので、どちらが効くかを並べて見たい:
#
#   * 罰 (Calm) は「跳ねないが蹴らない」でも満点になる。キック報酬との綱引きに
#     なるので、蹴り自体が減る方向へ倒れる危険がある。
#   * 上乗せ (Grounded) は r_direction への乗算なので、**蹴らなければ 1 円も出ない**。
#     「跳ねずに蹴る」だけが得になる。そのかわり跳ねること自体は罰されないので、
#     跳ぶ癖が強く残っていると勾配が育つまで時間がかかる。
#
# 測っている量 (:func:`~..walk_kick.mdp.kick_state._plant_contact_normalized`):
#   latch したステップの軸足の **法線接触力** を、片足立ちぶんの荷重 (0.5·m·g) で
#   正規化して [0, 1] にクランプしたもの。1 = 体重が軸足に乗っている、
#   0 = 軸足も浮いている = 跳びながら蹴っている。
#
# weight は 3.0 × _KICK_W_SCALE。_KICK_W_SCALE (= 0.6 / 2.0) は「仕様書の weight は
# キック窓 0.6 秒前提の配分なので、実際の窓 2.0 秒との比で割り戻す」係数で、
# walk_kick 系のキック報酬は全部これを掛けてある。素の 3.0 は項1 (kick_direction, 6.0)
# の半分で、方向が合った蹴り 1 回の収益を 1.5 倍程度に押し上げる水準。
#
# **フェードインは付けない。** Stage 4 は既に蹴れるポリシーからの fine-tune なので、
# 立ち上げ期間はそのまま「蹴らない方が得」の期間になる (基底の
# _freeze_fade_in_curricula の docstring)。カリキュラム項を作らず終値を直接入れる
# ことで、1 iteration 目から満額になる (基底が既存のランプを潰しているのと同じ状態)。
#
# NOTE: この項は kick_state を呼ぶので v_thresh の配布対象。基底の __post_init__ が
#       配り終えた **後** に足すことになるので、足した直後にもう一度
#       _apply_v_thresh を呼ぶ (基底の -- 4 で extra_ball_touch を足してから -- 5 で
#       配っているのと同じ順序を、継承の外側で作り直している)。
# NOTE: log_contact_geometry を True にして Metrics/kick_direction/plant_contact を
#       出す。**学習には一切影響しない** (メトリクスが 4 つ増えるだけ) が、
#       この項が効いているかはこの値が上がるかどうかでしか判定できない。
# --------------------------------------------------------------------------- #
_GROUNDED_WEIGHT = 3.0 * _KICK_W_SCALE


def apply_grounded(cfg) -> None:
    """軸足の接地を測る報酬項を足し、v_thresh を配り直す。

    PLAY からも呼べる (報酬項の増減は観測にも行動にも影響しないので、
    checkpoint の引き継ぎにも PLAY の見え方にも影響しない)。
    """
    cfg.rewards.kick_plant_grounded = RewTerm(
        func=mdp.kick_plant_grounded,
        weight=_GROUNDED_WEIGHT,
        # sigma_direction は同じタスクの他のキック項と必ず同じ値にすること
        # (r_direction を共有しているので、違う値を渡すと項ごとに別の方向精度で
        #  採点することになる)。
        params={**_KICK_STATE_PARAMS, "sigma_direction": _SIGMA_DIRECTION},
    )
    # 項を足した後に配り直す (基底の -- 5 は既に走り終わっている)。
    _apply_v_thresh(cfg, _LONG_PASS_V_THRESH)

    # Metrics/kick_direction/plant_contact を出す (学習には影響しない)。
    cfg.commands.kick_direction.log_contact_geometry = True


@configclass
class K1WalkLongPassFewaGroundedEnvCfg(K1WalkLongPassFewaEnvCfg):
    """Stage 4 (軸足接地)。基底との差は報酬項が 1 つ増えることだけ。

    仮説: 実機で見えた「蹴りながら跳ぶ」は、跳んで蹴っても報酬が変わらないから
    起きている。接地している蹴りにだけ上乗せで払えば、跳ばない蹴りへ寄る。

    見るところ: ``Metrics/kick_direction/plant_contact`` (kick_rate で割り戻す) が
    上がるか。上がらないまま ``Episode_Reward/kick_plant_grounded`` も 0 のままなら、
    contact_forces センサの body index が解決できていない可能性がある
    (起動ログの ``[WARN] kick_state:`` を確認すること)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_grounded(self)


@configclass
class K1WalkLongPassFewaGroundedEnvCfg_PLAY(K1WalkLongPassFewaEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        apply_grounded(self)


# --------------------------------------------------------------------------- #
# 帯 6.0 を前提にした組み合わせ (本命の候補群)
#
# 目的は「強く **かつ** 跳ねない」なので、単独変種 (calm / grounded) は目的の
# 半分しか狙っていない。実機に載せる候補は **全部に帯 6.0 を入れた上で**、
# 跳ね対策の機構 (罰 = Calm / 上乗せ = Grounded / 両方) だけを振る。
# --------------------------------------------------------------------------- #
@configclass
class K1WalkLongPassFewaBand6GroundedEnvCfg(K1WalkLongPassFewaBand6EnvCfg):
    """Stage 4 (帯 6.0 + 軸足接地の上乗せ)。跳ね対策を罰ではなく報酬側で持つ版。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_grounded(self)


@configclass
class K1WalkLongPassFewaBand6GroundedEnvCfg_PLAY(K1WalkLongPassFewaBand6EnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        apply_grounded(self)


@configclass
class K1WalkLongPassFewaBand6CalmGroundedEnvCfg(K1WalkLongPassFewaBand6CalmEnvCfg):
    """Stage 4 (帯 6.0 + 跳ね罰 + 軸足接地の上乗せ)。跳ね対策を両方の機構で持つ版。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_grounded(self)


@configclass
class K1WalkLongPassFewaBand6CalmGroundedEnvCfg_PLAY(K1WalkLongPassFewaBand6CalmEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        apply_grounded(self)
