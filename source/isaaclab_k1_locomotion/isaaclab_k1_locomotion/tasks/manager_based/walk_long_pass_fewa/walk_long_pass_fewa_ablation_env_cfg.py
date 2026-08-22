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
======================  ==========================================  =====================================
"""

from isaaclab.utils import configclass

from .walk_long_pass_fewa_env_cfg import K1WalkLongPassFewaEnvCfg, K1WalkLongPassFewaEnvCfg_PLAY

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
