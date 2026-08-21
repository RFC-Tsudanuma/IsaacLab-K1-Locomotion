# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークミドルキック環境（5-10 m 飛ぶキックを指令どおりに出す / weak レシピの帯違い）。

戦略側の要求は「5-10 m 飛ぶキックを、指令した強さで出せること」。
:mod:`..walk_weak_kick` が「指令どおりの強さのキック」を実際に成立させたレシピなので、
**そのレシピをそのまま使い、指令帯だけを高速側へ張り替える**のがこのタスク。
一言でいえば **middle = weak + 帯の張り替え + 軸足配置**であり、weak との差は
指令帯・σ_velocity の終点 (0.5)・``kick_plant_foot`` の 3 点だけ。それ以外の
報酬構造・カリキュラムの窓・DR は weak と 1 バイトも変えていない。

必要な球速の較正（実機 1 点 + 転がり減速）
------------------------------------------
実機で指令 2.0 m/s のキックが 2 m 転がった。d = v²/2a から a ≈ 1.0 m/s²。
この a で戦略要求の飛距離を球速に逆算すると::

    v = 3.2 m/s -> d = 3.2² / 2.0 = 5.1 m   (要求下限 5 m)
    v = 4.5 m/s -> d = 4.5² / 2.0 = 10.1 m  (要求上限 10 m)

よって指令帯は **(3.2, 4.5) m/s**。基底 walk_kick の帯 (0.25, 2.0) をこの帯へ
丸ごと置き換える。

なぜ帯カリキュラムを入れないのか
--------------------------------
このタスクは weak と同じく **walk phase の checkpoint から stage 2 を作り直す**。
このやり方では、帯を低いところから段階的に動かす理由がない。

キック発見期 (0 → 1500 iteration) に学習信号を出しているのは次の 3 つで、
どれも指令帯に依存しない:

* latch (``v_thresh`` 0.8 固定) — キックが起きたことの判定。帯とは無関係。
* ``kick_direction`` (weight 最大) — 方向のみを見る。walk_kick 系では速度ゲートを
  掛けていないので帯に依存しない。
* ``kick_velocity_strong`` — r_dir × v_ball の青天井項。**速いほど得**なだけで、
  やはり帯を見ていない。weak のレシピ (b) により 500-1500 iteration は満額。

帯が効くのは ``kick_velocity_scaled`` の採点位置だけである。そして学習初期の
下手なキックは、帯が (0.25, 2.0) にあっても (3.2, 4.5) にあっても scaled では
どのみちほぼ 0 点にしかならない。実際の学習経路は

    strong が実蹴り速度を帯の上 (walk_kick_360 の実測 v ≈ 6.0 m/s) まで押し上げる
    → 1500 以降、strong のフェードアウト・σ アニール・overshoot 罰が
      その 6.0 を帯 (3.2, 4.5) の中へ絞り込む

であって、**帯は「上から降りてくる先」として最初から終点に置いてある方が素直**。
上げていく形の帯カリキュラムはむしろ、strong が既に押し上げた速度より下に帯を
置き続けることになり、絞り込みの開始を遅らせるだけになる。よって帯は固定。

latch 閾値を帯に連動させて上げてはいけない
------------------------------------------
帯を上げたのだから latch の閾値 (``v_thresh`` = 0.8 m/s) も上げたくなるが、
**これは絶対にやらないこと**。latch は項1-3 と ``kick_finished`` の全てを束ねる
ゲートなので、閾値をポリシーの実力より上に置いた瞬間に**キック報酬が全滅**する。
報酬が 0 になれば ``kick_finished`` が残りの歩行報酬を捨てるコストだけが残り、
ポリシーは「蹴らずに time_out まで歩く」へ落ちる。
0.8 固定なら帯がどこにあっても latch は発火し続け、σ の太い
``kick_velocity_scaled`` が指令方向への勾配を出し続ける。

weak のレシピ (a) (``v_thresh_eff = clamp(0.6·v_target, 0.2, 0.8)``) は
:func:`~..walk_weak_kick.walk_weak_kick_env_cfg._apply_weak_kick_recipe` 経由で
そのまま入るが、この帯では 0.6·v_target = 1.92-2.70 が常に clamp 上限 0.8 に
張り付くので、**実効閾値は基底と同じ 0.8 固定 = 実質 no-op** になる。
レシピを共有する副産物として無害に残っているだけで、意図的に効かせてはいない
(帯の下端が 0.8/0.6 = 1.34 m/s を下回るときにだけ意味を持つ機構)。

σ_velocity の終点だけ weak と変える (0.35 -> 0.5)
-------------------------------------------------
weak の 0.35 は帯 (0.25, 2.0) 用の値なのでそのままは使えない。0.5 にする理由は 2 つ:

* **識別性**: 終点帯の幅は 1.3 (3.2-4.5)。σ=0.5 なら下端 3.2 と上端 4.5 が 2.6σ
  離れており、指令の違いが報酬の違いとして十分に出る (指令 3.2 に v=4.5 を出すと
  f_vel = exp(−(1.3/0.5)²) ≈ 0.001)。
* **物理ノイズ耐性**: 高速域はボール物性 DR (摩擦・反発・質量) による到達速度の
  絶対ばらつきが大きい。同じスイングでも v_ball が ±0.2-0.3 振れるので、
  σ=0.35 まで絞ると自分の制御ではどうにもならない物理ノイズで報酬が半減し、
  「指令どおり蹴る」の勾配より「たまたま当たりの物性を引く」分散の方が大きくなる。

これ以外 (overshoot 罰の margin 0.2 / sat 1.0 / weight −2.0×_KICK_W_SCALE、
strong の折れ線、σ アニールの窓、ボール物性 DR) は weak と完全に同一。

軸足配置 (``kick_plant_foot``) を stage 2 から入れる
---------------------------------------------------
weak には無い項をひとつだけ足す。狙いは **「歩幅の中で通り抜けざまに弾く」蹴り方を
「軸足を置いてから振る」蹴り方へ寄せる**こと。

この項を入れる前の実測 (``k1_walk_middle_kick_360`` の 2026-08-15 run, 最終 iteration。
``kick_rate`` が 0.996 なのでメトリクスの値はほぼそのまま実効値)::

    plant_lat  0.207   <- 目標 0.19 に対して既にほぼ理想。横は直すところが無い
    plant_lon -0.422   <- 軸足がボールの 42cm 後方。ここだけが大きく外れている

蹴り足の足首は接触時にボール中心の 11-15cm 手前なので、−0.42 は「接触の瞬間に両足が
28cm 開いている」= 歩行のストライドの途中でボールを弾いている状態を意味する。
下流の指標 (kick_rate 0.996 / ball_touch_count 1.007 / 方向誤差 5.0° / 追従率 0.92) は
この蹴り方でも健全なので **これ自体が現状の失点源ではない**。それでも入れる理由は、
止まって構えてから振る方がボール位置の観測誤差に対して鈍く、stage 4 (観測ノイズ+遅延)
と実機での蹴り損ねに効くと見込めるため。

**stage 3 以降からの後入れにしないこと。** −0.42 に収束したポリシーに対して
σ_lon=0.10 の Gaussian は f_lon = exp(−0.39²/2·0.10²) ≈ 5e-4 で、値も勾配も死んでいる。
「まだキックの型が決まっていない stage 2 の発見期に混ぜておく」ことがこの項の前提で、
だからこそ :func:`_apply_middle_kick_recipe` (= stage 2/3/4 共通) に置いてある。

σ_lon だけは lob (0.10 固定) と変え、``kick_velocity_scaled`` の σ と同じ流儀で
**1.0 ではなく太い 0.30 から始めて 0.10 へ絞る** (窓は strong のフェードアウトと同じ
1500 → 3000)。0.30 なら −0.42 でも f_lon = 0.43 と勾配が生きるので、発見期にどこへ
転んでも引き戻せる。lat は既に目標圏内なので σ_lat = 0.06 固定のまま。

この項は非負で、他のキック報酬とは **加算** で並ぶ (乗算にしない)。かつ内部で
r_direction に乗算しているので ``kick_done`` ゲートを通り、「構えるだけで蹴らない」
では 1 点も入らない。weight 2.0 は direction (6.0) / scaled (4.0) / strong (3.0) より
明確に下 — 目的そのものではなく「蹴り方」の指定だからで、これを上げすぎると
速度追従より構えの最適化に学習が引っ張られる。

学習手順 (3 段。stage 1 はリポジトリ同梱の checkpoint を再利用)::

    ./scripts/rsl_rl/train_walk_kick_middle.sh

段ごとに手で回す場合は :class:`K1WalkMiddleKickEnvCfg` の docstring を参照。
``--reset_noise_std`` は **使わないこと** (理由は walk_weak_kick の docstring と同じ)。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _disable_ball_obs_jitter,
    _KICK_STATE_PARAMS,
    _KICK_W_SCALE,
    _SIGMA_DIRECTION,
    K1WalkKick360EnvCfg,
    K1WalkKickEnvCfg,
)
from ..walk_weak_kick.walk_weak_kick_env_cfg import _apply_weak_kick_recipe

# steps_per_iteration = num_steps_per_env (PPO config)。基底の _spi / weak の _SPI と
# 同じ値だが、あちらは __post_init__ のローカル変数なので参照できずリテラルで持つ。
_SPI = 24

# --------------------------------------------------------------------------- #
# 目標ボール速度レンジ [m/s]
#
# 実機較正 (指令 2.0 m/s のキックが 2 m 転がった) から転がり減速 a ≈ 1.0 m/s²。
# d = v² / 2a を戦略要求の飛距離について解くと:
#
#   v = 3.2 m/s -> 5.1 m   (要求下限 5 m)
#   v = 3.9 m/s -> 7.6 m   (帯の中央)
#   v = 4.5 m/s -> 10.1 m  (要求上限 10 m)
#
# 基底 walk_kick の帯 (0.25, 2.0) をこの帯へ **最初から固定で** 張り替える。
# 帯カリキュラム (linear_command_speed_range) は入れない: 発見期の学習信号は
# latch + kick_direction + kick_velocity_strong が担っていて帯に依存せず、帯は
# kick_velocity_scaled の採点位置でしかないため (詳細はモジュール docstring)。
#
# NOTE: a ≈ 1.0 は 1 点計測なので不確かさが残る。実機でこのポリシーの球速と飛距離を
#       測ったら d = v²/2a で a を引き直し、この帯を締め直すこと。
# --------------------------------------------------------------------------- #
_SPEED_RANGE = (3.2, 4.5)

# --------------------------------------------------------------------------- #
# kick_velocity_scaled の σ アニール終点
#
# weak は 0.35 (帯幅 1.75 の (0.25, 2.0) 用)。この帯 (幅 1.3) では 0.5 を使う。
#
# * 識別性: 3.2 と 4.5 が 2.6σ 離れる。指令 3.2 に v=4.5 を出すと
#   f_vel = exp(−(1.3/0.5)²) ≈ 0.001 なので「太い σ に隠れて蹴りすぎる」は効かない。
# * 物理ノイズ耐性: 高速域はボール物性 DR による到達速度の絶対ばらつきが大きく、
#   0.35 まで絞ると自分では制御できない振れで報酬が半減してしまう。
#
# アニールの窓 (500 -> 3000 iteration) と開始値 1.0 は weak のまま。ここだけ上書きする。
# --------------------------------------------------------------------------- #
_SIGMA_VELOCITY_END = 0.5

# --------------------------------------------------------------------------- #
# 軸足配置 (kick_plant_foot) の目標値・シェイピング係数・重み
#
# 目標値は walk_lob と同一 (K1 の実寸 K1_22dof.xml から導出。導出そのものは
# :func:`~..walk_kick.mdp.rewards.kick_plant_foot` の docstring を参照):
#
#   _PLANT_LAT_TARGET = 0.19 : 股関節の横オフセット ±0.096 = 通常スタンス幅 0.192。
#       軸足を横 0.19 に置くと蹴り足がちょうどキック線上 (横 0) に来る。
#   _PLANT_SIGMA_LAT  = 0.06 : 下限は衝突限界。ボール半径 0.11 + 足箱の半幅 0.035 =
#       0.145 より内側は軸足がボールに当たる。実測 plant_lat は既に 0.207 で
#       f_lat = 0.95 なので、こちらは最初から締めたままでよい (アニールしない)。
#   _PLANT_LON_TARGET = -0.03: body_pos_w は足リンク原点 (足首) を返すが足箱の中心は
#       前方 +0.026。足の **中心** をボール真横に置くには足首を −0.026。
#
# σ_lon だけは lob の 0.10 固定から変えて、太い 0.30 で始めて 0.10 へ絞る。
# 0.10 は現状の実測 −0.42 では f_lon ≈ 5e-4 で勾配が死んでおり、発見期の下手な配置が
# 軒並み 0 点になってしまうため (``kick_velocity_scaled`` の σ アニールと同じ理屈)。
# 0.30 なら −0.42 で f_lon = 0.43。窓は strong のフェードアウトと同じ 1500 → 3000 に
# 合わせ、「強く蹴る」から「指令どおりに・正しい構えで蹴る」への移行を一体で動かす。
#
# 重み 2.0 は direction (6.0) / scaled (4.0) / strong (3.0) より明確に下。この項は目的
# そのものではなく「蹴り方」の指定なので、大きくすると速度追従より構えの最適化に学習が
# 引っ張られる。フェードインの窓は基底のキック報酬と同じ 0 → 500 で、発見期にはすでに
# 満額で乗っている状態にする (この項の存在意義が「型が決まる前に混ぜておく」ことなので、
# 他のキック項より遅らせてはいけない)。
#
# NOTE: 学習後は ``Metrics/kick_direction/plant_lon`` が −0.42 から −0.03 側へ動いたかを
#       見ること。同時に ``kick_rate`` / ``kick_vel_ratio`` / ``ball_touch_count`` が
#       悪化していないかも必ず併せて見る。悪化するなら重みを下げるか σ_lon の終点を
#       緩める (この項は「蹴り方」の指定であって、キックの成立より優先しない)。
# --------------------------------------------------------------------------- #
_PLANT_LON_TARGET = -0.03
_PLANT_SIGMA_LON_START = 0.30
_PLANT_SIGMA_LON_END = 0.10
_PLANT_LAT_TARGET = 0.19
_PLANT_SIGMA_LAT = 0.06
_PLANT_FOOT_WEIGHT = 2.0

# σ_lon を絞る窓。weak の strong フェードアウト / overshoot フェードインと同じ。
_PLANT_SIGMA_ANNEAL_START_ITER = 1500
_PLANT_SIGMA_ANNEAL_END_ITER = 3000


def _apply_middle_kick_recipe(cfg: "K1WalkKickEnvCfg") -> None:
    """weak の 3 点セット + DR をそのまま適用し、指令帯と σ の終点だけ差し替える。

    軸足配置 (``kick_plant_foot``) だけは weak に無い追加項。**stage 2 から入れる**
    (理由はモジュール docstring の「軸足配置を stage 2 から入れる」節)。

    stage 2 (:class:`K1WalkMiddleKickEnvCfg`) と stage 3
    (:class:`K1WalkMiddleKick360EnvCfg`) で共通の処理をここに集約する。
    観測・行動・地形・ボール配置には一切触らないので、観測 55 次元と並びは不変
    (= walk phase checkpoint をそのまま ``--load_pretrained`` できる)。
    ``__post_init__`` の最後 (基底クラスの設定が全部済んだ後) に呼ぶこと。
    """
    # -- 1. weak のレシピをまるごと適用 ----------------------------------- #
    #
    # (a) latch 閾値の指令追従 / (b) kick_velocity_strong の折れ線 /
    # (c) σ アニール + overshoot 罰 / 層2 ボール物性 DR。
    # (a) はこの帯では 0.6·v_target = 1.92-2.70 が clamp 上限 0.8 に常に張り付くので
    # 実効的に基底と同じ v_thresh=0.8 固定 (= no-op)。レシピ共有の副産物として
    # 無害に残るだけで、**latch 閾値を帯に連動させて上げてはいけない**
    # (上げると帯がポリシーの実力を追い越した瞬間に全キック報酬が消える)。
    _apply_weak_kick_recipe(cfg)

    # -- 2. σ アニールの終点だけ帯幅に合わせて上書き ----------------------- #
    #
    # weak が作った curriculum 項をそのまま使い、end_value だけ 0.35 -> 0.5 に差し替える。
    # 開始値・窓 (500 -> 3000 iteration) は weak と同じなので触らない。
    cfg.curriculum.kick_velocity_scaled_sigma.params["end_value"] = _SIGMA_VELOCITY_END

    # -- 3. 指令帯を最初から終点へ張り替える ------------------------------- #
    #
    # カリキュラムで動かさず固定。strong が実蹴りを帯の上 (v ≈ 6.0) まで押し上げてから、
    # σ アニールと overshoot 罰がこの帯の中へ絞り込む、という降ろし方をする。
    cfg.commands.kick_direction.target_speed_range = _SPEED_RANGE

    # -- 4. 軸足配置 (kick_plant_foot) を追加 ------------------------------ #
    #
    # 「蹴る瞬間に軸足がボールの真横」を狙う項。weak には無い middle だけの追加。
    #
    # * 他のキック報酬とは **加算** で並べる (kick_velocity_scaled 等に掛けてはいけない。
    #   学習初期は軸足がまず合わないので、掛けると速度側の勾配がゼロ付近で死ぬ)。
    # * 項の内部で r_direction に乗算しているので、kick_done ゲート・方向精度・胴体の
    #   正対を全て通過した蹴りにしか払われない。「構えるだけで蹴らない」は 0 点。
    # * 非負 (罰にしない)。外した配置は「罰される」のではなく「報われない」に留める。
    # * sigma_direction は基底のキック項と必ず揃えること (_SIGMA_DIRECTION = 0.35)。
    #   項ごとに違うと方位を外したときの損得が食い違って何を最適化しているのか読めなくなる。
    #
    # sigma_lon はここでは開始値を置くだけで、下の curriculum が毎ステップ上書きする。
    cfg.rewards.kick_plant_foot = RewTerm(
        func=mdp.kick_plant_foot,
        weight=0.0,
        params={
            **_KICK_STATE_PARAMS,
            "sigma_direction": _SIGMA_DIRECTION,
            "lon_target": _PLANT_LON_TARGET,
            "sigma_lon": _PLANT_SIGMA_LON_START,
            "lat_target": _PLANT_LAT_TARGET,
            "sigma_lat": _PLANT_SIGMA_LAT,
        },
    )
    # フェードインは基底のキック報酬と同じ窓 (0 → 500)。発見期には満額で乗せておく。
    cfg.curriculum.kick_plant_foot_weight = CurrTerm(
        func=mdp.linear_reward_weight,
        params={
            "term_name": "kick_plant_foot",
            "start_weight": 0.0,
            "end_weight": _PLANT_FOOT_WEIGHT * _KICK_W_SCALE,
            "start_step": 0,
            "end_step": 500,
            "steps_per_iteration": _SPI,
        },
    )
    # 前後の採点を 0.30 → 0.10 へ絞る。窓は strong のフェードアウトと同じ 1500 → 3000。
    cfg.curriculum.kick_plant_foot_sigma_lon = CurrTerm(
        func=mdp.linear_reward_param,
        params={
            "term_name": "kick_plant_foot",
            "param_name": "sigma_lon",
            "start_value": _PLANT_SIGMA_LON_START,
            "end_value": _PLANT_SIGMA_LON_END,
            "start_step": _PLANT_SIGMA_ANNEAL_START_ITER,
            "end_step": _PLANT_SIGMA_ANNEAL_END_ITER,
            "steps_per_iteration": _SPI,
        },
    )


@configclass
class K1WalkMiddleKickEnvCfg(K1WalkKickEnvCfg):
    """Stage 2 (middle): 限定レンジで「5-10 m 相当の指令どおりのキック」を獲得する。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickEnvCfg` との差は
    :func:`_apply_middle_kick_recipe` (= weak のレシピ + 帯 (3.2, 4.5) + σ 終点 0.5) だけ。
    観測・コマンド次元・ボール配置・地形は同一なので、**stage 1 (walk phase) の
    checkpoint をそのまま使える**::

        # stage 2 (リポジトリ同梱の walk phase checkpoint から)
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Middle-Kick-v0 \
            --headless --num_envs 4096 --max_iterations 5000 \
            --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt
        # stage 3 (stage 2 の checkpoint から)
        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-v0 \
            --headless --num_envs 4096 --max_iterations 5000 \
            --load_pretrained logs/rsl_rl/k1_walk_middle_kick/<run>/model_<N>.pt

    walk phase の checkpoint に 2026-08-03 の run を使うのは、こちらが
    ``knee_close_penalty`` を入れた後の学習で、現在のタスクの報酬構成と整合するため
    (:file:`.gitignore` のコメント参照)。

    **``--max_iterations`` は 3000 以上で回すこと。** カリキュラムが 3000 iteration で
    ようやく終点 (strong=0 / σ=0.5 / overshoot 満額) に着くので、それより短いと
    「まだ強く蹴った方が得」な途中状態で終わる。

    NOTE: ``--reset_noise_std`` は使わないこと。stage 1 からの引き継ぎなので歩行の
          スイングは既に精密で、std を戻すとそれを壊す (walk_weak_kick の NOTE と同じ)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_middle_kick_recipe(self)


@configclass
class K1WalkMiddleKickEnvCfg_PLAY(K1WalkMiddleKickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkMiddleKick360EnvCfg(K1WalkKick360EnvCfg):
    """Stage 3 (middle): 全方位版。stage 2 (middle) の checkpoint から続ける。

    :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKick360EnvCfg` との差は
    :func:`_apply_middle_kick_recipe` だけ。360 版固有の設定 (ボール配置の全方位化・
    ``approach_penalty`` → ``ball_avoidance`` の差し替え・``episode_length_s=15.0``) は
    基底の ``__post_init__`` で済んでおり、レシピはそれらに触れないのでそのまま残る。

    コマンド例は :class:`K1WalkMiddleKickEnvCfg` の docstring を参照。

    NOTE: このタスクのカリキュラムは stage 2 と同じ窓 (0/500/1500/3000 iteration) を
          **もう一度 0 から**回す。``--load_pretrained`` は common_step_counter を
          0 のままにするので、strong が再び立ち上がって落ちる。stage 2 で獲得済みの
          キックに対しては「強く蹴る期間」が余計だが、回り込みという新しい行動を
          獲得する段でもあるので、ここでキック報酬を厚くしておく方が安全と判断した
          (walk_weak_kick の stage 3 と同じ流儀)。指令帯は固定なので、この再ランプで
          動くのは strong の重みと σ と overshoot だけ。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_middle_kick_recipe(self)


@configclass
class K1WalkMiddleKick360EnvCfg_PLAY(K1WalkMiddleKick360EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkMiddleKick360NoisyBallEnvCfg(K1WalkMiddleKick360EnvCfg):
    """Stage 4 (middle): 知覚ノイズ+遅延つき。stage 3 (middle, 360) の checkpoint から続ける。

    実機で蹴り損ねが出る原因のうち、報酬構造ではなく **観測の質** の側を潰す段。
    :class:`K1WalkMiddleKick360EnvCfg` との差は policy のボール位置観測だけで、
    :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` が
    「エピソードごとランダム遅延 2-6 ステップ (40-120ms) + 30Hz サンプル&ホールド +
    フレーム同期ガウスジッタ σ=6.7cm・クリップ ±20cm」に差し替える (詳細はあちらの docstring)。

    middle のレシピ (weak の 3 点セット + 帯 (3.2, 4.5) + σ 終点 0.5 + ボール物性 DR) は
    基底の ``__post_init__`` で全て済んでおり、観測差し替えは報酬にもコマンド帯にも
    触れないのでそのまま残る::

        _labpython2 scripts/rsl_rl/train.py \
            --task Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-Noisy-Ball-v0 \
            --headless --num_envs 4096 --max_iterations 3000 \
            --load_pretrained logs/rsl_rl/k1_walk_middle_kick_360/<run>/model_<N>.pt

    NOTE: この帯では観測ノイズの影響が weak より大きく出る可能性がある。狙う飛距離が
          5-10 m と長いぶん、同じ蹴り角の誤差でも着弾のずれが比例して広がるため。
          weak 版と揃えた ±5cm / 2-6 ステップから始めて、実機の遅延を計測できたら
          :data:`~..walk_kick.walk_kick_env_cfg._BALL_OBS_DELAY_STEP_RANGE` を
          「計測値 + マージン」に絞ること (両系統で共有している定数なので、
          帯ごとに変えたくなったらここで params を上書きする)。

    NOTE: 基底のカリキュラム (0/500/1500/3000 iteration) の扱いは weak 版と同じ
          (:class:`~..walk_weak_kick.walk_weak_kick_env_cfg.K1WalkKick360WeakNoisyBallEnvCfg`
          の NOTE 参照)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_noisy_ball_obs(self)


@configclass
class K1WalkMiddleKick360NoisyBallEnvCfg_PLAY(K1WalkMiddleKick360NoisyBallEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)
