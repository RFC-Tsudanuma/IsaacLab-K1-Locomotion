# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークミドルキック環境（5-10 m 飛ぶキックを指令どおりに出す / weak レシピの帯違い）。

戦略側の要求は「5-10 m 飛ぶキックを、指令した強さで出せること」。
:mod:`..walk_weak_kick` が「指令どおりの強さのキック」を実際に成立させたレシピなので、
**そのレシピをそのまま使い、指令帯だけを高速側へ張り替える**のがこのタスク。
一言でいえば **middle = weak + 帯の張り替え**であり、報酬構造・カリキュラムの窓・
DR は weak と 1 バイトも変えていない (σ の終点だけ後述の理由で 0.5)。

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

学習手順 (3 段。stage 1 はリポジトリ同梱の checkpoint を再利用)::

    ./scripts/rsl_rl/train_walk_kick_middle.sh

段ごとに手で回す場合は :class:`K1WalkMiddleKickEnvCfg` の docstring を参照。
``--reset_noise_std`` は **使わないこと** (理由は walk_weak_kick の docstring と同じ)。
"""

from isaaclab.utils import configclass

from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _disable_ball_obs_jitter,
    K1WalkKick360EnvCfg,
    K1WalkKickEnvCfg,
)
from ..walk_weak_kick.walk_weak_kick_env_cfg import _apply_weak_kick_recipe

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


def _apply_middle_kick_recipe(cfg: "K1WalkKickEnvCfg") -> None:
    """weak の 3 点セット + DR をそのまま適用し、指令帯と σ の終点だけ差し替える。

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
    フレーム同期ジッタ ±5cm」に差し替える (詳細はあちらの docstring)。

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
