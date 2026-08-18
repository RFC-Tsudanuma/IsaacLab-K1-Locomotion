# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウィークキック環境 + 回り込み G（walk_weak_kick の写し + 3 つの改造）。

:mod:`..walk_weak_kick.walk_weak_kick_env_cfg` の 4 段構成
(stage 2 → 3 → 4 → 5) をそのままなぞり、次の 3 点だけを足したもの。
弱いキックのレシピ (:func:`..walk_weak_kick.walk_weak_kick_env_cfg._apply_weak_kick_recipe`)
は **元のモジュールから import して再利用** しており、二重定義していない。

足した 3 点 (:mod:`.orbit_mods`)
-------------------------------
1. **回り込み型の目標終端 G** (``r_max=0.5`` / ``orbit_beta=0.6``)

   元の G はボールの真後ろの直線上にしか置けないので、ボールの正面にいるロボットへは
   「ボールを突き抜けろ」という指令になっていた。回り込みは ``ball_avoidance``
   (姿勢ができていないのに近いと罰) が遠回りに追い込むことでしか成立しておらず、
   指令と罰が逆向きに引っ張り合う構造だった。ここでは G 自体をボールを中心とする
   円弧の上に置き、**指令の側で回り込みを作る**。ボールから G までの距離は
   ``r_max`` で直接指定する。

2. **キック線を跨ぐときの遊び** (``overshoot_margin=0.25``)

   反対側の足でボールを蹴るには base がキック線を約 0.096 m (スタンス幅 0.192 m の
   半分) 越えて立つ必要がある。元は跨いだ瞬間に無条件で罰していたので、正しい
   蹴り方の一部まで罰していた。体幅ぶん + マージンまでは許し、それを超えた
   「回り直し」だけを罰する。

3. **ボールまわりの 4 点 DR** (足の反発 / ボール物性 / 初期回転 / 転がり減速)

   ボール物性の範囲は weak のレシピが入れる walk_loop_shoot 由来の範囲を
   **上書きする** (レシピを呼んだ後に :func:`.orbit_mods.apply_ball_param_dr` を
   呼ぶ順序が必須)。質量スケールの DR はレシピのものと同じ範囲を入れ直している。

学習手順は元の weak と同じ 4 段。stage 1 (walk phase) はリポジトリ同梱の
checkpoint を再利用する。観測は 55 次元・並びとも元の weak と同一なので、
checkpoint はそのまま載る (改造はどれも観測にも行動空間にも触らない)::

    # stage 2 (weak, orbit)
    _labpython2 scripts/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-K1-Walk-Kick-Weak-Orbit-v0 \
        --headless --num_envs 4096 --max_iterations 5000 \
        --load_pretrained logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt
    # stage 3 (360)
    _labpython2 scripts/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-v0 \
        --headless --num_envs 4096 --max_iterations 5000 \
        --load_pretrained logs/rsl_rl/k1_walk_kick_weak_orbit/<run>/model_<N>.pt

``--reset_noise_std`` は **使わないこと** (理由は元の weak の docstring と同じ)。

NOTE: G の作り方が変わるのは **``follow_ball=True`` の段だけ**。stage 2 以降は
      全部そうなので、4 段すべてに同じ改造を入れてある。段によって G が変わると
      前段で覚えた歩き方・回り込み方が次の段で通用しなくなる。
"""

from isaaclab.utils import configclass

from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _apply_walk_state_init,
    _disable_ball_obs_jitter,
    K1WalkKick360EnvCfg,
    K1WalkKickEnvCfg,
)
# 弱いキックのレシピ (latch 閾値の指令追従 / strong の折れ線 / σ アニール /
# overshoot 罰 / ボール物性 DR) は元のモジュールをそのまま使う。二重定義しない。
from ..walk_weak_kick.walk_weak_kick_env_cfg import _apply_weak_kick_recipe
from .orbit_mods import apply_ball_param_dr, apply_orbit_params


# --------------------------------------------------------------------------- #
# キック正報酬の倍率
#
# walk_kick_ball_avoid の診断に倣う: あちらでは「蹴ると早期終了で歩行の dense 収入
# ≈ +6〜8 を没収」「学習初期の蹴り 1 回の実収入 ≈ +0.3」で **蹴る = 約 −6 の取引**
# になっており、キック 3 項 ×3 + kick_latch_bonus で kick_rate が 0.998 まで戻った。
#
# weak はそこからさらに実入りが小さい。``kick_velocity_strong`` は 3000 iteration で
# 0 へ落ちる除去スケジュールに載っており、``kick_velocity_scaled`` は σ_velocity が
# 1.0 → 0.35 へ絞られるので、下手なうちは満額のごく一部しか入らない。この状態で
# ``ball_avoidance`` との比率が悪いと「広く回るが蹴らない」へ落ちる。そこで
# **正のキック項だけ** 同じ倍率で持ち上げる。
#
# 負側 (``kick_velocity_overshoot``) は weak の趣旨そのものなので触らない。
# ``kick_latch_bonus`` はここでは入れない (これでも蹴らないに落ちたら次の手)。
# --------------------------------------------------------------------------- #
_KICK_W_BOOST = 3.0


def _apply_kick_weight_boost(cfg) -> None:
    """正のキック 3 項の終端 weight を :data:`_KICK_W_BOOST` 倍する。

    **必ず :func:`..walk_weak_kick.walk_weak_kick_env_cfg._apply_weak_kick_recipe`
    の後に呼ぶこと。** ``kick_velocity_strong`` の折れ線はレシピが入れるので、
    先に呼ぶとレシピの上書きで倍率が消える。
    """
    cfg.curriculum.kick_direction_weight.params["end_weight"] *= _KICK_W_BOOST
    cfg.curriculum.kick_velocity_scaled_weight.params["end_weight"] *= _KICK_W_BOOST

    # strong は単調ランプではなく折れ線 [(0,0),(500,W),(1500,W),(3000,0)] なので、
    # 非ゼロのプラトーだけが倍になる (0 は 0 のまま、iteration の折れ点も不変)。
    # knots はレシピ側のモジュール定数を指しているので、**新しいリストに差し替える**
    # (in-place で書き換えると素の weak タスクまで巻き添えになる)。
    _strong = cfg.curriculum.kick_velocity_strong_weight
    _strong.params["knots"] = [(_it, _w * _KICK_W_BOOST) for _it, _w in _strong.params["knots"]]


def _apply_orbit_recipe(cfg) -> None:
    """weak のレシピを掛けたうえで、回り込み G / 跨ぎの遊び / ボール DR を足す。

    **順序が意味を持つ。**

    1. :func:`..walk_weak_kick.walk_weak_kick_env_cfg._apply_weak_kick_recipe`
       が ``kick_velocity_overshoot`` 報酬項を新しく作り、ボール物性 DR
       (walk_loop_shoot 由来の範囲) を入れる。
    2. :func:`_apply_kick_weight_boost` が正のキック 3 項を ×3 する。1 が
       ``kick_velocity_strong`` の折れ線を入れ直すので、その後でないと消える。
    3. :func:`.orbit_mods.apply_ball_param_dr` がボール物性 DR の範囲を
       **こちらの範囲で上書きする** (1 の後でないと上書きされる側になる)。
    4. :func:`.orbit_mods.apply_orbit_params` が回り込みのパラメータを
       kick_state を読む **全ての項** に配り、``ball_avoidance`` の σ_sole を
       0.20 に絞る。1 で追加された ``kick_velocity_overshoot`` も配布対象なので、
       必ず最後に呼ぶ。
    """
    _apply_weak_kick_recipe(cfg)
    _apply_kick_weight_boost(cfg)
    apply_ball_param_dr(cfg)
    apply_orbit_params(cfg)


@configclass
class K1WalkKickWeakOrbitEnvCfg(K1WalkKickEnvCfg):
    """Stage 2 (weak, orbit): 限定レンジで「指令どおりの強さのキック」を獲得する。

    :class:`~..walk_weak_kick.walk_weak_kick_env_cfg.K1WalkKickWeakEnvCfg` との差は
    :func:`_apply_orbit_recipe` が足す 3 点だけ。観測・行動空間は同一なので
    stage 1 (walk phase) の checkpoint をそのまま使える。

    **``--max_iterations`` は 3000 以上で回すこと。** weak のカリキュラムが
    3000 iteration でようやく終点 (strong=0 / σ=0.35 / overshoot 満額) に着く。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_orbit_recipe(self)


@configclass
class K1WalkKickWeakOrbitEnvCfg_PLAY(K1WalkKickWeakOrbitEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKick360WeakOrbitEnvCfg(K1WalkKick360EnvCfg):
    """Stage 3 (weak, orbit): 全方位版。stage 2 (weak, orbit) の checkpoint から続ける。

    回り込み G がいちばん効くのはこの段。ボールの正面から半周回り込むエピソードで、
    元の構成では「G はボールの真後ろ (= ボールの向こう側)」を指したまま
    ``ball_avoidance`` の罰だけが遠回りを作っていた。

    NOTE: ``ball_avoidance`` は残すが **σ_sole を 0.35 → 0.20 に絞って**、罰の届く範囲を
          指令の円弧 (``r_max`` = 0.5) の内側へ引っ込めてある。同時に、正のキック 3 項を
          ×3 して蹴り 1 回の実入りを上げている (:data:`_KICK_W_BOOST`)。
          切り分けの読み方: **半径が縮まらなければ指令 (G) 側の問題**、
          **蹴らなくなれば実入り側の問題**。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_orbit_recipe(self)


@configclass
class K1WalkKick360WeakOrbitEnvCfg_PLAY(K1WalkKick360WeakOrbitEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class K1WalkKick360WeakOrbitNoisyBallEnvCfg(K1WalkKick360WeakOrbitEnvCfg):
    """Stage 4 (weak, orbit): 知覚ノイズ+遅延つき。stage 3 の checkpoint から続ける。

    :class:`K1WalkKick360WeakOrbitEnvCfg` との差は policy のボール位置観測だけ
    (エピソードごとランダム遅延 2-6 ステップ + 30Hz サンプル&ホールド +
    ガウスジッタ σ=6.7cm・クリップ ±20cm)。観測差し替えは報酬にもコマンドにも
    触らないので、回り込みの設定はそのまま残る。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_noisy_ball_obs(self)


@configclass
class K1WalkKick360WeakOrbitNoisyBallEnvCfg_PLAY(K1WalkKick360WeakOrbitNoisyBallEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)


@configclass
class K1WalkKick360WeakOrbitNoisyBallWalkInitEnvCfg(K1WalkKick360WeakOrbitNoisyBallEnvCfg):
    """Stage 5 (weak, orbit): 歩行ポリシーの歩行状態から reset して再学習する。

    :class:`K1WalkKick360WeakOrbitNoisyBallEnvCfg` との差は **リセットの初期状態だけ**。
    状態プールのパスは環境変数 ``K1_WALK_STATES_NPZ`` で渡す (詳細は
    :func:`~..walk_kick.walk_kick_env_cfg._apply_walk_state_init`)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_walk_state_init(self)


@configclass
class K1WalkKick360WeakOrbitNoisyBallWalkInitEnvCfg_PLAY(K1WalkKick360WeakOrbitNoisyBallWalkInitEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        _disable_ball_obs_jitter(self)
