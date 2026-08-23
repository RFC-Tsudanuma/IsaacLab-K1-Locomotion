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
NOTE: stage 4 以降は蹴り方向の観測に自己位置推定の遅延 (0.15-0.30 s) が入る
      (:func:`~..walk_kick.walk_kick_env_cfg.enable_localization_delay`)。
      ボール知覚のノイズ+遅延と同じ扱いで、この段で初めて入るので、stage 3 の
      checkpoint から続けると方向の指標が一度悪化する。
"""

from isaaclab.utils import configclass

from ..walk_kick.walk_kick_env_cfg import (
    _apply_noisy_ball_obs,
    _apply_walk_state_init,
    _disable_ball_obs_jitter,
    enable_localization_delay,
    K1WalkKick360EnvCfg,
    K1WalkKickEnvCfg,
)
# 弱いキックのレシピ (latch 閾値の指令追従 / strong の折れ線 / σ アニール /
# overshoot 罰 / ボール物性 DR) は元のモジュールをそのまま使う。二重定義しない。
from ..walk_weak_kick.walk_weak_kick_env_cfg import _apply_weak_kick_recipe
from .orbit_mods import apply_ball_param_dr, apply_orbit_params


# --------------------------------------------------------------------------- #
# キック報酬の倍率は入れない (失敗記録 2026-08-19)
#
# 一度 walk_kick_ball_avoid に倣って「キック 4 項 (正 3 項 + overshoot 罰) を ×3」を
# 入れたが、**逆効果だったので外した**。
#
# 入れた理由と、それが間違いだった理由:
#   * 理由 … weak は kick_velocity_strong が 3000 iteration で 0 に落ち、
#     σ_velocity も 1.0 → 0.35 に絞られるので実入りが小さい。ball_avoidance との
#     比率が悪いと「広く回るだけで蹴らない」に落ちる、と考えた。
#   * 実際 … 3 本とも kick_rate 0.99-0.999 で、**その問題は起きなかった**。
#     存在しない問題への対策だった。
#
# 何が壊れたか (k1_walk_kick_weak_orbit/2026-08-19_01-18-45 の実測):
#
#   進捗   右足率   球速比
#    2%    0.603    1.34     ← 両足で蹴れている (過去最良の 2 本が 0.36-0.40)
#    5%    0.516    1.20
#   10%    0.424    1.88
#   20%    0.038    6.73     ← 片足に崩壊。同時に球速が指令の 6.7 倍へ暴走
#   30%    0.033    6.89
#   50%    0.031    1.20     ← overshoot が効き始めて球速だけ戻る
#  100%    0.018    1.26     ← 蹴り足は戻らない
#
# 機序: ``kick_velocity_strong`` の折れ線は [(0,0),(500,W),(1500,W),(3000,0)] で、
# ``kick_velocity_overshoot`` のランプは **1500 → 3000**。つまり
# **iteration 500-1500 は「速いほど得」だけが効いて、止める側が 0** の窓がある。
# ここを ×3 したので、その 1000 iteration だけ強く蹴る圧力が 3 倍になり、
# 対抗する項が無いまま暴走した。その暴走期に方策が片足へ固まり、1500 以降で
# overshoot が入って球速は戻ったが、蹴り足の選択は戻らなかった。
#
# overshoot も同率で ×3 したので比率は保たれる、と考えたのが誤り。
# **0 を 3 倍しても 0** で、問題の窓では効いていない。
#
# 対照: ×3 を入れていない walk_long_pass_orbit は右足のまま・方向誤差 3.74° で、
# 帯・回り込み G・σ_sole は同一。差分は ×3 だけだった。
#
# 旧 weak との最終比較:
#   旧 weak        右足率 0.99  方向誤差 6.8°  球速比 1.10-1.14
#   weak_orbit ×3  右足率 0.01  方向誤差 8.9°  球速比 1.25-1.26
#
# 再導入するなら、先に overshoot のランプ開始を 1500 → 500 に動かして
# strong のプラトーを覆うこと。窓を揃えずに倍率だけ触ってはいけない。
# --------------------------------------------------------------------------- #


def _apply_orbit_recipe(cfg) -> None:
    """weak のレシピを掛けたうえで、回り込み G / 跨ぎの遊び / ボール DR を足す。

    **順序が意味を持つ。**

    1. :func:`..walk_weak_kick.walk_weak_kick_env_cfg._apply_weak_kick_recipe`
       が ``kick_velocity_overshoot`` 報酬項を新しく作り、ボール物性 DR
       (walk_loop_shoot 由来の範囲) を入れる。
    2. :func:`.orbit_mods.apply_ball_param_dr` がボール物性 DR の範囲を
       **こちらの範囲で上書きする** (1 の後でないと上書きされる側になる)。
    3. :func:`.orbit_mods.apply_orbit_params` が回り込みのパラメータを
       kick_state を読む **全ての項** に配り、``ball_avoidance`` の σ_sole を
       0.20 に絞る。1 で追加された ``kick_velocity_overshoot`` も配布対象なので、
       必ず最後に呼ぶ。
    """
    _apply_weak_kick_recipe(cfg)
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
          指令の円弧 (``r_max`` = 0.5) の内側へ引っ込めてある。キック報酬の倍率は
          **入れない** (上の失敗記録参照)。
          切り分けの読み方: **半径が縮まらなければ指令 (G) 側の問題**、
          **蹴らなくなれば実入り側の問題** (kick_rate で見る。実測では 0.99 以上で
          問題は出ていない)。
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

    :class:`K1WalkKick360WeakOrbitEnvCfg` との差は policy の観測 2 スロットだけ。
    観測の差し替えは報酬にもコマンドにも触らないので、回り込みの設定はそのまま残る。

    * ``prev_ball_pos``: ボール知覚 (エピソードごとランダム遅延 0-6 ステップ +
      30Hz サンプル&ホールド + ガウスジッタ σ=6.7cm・クリップ ±20cm)。
    * ``kick_direction``: 自己位置推定の遅延 0.15-0.30 s
      (:func:`~..walk_kick.walk_kick_env_cfg.enable_localization_delay`)。
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        _apply_noisy_ball_obs(self)

        # -- 蹴り方向の観測に自己位置推定の遅延を掛ける
        #
        # 実機の自己位置推定はカメラのランドマーク認識と InEKF (FK + IMU) ででき
        # ており、出力が policy に届くまでに遅れがある。蹴り方向はフィールド地図上の
        # 座標で与えるので、体基準に直すにはロボット自身のヨー角が要る。遅れたヨー角で
        # 変換すると policy が見る蹴り方向が実際とずれる。既定 0.15-0.30 s で、
        # env ごと・エピソードごとに引き直す。詳細は enable_localization_delay。
        #
        # 上のボール知覚とは別枠。ボール位置・速度はカメラが体基準で直接測る量で
        # 自己位置推定を通らないので含めない。掛かるのは kick_direction 1 スロットだけ。
        #
        # NOTE: Stage 1-3 では掛けない。ボール知覚のノイズ+遅延と同じ扱いで、
        #       センサ由来の遅延はこの段で初めて入る。段を跨いで条件が変わるので、
        #       Stage 3 の checkpoint から入ると方向の指標
        #       (Metrics/kick_direction/kick_dir_error_deg) が一度悪化する。
        # NOTE: この系統の観測は 55 次元・1 フレームで履歴が無い。policy が使える
        #       手掛かりは同じフレームのジャイロ (base_ang_vel) だけなので、
        #       履歴入力を持つ walk_long_pass_orbit より条件が厳しい。
        enable_localization_delay(self)


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
