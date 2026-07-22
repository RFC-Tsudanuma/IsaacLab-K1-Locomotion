# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークループシュート環境（はっきり浮かせるシュート専用ポリシー）。

:class:`K1WalkLoopPassEnvCfg` を継承し、「もっと高く・もっと遠くへ飛ばす」方向に
振り直したもの。Walk-Loop-Pass は実機で「軽く浮かせるパス」として機能したが、
シュートと呼べる浮きは出なかった。その原因は報酬の強さではなく **幾何** にある。

射出仰角を決めているのは接触時の足の高さだけ
--------------------------------------------
足コライダー (MuJoCo の box: pos=(0.026,0,-0.02), size=(0.09,0.035,0.018)) は
接地時、足先の上エッジが地面から 3.6cm の位置にある。球とボックスの接触では法線が
「エッジ → 球中心」を向くので、射出仰角はこのエッジ高さとボール半径 (11cm) だけで
決まる::

    足の持ち上がり   接触エッジ高   射出仰角
      + 0.0 cm        3.6 cm       42.3°
      + 1.0 cm        4.6 cm       35.6°
      + 2.0 cm        5.6 cm       29.4°
      + 3.6 cm        7.2 cm       20.2°
      + 5.0 cm        8.6 cm       12.6°
      + 7.4 cm       11.0 cm        0.0°   ← エッジがボール中心の高さに並ぶ

**足が 2cm 高いだけで 13° 失う。** 実機で全く飛ばなかったのは、カーペット・ソール厚・
速い振りでの追従遅れなどで足が数 cm 高い位置を通っているためと考えられる。

なお足首の底屈は +0.345 rad (約 20°) しかなく、計算すると爪先を下げてもエッジ高さは
2mm しか下がらない（薄い箱を回すだけで相殺される）。**「爪先ですくう」動作は K1 の足
では効かない。** 効くのは足を低く通すことだけ。

Walk-Loop-Pass からの変更点
---------------------------
1. ``kick_elevation`` を Gaussian から **片側飽和** に変える（最重要）。
   Gaussian で 30° を狙わせると、ポリシーは「足を 2cm 浮かせた状態」に *最適化* して
   しまう。それが Walk-Loop-Pass の挙動そのもの。実機ではそこからさらに目減りするので
   浮きが消える。片側飽和なら「できる限り低く通す」が最適解になり、実機で目減りしても
   浮きが残る余裕 (マージン) ができる。
2. 目標ボール速度を上げる。シュートとしての飛距離を出すため。

NOTE: ドメインランダム化で幾何のマージンを稼ぐ案もあったが、Isaac Lab では spawn 後に
      コライダー半径を env ごとに変えられないため、ボール半径や地面高さのランダム化は
      素直には書けない。片側飽和は「常に物理上限を狙わせる」ことで実質的に同じ効果
      (マージン最大化) を狙っている。まずはこちらの効果を見てから次を考えること。
"""

from isaaclab.utils import configclass

from ..walk_loop_pass.walk_loop_pass_env_cfg import K1WalkLoopPassEnvCfg

# --------------------------------------------------------------------------- #
# 片側飽和の飽和角 φ_sat [rad]
#
# f(φ) = clamp(φ / φ_sat, 0, 1)。φ_sat 以上は頭打ちなので青天井にはならず、
# 「踏みつけて真上に跳ね上げる」exploit は r_direction 乗算と併せて塞がれたまま。
#
# 0.61 rad ≈ 35°。接触の第一瞬間の理論上限が 42° で、ボールが足先を乗り越える間に
# 実効角はそれより下がるため、35° は「ほぼ上限」を意味する。ここを飽和点にすることで
# 「足をできる限り低く通す」が最適解になる。
#
# NOTE: 42° に近づけすぎると達成不能域で勾配が飽和せず、いつまでも f<1 のまま
#       「もっと低く」を要求し続けて爪先を地面に擦る解に落ちうる。まず 35° で様子を見る。
# --------------------------------------------------------------------------- #
_SHOOT_PHI_SAT = 0.61

# --------------------------------------------------------------------------- #
# シュート用の目標ボール速度 [m/s]（3D ノルム基準）
#
# Walk-Loop-Pass は (2.0, 3.0)。シュートは飛距離が要るので上げる。
# 45° 換算の飛距離は v=3.0 で約 0.9m、v=4.0 で約 1.6m。
# 上限 4.0 は Walk-Kick の帯 (1.0-4.0) の上限と同じで、地面蹴りでは到達実績がある値。
# --------------------------------------------------------------------------- #
_SHOOT_SPEED_RANGE = (3.0, 4.0)

# kick_velocity_scaled の速度シェイピング係数 [m/s]。帯が 3.0-4.0 と Pass (2.0-3.0) と
# 同じ幅なので、Pass の 0.7 をそのまま引き継ぐ（明示のため再掲）。
_SHOOT_SIGMA_VELOCITY = 0.7


@configclass
class K1WalkLoopShootEnvCfg(K1WalkLoopPassEnvCfg):
    """ループシュート専用。Walk-Loop-Pass と観測・行動空間は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 仰角報酬を片側飽和に切り替える（最重要の変更）
        #
        # phi_sat を渡すと kick_elevation は Gaussian ではなく clamp(φ/φ_sat, 0, 1) を使う。
        # phi_target / sigma_phi は無視されるが、設定の履歴として残しておく。
        self.rewards.kick_elevation.params["phi_sat"] = _SHOOT_PHI_SAT

        # -- 2. 目標ボール速度をシュートの帯へ
        self.commands.kick_direction.target_speed_range = _SHOOT_SPEED_RANGE
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _SHOOT_SIGMA_VELOCITY


@configclass
class K1WalkLoopShootEnvCfg_PLAY(K1WalkLoopShootEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
