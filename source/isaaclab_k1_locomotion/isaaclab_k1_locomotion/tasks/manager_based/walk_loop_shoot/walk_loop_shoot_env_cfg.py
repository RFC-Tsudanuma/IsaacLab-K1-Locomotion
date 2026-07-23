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
1. ``kick_elevation`` (角度ベース) を ``kick_loft`` (**vz ベースの片側飽和**) に
   差し替える（最重要）。頂点高さは vz²/2g で vz = v·sinφ だけで決まる。角度ベースは
   速度に無関心なため「速度帯の下限で角度だけ付ける」解を許し、初版 (角度片側飽和 +
   速度帯 3.0-4.0) は実測 φ≈1.4° のほぼ水平な蹴りに収束した。
2. 目標ボール速度を (4.0, 5.5) に上げる。φ の幾何上限 (~35°) の下では速度が
   唯一の残りレバーのため（(v=4.0, φ=35°) の完璧な実行でも頂点 0.27 m しか出ない）。

NOTE: feet_phase / feet_slide は緩めない (詳細は __post_init__ の NOTE 参照)。
      当初緩めたら歩行を再獲得できず一度も歩けなかった。

NOTE: ドメインランダム化で幾何のマージンを稼ぐ案もあったが、Isaac Lab では spawn 後に
      コライダー半径を env ごとに変えられないため、ボール半径や地面高さのランダム化は
      素直には書けない。片側飽和は「常に物理上限を狙わせる」ことで実質的に同じ効果
      (マージン最大化) を狙っている。まずはこちらの効果を見てから次を考えること。
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import _KICK_STATE_PARAMS, _KICK_W_SCALE, _SIGMA_DIRECTION
from ..walk_loop_pass.walk_loop_pass_env_cfg import K1WalkLoopPassEnvCfg

# --------------------------------------------------------------------------- #
# Loft 報酬の飽和上昇速度 vz_sat [m/s]
#
# kick_loft = r_dir * clamp(vz / vz_sat, 0, 1)。頂点高さは vz²/2g で vz だけで決まる。
# 2.5 m/s は頂点 0.32 m・滞空 0.51 s に相当し、「はっきり浮いた」と見える帯。
# 速度帯の下限 4.0 m/s なら φ=33°、上限 5.5 m/s なら φ=24° で飽和に届く。
#
# 当初は角度の片側飽和 (φ_sat=0.61) を使っていたが、角度は速度に無関心なため
# 「速度帯の下限で角度だけ付ける」解を許してしまい、実測 φ≈1.4° の水平蹴りに
# 収束した。vz は「強く」と「上に」を同時に要求する。
# --------------------------------------------------------------------------- #
_SHOOT_VZ_SAT = 2.5

# --------------------------------------------------------------------------- #
# シュート用の目標ボール速度 [m/s]（3D ノルム基準）
#
# 当初 (3.0, 4.0) だったが、仕様上限の (v=4.0, φ=35°) を完璧に実行しても頂点 0.27 m・
# 滞空 0.47 s しか出ない (頂点 = (v·sinφ)²/2g、φ は幾何上限 ~42°)。「はっきり飛ぶ」には
# 速度を盛るしかないので上げる。B-Human の評価表では K1 の Strong キックが sim で
# 5-6.8 m/s 出ており、物理的に届く帯。
# --------------------------------------------------------------------------- #
_SHOOT_SPEED_RANGE = (4.0, 5.5)

# kick_velocity_scaled の速度シェイピング係数 [m/s]。帯を上げて広げたぶん緩める。
# kick_vel_ratio が 0.8 程度に留まる現状で厳しくシェイプすると勾配が薄くなるため。
_SHOOT_SIGMA_VELOCITY = 1.0


@configclass
class K1WalkLoopShootEnvCfg(K1WalkLoopPassEnvCfg):
    """ループシュート専用。Walk-Loop-Pass と観測・行動空間は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1. 仰角報酬 (kick_elevation) を Loft 報酬 (kick_loft) に差し替える（最重要）
        #
        # 角度ベースの報酬は速度に無関心で、「速度帯の下限でわずかに角度を付ける」
        # 水平蹴り (実測 φ≈1.4°) に収束した。頂点高さを決める vz を直接狙わせる。
        self.rewards.kick_elevation = None
        self.curriculum.kick_elevation_weight = None
        self.rewards.kick_loft = RewTerm(
            func=mdp.kick_loft,
            weight=0.0,
            params={
                **_KICK_STATE_PARAMS,
                "sigma_direction": _SIGMA_DIRECTION,
                "vz_sat": _SHOOT_VZ_SAT,
            },
        )
        self.curriculum.kick_loft_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "kick_loft",
                "start_weight": 0.0,
                "end_weight": 5.0 * _KICK_W_SCALE,
                "start_step": 0,
                "end_step": 500,
                "steps_per_iteration": 24,
            },
        )

        # -- 2. 目標ボール速度をシュートの帯へ
        self.commands.kick_direction.target_speed_range = _SHOOT_SPEED_RANGE
        self.rewards.kick_velocity_scaled.params["sigma_velocity"] = _SHOOT_SIGMA_VELOCITY

        # NOTE: feet_phase / feet_slide は **緩めない**。
        #
        # walk phase の checkpoint は kick 環境に置くと iter 0 で episode_length が
        # 15-20 まで一旦崩れ、feet_phase (2.0) / feet_slide (-0.5) に導かれて歩行を
        # 再獲得して初めて 140 前後まで上がる (成功した walk_kick run で確認)。
        # 当初この 2 項を 1.0 / -0.1 に弱めたところ、std 0.05 の細い探索では歩行の
        # 再獲得ができず、episode_length 20 のまま固着して一度も歩けなかった。
        #
        # 「足を低く通す」圧力は kick_loft (vz ベース) が担う。角度ベースの旧 kick_elevation
        # と違い vz は速度に反応するので、水平蹴りには報酬が出ない。まずこれだけで
        # 足が下がるか見る。足りなければ feet を「歩行再獲得後に」段階的に緩める
        # カリキュラム (start_step を 1000 以降に置く) を足すこと。いきなり弱めないこと。


@configclass
class K1WalkLoopShootEnvCfg_PLAY(K1WalkLoopShootEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
