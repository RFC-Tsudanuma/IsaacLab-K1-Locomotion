# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークループシュート環境（はっきり浮かせるシュート専用ポリシー）。

:class:`K1WalkLoopPassEnvCfg` を継承し、「もっと高く・もっと遠くへ飛ばす」方向に
振り直したもの。Walk-Loop-Pass は実機で「軽く浮かせるパス」として機能したが、
シュートと呼べる浮きは出なかった。

目標は **ボール中心が K1 の身長 (足裏→頭部 ≈ 0.80 m) まで上がること**。
試合で AMP のキックがその高さまで浮かせた実績があるので、そこを基準に置いている。
浮き 0.69 m ⇔ vz = 3.68 m/s。

射出仰角を決めるのは幾何ではなく足の速度
----------------------------------------
当初は「足コライダー (box: pos=(0.026,0,-0.02), size=(0.09,0.035,0.018)、接地時の
上エッジ高 3.6cm) と球の静的な接触で法線が決まるので、射出仰角は接触時の足の高さ
だけで決まり上限 42°」という幾何モデルを立て、足を低く通すことを目指していた。

**この読みは PLAY の実測で否定された。** 実測は足裏 6-7.5 cm で φ = 29-36°。
幾何モデルの予測 (足裏 6.5cm → φ≈5°) と 7 倍ずれる。

実際には足は **振り上がりながら** ボールに当たっており、射出方向は接触点の幾何では
なく **足の速度ベクトル (特に上向き成分)** が支配している。足がスイングの上昇局面で
ボールをすくい上げる形。したがって:

* φ に「幾何上限 42°」のような制約は無い。上限は足の上向き速度で決まる。
* 「足を低く通す」ことは目的ではない。``sole_height_at_kick`` は良し悪しの指標では
  なく「スイングのどこで当たっているか」を見る指標として読むこと。
* feet_phase / feet_slide を緩める理由も無い (下の NOTE 参照)。

Walk-Loop-Pass からの変更点
---------------------------
1. キック高さの報酬を 2 本立てにする。頂点高さは vz²/2g で vz = v·sinφ だけで決まるので、
   vz を上げる = v と φ の両方を押し上げる必要がある。
   a. ``kick_loft`` (vz ベースの片側飽和) … vz を直接狙う。初版の角度ベース
      ``kick_elevation`` 単独は速度に無関心で φ≈1.4° の水平蹴りに収束したため導入。
   b. ``kick_elevation`` (φ の片側飽和, φ_sat=40°) … 角度に下限を課す。kick_loft 単独だと
      ポリシーが速度側に逃げて φ が 28° で頭打ちになり loft が 0.47 m で寝たため、
      角度も明示的に押し上げる。
2. 目標ボール速度を (5.5, 7.0) に上げる。vz を稼ぐには速度も要るため。v3d は指令帯に
   忠実に追従するので帯を上げないと伸びない (ただし実測では ~6.5 m/s で頭打ち)。

NOTE: feet_phase / feet_slide は緩めない (詳細は __post_init__ の NOTE 参照)。
      当初 (幾何モデルに基づき「足を低く通す探索を塞いでいる」と考えて) 緩めたところ、
      歩行を再獲得できず一度も歩けなかった。しかも上記のとおり狙い自体が的外れだった。
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
#
# 3.7 m/s は「ボール中心が K1 の身長 (足裏→頭部 ≈ 0.80 m) まで上がる」値。
# 浮き = 0.80 − 0.11 (接地時のボール中心) = 0.69 m、vz = sqrt(2g·0.69) = 3.68。
# 試合で AMP のキックが K1 身長まで浮かせた実績があるので、そこを目標に置く。
#
# 当初 2.5 (浮き 0.32 m) にしていたが、実測 vz≈2.6 でほぼ飽和し圧力が消えていた。
# 飽和型は線形ランプなので、届かない値を置いても勾配は死なない。
#
# 当初は角度の片側飽和 (φ_sat) を使っていたが、角度は速度に無関心なため
# 「速度帯の下限で角度だけ付ける」解を許し、実測 φ≈1.4° の水平蹴りに収束した。
# vz は「強く」と「上に」を同時に要求する。
# --------------------------------------------------------------------------- #
_SHOOT_VZ_SAT = 3.7

# --------------------------------------------------------------------------- #
# シュート用の目標ボール速度 [m/s]（3D ノルム基準）
#
# 実測 φ≈32° の下で vz=3.7 に届かせるには v = 3.7/sin(32°) ≈ 6.9 m/s が要る
# (φ が 40° まで伸びれば 5.7 m/s)。v3d は指令帯の中央付近に忠実に追従するので、
# 帯を上げないと速度は伸びない。
#
# 上限 7.0 は B-Human の K1 実測 (sim で 5-6.8 m/s) の上限域で、やや背伸びした値。
# 届かなくても kick_loft (飽和型・線形ランプ) の勾配は死なないが、
# kick_vel_ratio は 0.8 前後まで下がる見込み。それは想定内として、
# loft の実測が伸びているかで判断すること。
# --------------------------------------------------------------------------- #
_SHOOT_SPEED_RANGE = (5.5, 7.0)

# kick_velocity_scaled の速度シェイピング係数 [m/s]。帯を上げたぶん緩める。
# 厳しくシェイプすると、届きにくい高速域で勾配が薄くなるため。
_SHOOT_SIGMA_VELOCITY = 1.2

# --------------------------------------------------------------------------- #
# 角度の片側飽和 φ_sat [rad]（kick_loft と併用する kick_elevation 用）
#
# kick_loft (vz) だけだと、ポリシーは φ を上げるより v を上げる方を選ぶ。しかし
# v は ~6.5 m/s で頭打ちになり、φ は 27-28° で固着して loft が 0.47 m で寝た
# (vz=6.5·sin28°=3.05)。φ=38° まで上がれば同じ v=6.5 で vz=4.0 (loft 0.8m) 出るのに、
# 速度側に逃げて角度を探索しない。そこで角度にも下限を要求する項を戻す。
#
# 0.70 rad ≈ 40°。実測上限らしき 28° より十分上に置いて「もっと角度を」の勾配を残す。
# 片側飽和 (kick_elevation の phi_sat モード) なので 40° 以上で頭打ち、
# r_direction 乗算で踏みつけ exploit も塞がれたまま。
#
# NOTE: これで φ が 28° から動かなければ、28° がこの箱足コライダー + スイングで出せる
#       角度の物理上限。その場合は目標高さを下げる判断になる (報酬では上がらない)。
# --------------------------------------------------------------------------- #
_SHOOT_PHI_SAT = 0.70


@configclass
class K1WalkLoopShootEnvCfg(K1WalkLoopPassEnvCfg):
    """ループシュート専用。Walk-Loop-Pass と観測・行動空間は同一。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # -- 1a. Loft 報酬 (kick_loft, vz ベース)
        #
        # 角度ベースの旧 kick_elevation は速度に無関心で「速度帯の下限で角度だけ付ける」
        # 水平蹴り (φ≈1.4°) に収束したため、頂点高さを決める vz を直接狙う項に置換した。
        # ただし vz 単独だと逆にポリシーが速度側 (v) に逃げて φ が 28° で頭打ちになった
        # ので、下の 1b で角度にも下限を課す。両輪で vz を押し上げる。
        self.rewards.kick_elevation = None  # デフォルト定義 (Gaussian) を一度消す
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

        # -- 1b. 角度報酬 (kick_elevation, φ の片側飽和) を kick_loft と併用で復活
        #
        # phi_sat を渡すと kick_elevation は Gaussian ではなく clamp(φ/φ_sat, 0, 1) を使う。
        # 「速度は十分・角度が足りない」現状で、φ を 40° まで押し上げる勾配を足す。
        self.rewards.kick_elevation = RewTerm(
            func=mdp.kick_elevation,
            weight=0.0,
            params={
                **_KICK_STATE_PARAMS,
                "sigma_direction": _SIGMA_DIRECTION,
                "phi_sat": _SHOOT_PHI_SAT,
            },
        )
        self.curriculum.kick_elevation_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "kick_elevation",
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
