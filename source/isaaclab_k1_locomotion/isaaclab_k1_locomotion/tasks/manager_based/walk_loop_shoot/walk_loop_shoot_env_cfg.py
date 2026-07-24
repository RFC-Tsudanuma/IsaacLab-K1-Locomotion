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
3. ボール物性 (摩擦・反発・質量) を env ごとにランダム化する。すくい上げ型のキックは
   ボール物性への依存が強く、固定値で学習すると接触モデルに overfit する。実際
   IsaacLab で十分浮くポリシーが MuJoCo では 3-5 回に 1 回しか浮かなかった。

NOTE: feet_phase / feet_slide は緩めない (詳細は __post_init__ の NOTE 参照)。
      当初 (幾何モデルに基づき「足を低く通す探索を塞いでいる」と考えて) 緩めたところ、
      歩行を再獲得できず一度も歩けなかった。しかも上記のとおり狙い自体が的外れだった。
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..walk_kick import mdp
from ..walk_kick.walk_kick_env_cfg import _KICK_STATE_PARAMS, _KICK_W_SCALE, _SIGMA_DIRECTION
from ..walk_loop_pass.walk_loop_pass_env_cfg import K1WalkLoopPassEnvCfg

# --------------------------------------------------------------------------- #
# Loft 報酬の飽和上昇速度 vz_sat [m/s]
#
# kick_loft = r_dir * clamp(vz / vz_sat, 0, 1)。頂点高さは vz²/2g で vz だけで決まる。
#
# 4.5 m/s は **この構成で狙える理論上限**。導出:
#   * 総速度 v3d は 6.3-6.5 m/s で打ち止め (実測の分解 5.19²+3.58² → 6.30 が、関節の
#     velocity 上限から出る足先速度上限 6.54 とほぼ一致。ハードウェアの壁)。
#   * φ の設計上限は 45°。それを超えると水平成分が上昇成分を下回り、「前に飛ぶ
#     シュート」ではなく「真上へのロブ」になって kick_direction (重み最大) と衝突する。
#   → vz 上限 = 6.4·sin(45°) = 4.53。loft 1.03 m / apex 1.14 m 相当。
#
# 履歴: 2.5 (実測 vz≈2.6 で飽和) → 3.7 (実測 vz≈3.58, 飽和度 97% で頭打ち) → 4.5。
# 飽和型は線形ランプなので、届かない値を置いても勾配は死なない。飽和させると
# 圧力が消えて他の報酬とのトレードオフで下がり始める、を 2 回踏んでいる。
#
# 当初は角度の片側飽和 (φ_sat) 単独だったが、角度は速度に無関心なため
# 「速度帯の下限で角度だけ付ける」解を許し、実測 φ≈1.4° の水平蹴りに収束した。
# vz は「強く」と「上に」を同時に要求する。
# --------------------------------------------------------------------------- #
_SHOOT_VZ_SAT = 4.5

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
# 0.79 rad ≈ 45°。vz_sat の導出と同じ設計上限 (45° 超は前に飛ぶシュートでなくなる)
# に合わせる。片側飽和 (kick_elevation の phi_sat モード) なので 45° 以上で頭打ち、
# r_direction 乗算で踏みつけ exploit も塞がれたまま。
#
# 履歴: φ 実測は 28° (vz 単独) → 31° (φ_sat=40° 併用) → 33.4° (DR 追加後) と、
# 報酬をかけるたびに更新されてきた。φ の物理上限は事前予測が 3 回外れており不明。
# 45° は物理ではなく「シュートとして意味を保つ」設計上の上限。
# --------------------------------------------------------------------------- #
_SHOOT_PHI_SAT = 0.79

# --------------------------------------------------------------------------- #
# ボール物性のドメインランダマイゼーション範囲
#
# すくい上げ型のキックは vz を 2 経路で稼ぐ:
#   * 接線方向 (摩擦)   … 足がボールを上へ「引きずり上げる」
#   * 法線方向 (反発)   … 足がボールの下側を叩いた瞬間の跳ね返り
# どちらもボール物性に強く依存するため、単一の値で学習すると **その物理エンジンの
# 接触モデルに overfit する**。実際、IsaacLab (摩擦 1.0/0.8, restitution 0.6) で
# 十分浮くポリシーが MuJoCo (摩擦 0.4, solref dampratio=1 ⇒ 反発ほぼ 0) では
# 3-5 回に 1 回しか浮かなかった。
#
# 範囲は **両シミュレータの値を内包する** ように取る:
#   静摩擦 0.3-1.0  (MuJoCo 0.4 / IsaacLab 1.0 の両方を含む)
#   動摩擦 0.2-0.8  (同上)
#   反発   0.0-0.7  (MuJoCo ~0 / IsaacLab 0.6 の両方を含む)
# これで「どの物性でも浮く蹴り方」に寄せる。実機がどちら寄りかは不明なので、
# どちらか一方に合わせるのではなく両方をカバーする方針を取っている。
#
# NOTE: 半径だけは spawn 後に env ごとへ変えられないので固定 (0.11 m)。
# NOTE: mode="startup" なので env ごとに固定値が 1 回だけ割り当てられる。4096 env
#       あれば分布としては十分。エピソードごとに振りたい場合は mode="reset"。
# --------------------------------------------------------------------------- #
_BALL_STATIC_FRICTION_RANGE = (0.3, 1.0)
_BALL_DYNAMIC_FRICTION_RANGE = (0.2, 0.8)
_BALL_RESTITUTION_RANGE = (0.0, 0.7)
# 質量 [kg] のスケール範囲。既定 0.45 kg に対して 0.40-0.52 kg。
_BALL_MASS_SCALE_RANGE = (0.9, 1.15)


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

        # -- 3. ボール物性のドメインランダマイゼーション（sim2sim / sim2real 対策）
        #
        # loop_pass が spawn で固定していた physics_material (摩擦 1.0/0.8・反発 0.6) を
        # env ごとに上書きする。範囲の根拠は _BALL_*_RANGE のコメント参照。
        self.events.ball_physics_material = EventTerm(
            func=loco_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "static_friction_range": _BALL_STATIC_FRICTION_RANGE,
                "dynamic_friction_range": _BALL_DYNAMIC_FRICTION_RANGE,
                "restitution_range": _BALL_RESTITUTION_RANGE,
                "num_buckets": 64,
            },
        )
        self.events.ball_mass = EventTerm(
            func=loco_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "mass_distribution_params": _BALL_MASS_SCALE_RANGE,
                "operation": "scale",
            },
        )

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
