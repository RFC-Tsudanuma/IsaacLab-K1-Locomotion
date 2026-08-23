# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ウォークロブ環境（高さ特化のロブキック専用ポリシー）。

:class:`K1WalkLoopShootEnvCfg` を継承し、目的関数を「シュート」から **「ロブ」** へ
振り直したもの。

目的（優先順位はこの順）
------------------------
1. **確実に高く上げる。** ボール中心が K1 の身長 0.9 m を超えること。
   ボール半径 0.11 m なので必要な浮きは 0.79 m ⇔ vz = 3.94 m/s。
2. **sim2real 転移が容易であること。** 実機では射出仰角も反発も sim より目減りする。
   sim で「ぎりぎり 0.9 m」の解は実機で必ず届かないので、sim 側には余裕を持たせる。
3. 前方向の精度は **妥協する**。「狙った方位へ指定の速さで飛ばす」ことは
   Walk-Loop-Pass / Walk-Loop-Shoot の担当で、こちらは高さだけを取りに行く。

Walk-Loop-Shoot からの変更点
----------------------------
1. **速度追従報酬 (kick_velocity_scaled) を撤去する。**
   この項は f(v) = exp(−((v − v_target)/σ)²) の Gaussian で、指令帯 (5.5-7.0 m/s) より
   **速すぎても罰する**。一方こちらが最大化したいのは vz = v·sin φ で、v にも φ にも
   青天井の圧力をかけたい。「指令より速いと損」という形は vz 最大化と正面から衝突し、
   shoot では実際に v が 6.5 m/s 付近で頭打ちになる要因の一つになっていた。
   高さ特化のロブでは「指令速度への一致」自体が目的から外れるので項ごと落とす。
   撤去は報酬項とそのカリキュラム (``kick_velocity_scaled_weight``) の両方で行う。

   NOTE: ``kick_velocity_strong`` は loop_pass の時点で既に None (項ごと撤去) なので、
         これで速度系の報酬は ``kick_loft`` だけになる。「強く蹴れ」の圧力は
         kick_loft の vz = v·sinφ が担う (vz を上げるには v も必要) ので、
         速度への圧力そのものが消えるわけではない。

2. **kick_loft の vz_sat を 4.5 → 5.0 に上げる。**
   vz=5.0 は浮き 5.0²/(2·9.81) = 1.27 m 相当。目標の 0.79 m に対して 0.48 m の上積みで、
   これが実機での目減り分のマージン。飽和型 (線形ランプ) なので届かなくても勾配は
   死なず、逆に飽和させると圧力が消えて他項とのトレードオフで下がり始める
   (shoot で 2 回踏んだ失敗)。届かない前提で高めに置くのが正しい使い方。

3. **kick_elevation の phi_sat を 0.79 (45°) → 1.05 (60°) に上げる。**
   shoot の 45° 上限は「それ以上は前に飛ぶシュートでなくなる」という設計上の制約で、
   物理的な天井ではなかった。ロブでは前方向を妥協するのでこの制約を外す。
   ただし **真上は許さない**: 60° で頭打ちにしておくことで、後述の方向ゲートと合わせて
   「踏みつけて真上にポップ」型の解へ落ちるのを防ぐ。

4. **方向ゲートを緩める (sigma_direction 0.35 → 0.6)。** 項1-3 / loft / elevation が
   共有する f(τ) = exp(−τ²/2σ²) のシェイピング係数。0.6 は τ = 0.6 rad (34°) で
   f = 0.61、τ = 1.0 rad (57°) で f = 0.25。方位のずれに対する減点をおよそ半分の
   きつさにして、「高さを取るために方位が多少ぶれる」蹴り方を許す。

   NOTE: **方向ゲートそのものは絶対に撤去しないこと。** loft / elevation は
         ``r_direction`` への **乗算** で組まれており、この乗算が
         「ボールを踏みつけて真上にポップさせる」「かすらせて転がす」といった
         転移しない exploit を塞ぐ唯一の構造になっている。r_direction を外すと
         「ボールが上に飛びさえすればよい」になり、sim の接触モデル固有の解に落ちる。
         緩めるのは可、外すのは不可。

5. **軸足配置の報酬 (kick_plant_foot) を追加する。**
   「蹴る瞬間に軸足 (蹴っていない方の足) がボールの真横に来ている」ことを狙う項。
   軸足がボールの後方に残ったまま蹴ると、蹴り足はボールの **向こう側の高い位置** に
   当たるため足裏を低く潜らせられず、仰角が出ない。ロブは仰角が全てなので、
   「どう蹴るか」の中でも軸足の前後位置は特に効くはず、というのが仮説。

   既存のキック報酬とは **加算** で並べる (``kick_loft`` に掛けない)。項の内部は
   他と同じく ``r_direction`` への乗算で、非負 (外した配置は罰さず「報われない」だけ)。
   目標値の導出は :func:`~..walk_kick.mdp.rewards.kick_plant_foot` の docstring を参照。

   効いているかは ``Metrics/kick_direction/plant_lon`` (→ −0.03 に寄るか) と
   ``sole_height_at_kick`` (→ 下がるか) の 2 つで見る。前者が動いたのに後者が動かない
   なら、軸足位置は仰角の律速ではなかったということなので項を落とす判断ができる。

6. **アクチュエータの T-N カーブ差し替えは撤回した。**
   sim2real 転移の本命として Booster 公式の
   :class:`~..locomotion.actuators.BoosterDelayedPDActuator` (速度に応じてトルク上限が
   落ちる T-N カーブ) を一度導入したが、学習が全く進まなくなったため撤回した。
   walk_kick / walk_pass / walk_loop_* と同じ素の ``DelayedPDActuator`` に戻している。
   これにより stage 1 (walk phase) と stage 2 の物理は walk_kick 系と完全に同一になった
   ので、原理上は ``k1_walk_kick_walk_phase`` の checkpoint をそのまま
   ``--load_pretrained`` に使い回せる (walk_lob 専用の walk phase を別に回す必要はない)。

7. **すくい上げの報酬 (kick_foot_lift) を追加する。**
   「蹴る瞬間に蹴り足が上向きに動いている」ことを狙う項。Isaac Sim では浮くのに
   MuJoCo・実機では浮かない、という転移失敗への対策。

   原因はボールの反発係数で、Isaac の既定 (e≈0.6) では **足を水平に突っ込んで地面との
   間でボールを弾ませる** だけで vz が出るのに対し、MuJoCo・実機 (e≈0) ではその成分が
   丸ごと消える。``kick_loft`` / ``kick_elevation`` は結果 (ボールの vz・仰角) しか
   見ないのでこの機構の違いを区別できず、シミュレータ固有の解でも満点が出てしまう。
   この項は **原因側** (接触の瞬間の足の鉛直速度) を直接報酬にして、浮かせる
   メカニズムを反発非依存 = 運動学依存の側へ寄せる。

   既存項とは **加算** で並べる (``kick_loft`` に掛けない)。項の内部は他と同じく
   ``r_direction`` への乗算で、非負・飽和型 (青天井にしない)。kick_plant_foot と
   まったく同じ契約。導出は :func:`~..walk_kick.mdp.rewards.kick_foot_lift` の
   docstring と _FOOT_LIFT_VZ_SAT のコメントを参照。

   効いているかは ``Metrics/kick_direction/foot_vz`` (→ 正へ立ち上がるか) と
   ``kick_apex_height`` の 2 つで見る。apex は出ているのに foot_vz が 0 付近のままなら、
   その解はまだ反発に依存しており実機へ転移しない。

   NOTE: DR の反発範囲 (0.0-0.7) は既に e=0 を含んでいるが、それでも Isaac 固有の解が
         残るのは、DR が「たまたま e が低い env でも動く解」を要求するだけで
         「どの機構で浮かせるか」を指定しないため。この項はそこを直接指定する。

8. **ボール位置観測に知覚ノイズ+遅延を入れる (stage 2 から)。**
   :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` を
   ``__post_init__`` の最後で呼び、policy の ``prev_ball_pos`` を
   「エピソードごとランダム遅延 0-6 ステップ (0-120ms) + 30Hz サンプル&ホールド +
   フレーム同期ガウスジッタ σ=6.7cm・クリップ ±20cm」に差し替える。
   項7 と同じ **実機・MuJoCo でロブが失敗する** 問題への対策だが、そちらが「浮かせる
   機構」= 物理側の転移だったのに対し、こちらは「ボールがどこにあると思っているか」
   = 観測側の転移を潰す。実機の認識は遅れるうえ、たまに大きく外す裾を持つので、
   固定 1 ステップ遅延 + 毎ステップ独立 Unoise(±2cm) の既定では見えない誤差形になる。

   **weak / middle との違い**: あちらは 360 (stage 3) の後に stage 4 として
   :class:`~..walk_weak_kick.walk_weak_kick_env_cfg.K1WalkKick360WeakNoisyBallEnvCfg` /
   :class:`~..walk_middle_kick.walk_middle_kick_env_cfg.K1WalkMiddleKick360NoisyBallEnvCfg`
   を別クラス・別タスクとして積む。walk_lob には 360 ステージが無く 2 段構成なので、
   段を足さずに **stage 2 本体へ直接** 適用する。したがって walk_lob には
   「ノイズ無し版の stage 2」は存在しない (ノイズ有りが既定)。

   観測は 55 次元・並びとも変わらない (差し替えるのは ``prev_ball_pos`` の
   func/params/noise だけ) ので、walk phase や loop_shoot / loop_pass の checkpoint は
   これまでどおり ``--load_pretrained`` でそのまま載る。critic の ``prev_ball_pos``
   (ノイズ無し) と ``ball_vel`` には触らない。詳細と「触らないもの」は
   :func:`~..walk_kick.walk_kick_env_cfg._apply_noisy_ball_obs` の docstring 参照。

   PLAY では :func:`~..walk_kick.walk_kick_env_cfg._disable_ball_obs_jitter` で
   **ジッタだけ** 0 にする。遅延とサンプル&ホールドは観測パイプラインの構造そのもの
   (= PLAY で見たいもの) なので残す。``enable_corruption = False`` は ObsTerm の
   ``noise`` しか切らず、関数側へ移したジッタには効かないのでこの呼び出しが必須。

   NOTE: ``--reset_noise_std`` は使わないこと (weak / middle と同じ)。ノイズ耐性の
         獲得が目的で、獲得済みの蹴り方を壊す理由がない。

継承したまま変えないもの
------------------------
* ボール物性の DR (摩擦 0.3-1.0 / 0.2-0.8、反発 0.0-0.7、質量スケール 0.9-1.15)。
  すくい上げ型のキックは接触モデルへの overfit が起きやすく、sim2real 転移の観点では
  shoot より **こちらの方が重要度が高い**。範囲もそのまま引き継ぐ。
* 目標ボール速度帯 (5.5, 7.0)。kick_velocity_scaled を落としたのでこの帯は
  もはや報酬には一切使われないが、``target_kick_velocity`` 観測 (55 次元の 1 つ) と
  ``kick_vel_ratio`` メトリクスの分母としては生きているので変更しない。
* feet_phase / feet_slide。緩めない理由は walk_loop_shoot の NOTE と同じ
  (walk phase の checkpoint から歩行を再獲得する導線がこの 2 項なので、
  弱めると一度も歩けないまま固着する)。
* 観測 55 次元・行動空間。walk phase の checkpoint をそのまま
  ``--load_pretrained`` で引き継げる。

    ./scripts/rsl_rl/train_walk_lob.sh
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
    K1WalkKickWalkPhaseEnvCfg,
)
from ..walk_loop_shoot.walk_loop_shoot_env_cfg import K1WalkLoopShootEnvCfg

# --------------------------------------------------------------------------- #
# Loft 報酬の飽和上昇速度 vz_sat [m/s]
#
# kick_loft = r_dir * clamp(vz / vz_sat, 0, 1)。頂点高さは vz²/2g で vz だけで決まる。
#
# 目標は「ボール中心が K1 身長 0.9 m を超える」こと。ボール半径 0.11 m なので
# 必要な浮きは 0.79 m ⇔ vz = sqrt(2·9.81·0.79) = 3.94 m/s。
#
# vz_sat はそこに **マージンを載せた** 5.0 (浮き 1.27 m) を置く。理由:
#   * 実機では接触時の足速度も反発も sim より目減りするため、sim でぴったり 0.79 m の
#     解は実機でまず届かない。sim 側の到達点を高めに引く必要がある。
#   * 飽和型は線形ランプなので、届かない値を置いても勾配は死なない。逆に飽和させると
#     圧力が消えて他の報酬とのトレードオフで下がり始める (shoot で 2 回発生)。
#
# shoot の 4.5 は「φ ≤ 45° を保ったまま到達しうる上限 (6.4·sin45°)」だった。
# ロブは φ の上限を 60° まで開けるので、同じ v=6.4 でも vz は 5.5 まで届きうる。
# --------------------------------------------------------------------------- #
_LOB_VZ_SAT = 5.0

# --------------------------------------------------------------------------- #
# 角度の片側飽和 φ_sat [rad]
#
# 1.05 rad ≈ 60°。shoot の 0.79 rad (45°) は「45° を超えると水平成分が上昇成分を
# 下回り、前に飛ぶシュートでなくなる」という **設計上の** 制約だった。ロブは前方向を
# 妥協する前提なのでこの制約は撤廃する。
#
# それでも 90° (真上) にしないのは、真上まで許すと「ボールの真上から踏みつけて
# 弾き出す」型の解が最適になりうるため。60° で頭打ちにすれば、報酬上は 60° を超えても
# 得をしないので、前進成分を完全に捨てる動機が生まれない。
# 方向ゲート (r_direction 乗算) との二重の歯止め。
# --------------------------------------------------------------------------- #
_LOB_PHI_SAT = 1.05

# --------------------------------------------------------------------------- #
# 方向シェイピング係数 σ_direction [rad]
#
# f(τ) = exp(−τ² / 2σ²)、r_direction = (f − 0.5)·2·p_style。
# σ=0.35 (walk_kick 既定) では τ=0.35 rad (20°) で f=0.61、τ=0.6 rad (34°) で f=0.23、
# τ≈0.41 rad (24°) 以降は r_direction がクリップされて 0 になる = キック報酬が全て消える。
# 高さを取りにいくと接触点がずれて方位もぶれるので、この幅では「高く上げる蹴り方」の
# 探索そのものが罰されてしまう。
#
# σ=0.6 では τ=0.6 rad (34°) で f=0.61、r_direction が 0 になるのは τ≈0.71 rad (41°)。
# 方位の許容をおよそ 2 倍に開いた形。
#
# 繰り返しになるが **0 にしたり項を外したりはしない**。r_direction 乗算は
# 踏みつけ / かすらせ exploit を塞ぐ構造そのもの (モジュール docstring の NOTE 参照)。
# --------------------------------------------------------------------------- #
_LOB_SIGMA_DIRECTION = 0.6

# --------------------------------------------------------------------------- #
# 軸足配置 (kick_plant_foot) の目標値
#
# 「蹴る瞬間に軸足がボールの真横に来ている」ことを狙う。軸足がボールの後方に残ったまま
# 蹴ると、蹴り足はボールの向こう側の **高い位置** に当たるため足裏を低く潜らせられず、
# 仰角が出ない (sole_height_at_kick が下がらない)。ロブは仰角が全てなので、
# 「どう蹴るか」の中でも軸足の前後位置は特に効くはず、というのがこの項の仮説。
#
# 値は K1 の実寸 (K1_22dof.xml) から決めた。導出は
# :func:`~..walk_kick.mdp.rewards.kick_plant_foot` の docstring に書いてある。
#
#   _PLANT_LAT_TARGET = 0.19 : 股関節の横オフセット ±0.096 = 通常スタンス幅 0.192。
#       軸足を横 0.19 に置くと蹴り足がちょうどキック線上 (横 0) に来る。
#   _PLANT_SIGMA_LAT  = 0.06 : 下限は衝突限界。ボール半径 0.11 + 足箱の半幅 0.035 =
#       0.145 より内側は軸足がボールに当たる。
#   _PLANT_LON_TARGET = -0.03: body_pos_w は足リンク原点 (足首) を返すが足箱の中心は
#       前方 +0.026。足の **中心** をボール真横に置くには足首を −0.026。
#   _PLANT_SIGMA_LON  = 0.10 : ±0.1 で半値。歩幅の分解能的にこれ以上は締めない。
#
# NOTE: まず学習を回す前に Metrics/kick_direction/plant_lon を見て **現状の軸足位置**を
#       確認すること。現状が −0.25 のように大きく後方なら、いきなり −0.03 を狙わせるより
#       カリキュラムで目標を動かす方が素直（今は固定目標で入れてある）。
# --------------------------------------------------------------------------- #
_PLANT_LON_TARGET = -0.03
_PLANT_SIGMA_LON = 0.10
_PLANT_LAT_TARGET = 0.19
_PLANT_SIGMA_LAT = 0.06

# --------------------------------------------------------------------------- #
# kick_plant_foot の最終重み (× _KICK_W_SCALE)
#
# 2.0 は direction (6.0) / loft (5.0) / elevation (5.0) より明確に下。この項は目的
# そのものではなく **「蹴り方」の指定** なので、大きくすると高さより構えの最適化に
# 学習が引っ張られる。kick_rate と kick_apex_height を見ながら上げること。
# --------------------------------------------------------------------------- #
_PLANT_FOOT_WEIGHT = 2.0

# --------------------------------------------------------------------------- #
# すくい上げ報酬 (kick_foot_lift) の飽和足速度 vz_foot_sat [m/s] と最終重み
#
# kick_foot_lift = r_dir * clamp(foot_vz / vz_foot_sat, 0, 1)。foot_vz は latch 時に
# 凍結した蹴り足のワールド鉛直速度 (+ = 上向き)。
#
# なぜ要るか: walk_lob は Isaac Sim では浮くのに MuJoCo・実機では浮かない。ボールの
# 反発係数が Isaac では e≈0.6、MuJoCo・実機では ≈0 で、**足を水平に突っ込んで地面との
# 間でボールを弾ませる** 型の解は e が消えると丸ごと機能しなくなる。kick_loft /
# kick_elevation は結果 (ボールの vz・仰角) しか見ないのでこの機構の違いを区別できず、
# シミュレータ固有の解でも満点が出てしまう。この項は原因側 (接触の瞬間に足自身が上へ
# 動いているか) を直接報酬にして、浮かせるメカニズムを反発非依存の側へ寄せる。
#
#   _FOOT_LIFT_VZ_SAT = 2.0 : ボール vz 目標 (_LOB_VZ_SAT = 5.0) に対して運動学的に
#       必要な足速度の目安。剛体衝突なので 1:1 では移らないが、飽和型 (線形ランプ) は
#       届かない値を置いても勾配が死なないので厳密である必要はない。実測は
#       Metrics/kick_direction/foot_vz で見て、飽和しきっていたら上げること。
#   _FOOT_LIFT_WEIGHT = 2.0 : kick_plant_foot と同じ。direction (6.0) / loft (5.0) /
#       elevation (5.0) より明確に下。これも目的そのものではなく **「蹴り方」の指定**
#       なので、大きくすると高さより蹴り方の最適化に学習が引っ張られる。
#
# NOTE: 既存項とは **加算** で並べる (kick_loft に掛けない)。学習初期はすくい上げが
#       まず出ないので、掛けると loft の勾配がゼロ付近で死ぬ。kick_plant_foot と同じ契約。
# --------------------------------------------------------------------------- #
_FOOT_LIFT_VZ_SAT = 2.0
_FOOT_LIFT_WEIGHT = 2.0


@configclass
class K1WalkLobEnvCfg(K1WalkLoopShootEnvCfg):
    """高さ特化のロブキック専用。Walk-Loop-Shoot と観測・行動空間は同一。"""

    def __post_init__(self) -> None:
        # NOTE: **必ず super() を先に呼ぶこと。**
        #       親 (K1WalkLoopShootEnvCfg / K1WalkLoopPassEnvCfg) の __post_init__ は
        #       `self.rewards.kick_velocity_scaled.params[...] = ...` を実行するので、
        #       先に None を代入すると親側で AttributeError になる。
        #       「親が組み立て終わったものを、あとから壊す」順序でなければならない。
        super().__post_init__()

        # -- 1. 速度追従 (kick_velocity_scaled) を撤去
        #
        # f(v) = exp(−((v − v_target)/σ)²) は指令帯より速くても罰する Gaussian で、
        # vz = v·sin φ の最大化と衝突する。高さ特化のロブでは「指令速度への一致」が
        # そもそも目的に入っていないので、報酬項とカリキュラムの両方を落とす。
        # (カリキュラム側の名前は walk_kick_env_cfg の `kick_velocity_scaled_weight`。
        #  報酬項だけ消してカリキュラムを残すと linear_reward_weight が
        #  存在しない term を触りにいくため、必ず対で消すこと。)
        self.rewards.kick_velocity_scaled = None
        self.curriculum.kick_velocity_scaled_weight = None

        # -- 2. kick_loft: vz の飽和点を 4.5 → 5.0 (浮き 1.27 m) へ、方向ゲートを緩める
        self.rewards.kick_loft.params["vz_sat"] = _LOB_VZ_SAT
        self.rewards.kick_loft.params["sigma_direction"] = _LOB_SIGMA_DIRECTION

        # -- 3. kick_elevation: φ の飽和点を 45° → 60° へ、方向ゲートを緩める
        #
        # phi_sat が入っているので kick_elevation は Gaussian ではなく
        # clamp(φ / φ_sat, 0, 1) の片側飽和モードで動く (親から継承)。
        self.rewards.kick_elevation.params["phi_sat"] = _LOB_PHI_SAT
        self.rewards.kick_elevation.params["sigma_direction"] = _LOB_SIGMA_DIRECTION

        # -- 4. kick_direction 本体の方向ゲートも同じ幅に揃える
        #
        # 3 項でシェイピング係数がばらばらだと、方位を外したときの損得が項ごとに
        # 食い違って何を最適化しているのか読めなくなる。σ は揃えること。
        self.rewards.kick_direction.params["sigma_direction"] = _LOB_SIGMA_DIRECTION

        # NOTE: `kick_velocity_strong` は loop_pass の __post_init__ で None (項ごと撤去)
        #       になっており、ここには存在しない。これで有効なキック報酬は
        #       kick_direction / kick_loft / kick_elevation の 3 本 (+ ball 追従系)。

        # -- 5. 軸足配置 (kick_plant_foot) を追加
        #
        # 「蹴る瞬間に軸足がボールの真横」を狙う項。既存項とは **加算** で並べる
        # (kick_loft に掛けてはいけない。学習初期は軸足がまず合わないので loft の勾配が
        #  ゼロ付近で死ぬ)。項自体は r_direction への乗算なので、方向ゲート・kick_done
        # ゲート・胴体の正対を通過した蹴りにしか払われない。
        #
        # sigma_direction は他の 3 項と必ず揃えること (_LOB_SIGMA_DIRECTION)。
        # 項ごとに違うと方位を外したときの損得が食い違って何を最適化しているのか読めなくなる。
        self.rewards.kick_plant_foot = RewTerm(
            func=mdp.kick_plant_foot,
            weight=0.0,
            params={
                **_KICK_STATE_PARAMS,
                "sigma_direction": _LOB_SIGMA_DIRECTION,
                "lon_target": _PLANT_LON_TARGET,
                "sigma_lon": _PLANT_SIGMA_LON,
                "lat_target": _PLANT_LAT_TARGET,
                "sigma_lat": _PLANT_SIGMA_LAT,
            },
        )
        self.curriculum.kick_plant_foot_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "kick_plant_foot",
                "start_weight": 0.0,
                "end_weight": _PLANT_FOOT_WEIGHT * _KICK_W_SCALE,
                "start_step": 0,
                "end_step": 500,
                "steps_per_iteration": 24,
            },
        )

        # -- 6. すくい上げ (kick_foot_lift) を追加
        #
        # 「蹴る瞬間に蹴り足が上へ動いている」ことを狙う項。walk_lob が Isaac Sim では
        # 浮くのに MuJoCo・実機では浮かない問題への対策で、浮かせるメカニズムを
        # ボールの反発係数依存 (Isaac e≈0.6 / MuJoCo・実機 e≈0) から運動学依存へ移す。
        # 詳しい理屈は _FOOT_LIFT_VZ_SAT のコメントと
        # :func:`~..walk_kick.mdp.rewards.kick_foot_lift` の docstring を参照。
        #
        # kick_plant_foot とまったく同じ契約で置く: 既存項とは **加算** で並べ
        # (kick_loft に掛けない)、項の内部は r_direction への乗算、非負 (すくい上げの
        # ない蹴りは罰さず「報われない」だけ)、飽和で青天井にしない。
        #
        # sigma_direction は他のキック報酬と必ず揃えること (_LOB_SIGMA_DIRECTION)。
        #
        # 効いているかは ``Metrics/kick_direction/foot_vz`` (→ 正へ立ち上がるか) と
        # ``kick_apex_height`` の 2 つで見る。apex は出ているのに foot_vz が 0 付近の
        # ままなら、その解はまだ反発に依存しており実機へ転移しない。
        self.rewards.kick_foot_lift = RewTerm(
            func=mdp.kick_foot_lift,
            weight=0.0,
            params={
                **_KICK_STATE_PARAMS,
                "sigma_direction": _LOB_SIGMA_DIRECTION,
                "vz_foot_sat": _FOOT_LIFT_VZ_SAT,
            },
        )
        self.curriculum.kick_foot_lift_weight = CurrTerm(
            func=mdp.linear_reward_weight,
            params={
                "term_name": "kick_foot_lift",
                "start_weight": 0.0,
                "end_weight": _FOOT_LIFT_WEIGHT * _KICK_W_SCALE,
                "start_step": 0,
                "end_step": 500,
                "steps_per_iteration": 24,
            },
        )

        # -- 7. ボール位置観測を実機の認識パイプライン寄りにする
        #
        # 項6 (kick_foot_lift) と同じ「実機・MuJoCo でロブが失敗する」問題への対策の、
        # 観測側。
        # 固定 1 ステップ遅延 + 毎ステップ独立 Unoise(±2cm) を、エピソードごとランダム
        # 遅延 0-6 ステップ + 30Hz サンプル&ホールド + フレーム同期ガウスジッタ
        # (σ=6.7cm・クリップ ±20cm) に差し替える。
        #
        # weak / middle は 360 (stage 3) の後に stage 4 の別クラスとして積んでいるが、
        # walk_lob には 360 ステージが無い 2 段構成なので、段を足さずに stage 2 本体へ
        # 直接入れる (= walk_lob にノイズ無しの stage 2 は無い)。
        #
        # **必ず全ての報酬・カリキュラム登録が済んだ後 (= __post_init__ の最後) に呼ぶこと。**
        # 差し替えるのは policy の prev_ball_pos の func/params/noise だけで、観測の
        # 次元・並び (55 次元) は変わらないので checkpoint はそのまま載る。
        # critic 側の prev_ball_pos と ball_vel には触らない (あちらの docstring 参照)。
        _apply_noisy_ball_obs(self)


@configclass
class K1WalkLobEnvCfg_PLAY(K1WalkLobEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # enable_corruption = False は ObsTerm の noise しか切らないので、関数側へ移した
        # ジッタは別途 0 にする。遅延とサンプル&ホールドは観測パイプラインの構造
        # (= PLAY で見たいもの) なので残す。weak / middle の PLAY と同じ扱い。
        _disable_ball_obs_jitter(self)


@configclass
class K1WalkLobWalkPhaseEnvCfg(K1WalkKickWalkPhaseEnvCfg):
    """Stage 1 (歩行のみ) の walk_lob 版。

    T-N カーブ付きアクチュエータを一度導入したが学習が全く進まなくなったため撤回した
    (walk_lob_env_cfg モジュール docstring の変更点 6 参照)。アクチュエータは
    ``K1WalkKickWalkPhaseEnvCfg`` から一切変更していないので、**物理は walk_kick の
    walk phase と完全に同一**。experiment 名 (``k1_walk_lob_walk_phase``) を分けて
    ログを混ざらせないためだけにクラスを残してあるが、実体は空のサブクラスなので、
    ``k1_walk_kick_walk_phase`` の既存 checkpoint をそのまま流用しても支障はない。
    """


@configclass
class K1WalkLobWalkPhaseEnvCfg_PLAY(K1WalkLobWalkPhaseEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 4
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
