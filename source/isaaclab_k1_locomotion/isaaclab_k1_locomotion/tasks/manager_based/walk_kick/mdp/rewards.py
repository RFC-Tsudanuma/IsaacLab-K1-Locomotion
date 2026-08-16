# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick のキック報酬。

B-Human "A Modular Ball Kicking Behavior with Reinforcement Learning" の
報酬テーブルを K1 向けに実装したもの。latch 状態は :mod:`.kick_state` が管理する。

フェーズゲート:
  * pre-latch  (kick_done=false): 項1-3 = 0、項4/5 有効
  * L 発火時                    : τ_direction, v_ball, p_style を凍結、kick_done=true、G を P_kick に固定
  * post-latch (kick_done=true) : 項1-3 を凍結値で毎ステップ dense に払う、項5 = 0、項4 継続
  * 項6 (overshoot) は常時有効
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .kick_state import kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _r_direction(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float,
) -> tuple[torch.Tensor, dict]:
    """r_direction = (f(τ_direction) − 0.5) * 2 * p_style （いずれも凍結値）。

    f(τ) = exp(−τ² / 2σ²) なので τ=0 (方向ぴったり) で f=1 → r_direction = +p_style、
    方向が大きく外れると f→0 → r_direction = −p_style。
    pre-latch では凍結値が 0 のままなので、呼び出し側で kick_done ゲートを掛けること。

    NOTE: 負値は 0 にクリップする。項1-3 は post-latch に凍結値を毎ステップ dense で払うが、
          転倒すると base_contact でエピソードが終わり支払いも止まる。負のまま払うと
          「方向を外したキックの後は早く転んで損失を止めた方が得」という抜け道ができるため
          (転倒罰は -100 * dt = -2.0 の一度きりなのに対し、負の dense 払いは窓の秒数だけ
          累積する)。クリップの代償として、方向を外したキックは「罰される」のではなく
          「報われない」(= 蹴らないのと同値) 扱いになる。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    tau = state["tau_direction_frozen"]
    f_dir = torch.exp(-(tau**2) / (2.0 * sigma_direction**2))
    r_dir = (f_dir - 0.5) * 2.0 * state["p_style_frozen"]
    r_dir = torch.clamp(r_dir, min=0.0)

    # pre-latch は 0
    return r_dir * state["kick_done"].float(), state


def kick_direction(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    v_gate_frac: float = 0.0,
    sigma_gate: float = 0.05,
) -> torch.Tensor:
    """項1. Kick Direction。凍結した飛翔方向誤差 × 凍結 p_style。shape: (N,)

    ``v_gate_frac > 0`` で片側の速度ゲート g(v) = sigmoid((v_ball − f·v_target) / σ_gate)
    を掛ける。**弱すぎる蹴りだけを削り、蹴りすぎは削らない**（オーバーシュートは項2 が見る）。

    このゲートが必要な理由: 項1 は重み最大 (6.0) でありながら素の定義では速度に一切依存
    しないため、「接近中に足がボールをかすっただけ」でも方向さえ合っていれば満点が出る。
    しかも latch は不可逆で、その 100 ステップ後に kick_finished がエピソードを終わらせる。
    結果、ポリシーは振り足を出す前にエピソードが終わり「ちゃんと蹴った経験」を一度も
    サンプリングできなくなる（walk_pass の 3000 iteration で実際にこれが起きた）。

    デフォルト 0.0 でゲート無効 = 従来の挙動。walk_pass のように v_target が
    かすり当ての速度域と近いタスクでのみ有効にする。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    if v_gate_frac <= 0.0:
        return r_dir

    v_gate = v_gate_frac * state["v_target"]
    g = torch.sigmoid((state["v_ball_frozen"] - v_gate) / sigma_gate)
    return r_dir * g


def kick_velocity_scaled(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    sigma_velocity: float = 1.0,
    use_3d_speed: bool = False,
) -> torch.Tensor:
    """項2. Kick Velocity Scaled = r_direction * f(v_ball)。shape: (N,)

    f(v_ball) は「要求された蹴り速度 v_target にどれだけ一致したか」を測る。
    指令速度に対する一致度を見ないと可変キック強度が学習できないため、
    f(v) = exp(−((v − v_target) / σ)²) とした。

    ``use_3d_speed=True`` で水平ノルムではなく 3D ノルムを使う。ループシュート
    (walk_loop) 用。仰角 30° で蹴ると水平成分は 3D ノルムの 0.87 倍しかないため、
    水平で測ったままだと「指令速度に届いていない」と誤判定され、φ 報酬（浮かせろ）と
    速度報酬（もっと強く）が恒常的に綱引きしてしまう。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    v_meas = state["v_ball_3d_frozen"] if use_3d_speed else state["v_ball_frozen"]
    v_err = v_meas - state["v_target"]
    f_vel = torch.exp(-((v_err / sigma_velocity) ** 2))
    return r_dir * f_vel


def kick_velocity_strong(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
) -> torch.Tensor:
    """項3. Kick Velocity Strong = r_direction * v_ball（生の速度）。shape: (N,)"""
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)
    return r_dir * state["v_ball_frozen"]


def kick_velocity_overshoot(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    margin: float = 0.2,
    overshoot_sat: float = 1.0,
) -> torch.Tensor:
    """項10. Kick Velocity Overshoot = clamp(v_ball − (v_target + margin), 0, sat)。

    負の重みで使う。shape: (N,)

    **非対称**であることが肝。``kick_velocity_scaled`` の Gaussian は速すぎも遅すぎも
    同じだけ減点するが、減点は「報酬が減る」だけなので、``kick_velocity_strong``
    (生の球速に比例・上限なし) が同時に居ると **蹴りすぎの方が黒字**になる。実際
    walk_kick_360 の重み配分では、指令 0.5 に対して:

    * 正しく蹴る (v=0.5):  scaled 1.2×1.00 + strong 0.9×0.5 ≈ 1.65
    * 蹴りすぎ  (v=1.5):  scaled 1.2×e^-1 + strong 0.9×1.5 ≈ 1.79   ← こちらが得

    この項は超過分**だけ**を直接罰するので、上の不等号を反転させられる。

    Args:
        margin: この量までの超過は無罰 [m/s]。latch の量子化 (閾値をまたいだ瞬間の値を
            採る) と接触モデルのばらつきぶんの遊び。
        overshoot_sat: 超過量の飽和値 [m/s]。**青天井にしないこと**。

            NOTE: この項は他の項1-3 と同じく post-latch に dense で払われるので、
                  猶予窓 (2.0 秒 = 100 step) ぶん累積する。RewardManager は
                  value * weight * dt を毎 step 払うので、1 エピソードの総額は
                  ``超過量 × weight × 2.0`` になる。負の dense 払いなので、
                  ``_r_direction`` の NOTE と同じ「外したら早く転んで損切り」の
                  抜け道が理屈上ありうる (転倒罰は -100 × dt = -2.0 の一度きり)。
                  飽和させて総額を転倒罰より小さく保つことでこれを塞ぐ:
                  weight = -2.0 × _KICK_W_SCALE = -0.6, sat = 1.0 なら最大 -1.2 で、
                  転んで止めるより払い切った方が得な範囲に収まる。
                  weight や窓を変えるときはこの不等式を必ず引き直すこと。

    NOTE: 凍結値 (``v_ball_frozen``) を使う。飛翔中の減速後の値ではなく、
          **latch した瞬間の射出速度**が指令と比べる対象。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    excess = state["v_ball_frozen"] - (state["v_target"] + margin)
    excess = torch.clamp(excess, min=0.0, max=overshoot_sat)
    return excess * state["kick_done"].float()


def kick_elevation(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    phi_target: float = 0.52,
    sigma_phi: float = 0.25,
    phi_sat: float | None = None,
) -> torch.Tensor:
    """項7. Loop Shot (ループシュート) = r_direction * f(φ)。shape: (N,)

    φ は latch 時に凍結したボール射出仰角。f(φ) には 2 つのモードがある。

    * **Gaussian** (既定): f(φ) = exp(−((φ − φ_target) / σ_φ)²)。
      φ_target = 0.52 rad ≈ 30°、σ_φ = 0.25 rad ≈ 14°。目標仰角の「帯」を狙わせる。
    * **片側飽和** (``phi_sat`` 指定時): f(φ) = clamp(φ / φ_sat, 0, 1)。
      φ_sat まで単調増加し、以降は頭打ち。**「できる限り高く」を狙わせる。**

    どちらを使うかは目標が物理上限からどれだけ離れているかで決まる。K1 の足で出せる
    射出仰角には明確な天井があり (足先上エッジ高 3.6cm / ボール半径 11cm から約 42°、
    しかも接触時に足が 2cm 浮くだけで 29° まで落ちる)、Gaussian で 30° を狙わせると
    ポリシーは「足を 2cm 浮かせた状態」に最適化してしまう。実機ではそこからさらに
    目減りするので浮きが消える。天井付近を狙うタスク (walk_loop_shoot) では
    片側飽和を使い、足をできる限り低く通す解に寄せること。

    **r_direction への乗算であることが設計の肝**。加算にすると「方向を無視してボールを
    真上に跳ね上げる」「踏んで潰す」で報酬が取れてしまう。乗算なら kick_done ゲート・
    方向精度 (τ_direction)・胴体の正対 (p_style) を全て通過した蹴りにしか払われない。
    さらに latch トリガー自体が水平速度基準なので (kick_state 参照)、「前に飛んでいる」
    ことも暗黙の前提条件になっている。

    NOTE: 「vz が大きいほど得」という青天井の項には **絶対にしないこと**。必ず踏みつけ
          スクープに収束する。片側飽和モードも φ_sat で頭打ちにすることでこれを防いで
          いる。飽和させずに単調増加させてはいけない。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    phi = state["phi_frozen"]
    if phi_sat is not None:
        # 打ち下ろし (φ<0) は 0。φ_sat 以上は 1 で頭打ち。
        f_phi = torch.clamp(phi / phi_sat, min=0.0, max=1.0)
    else:
        f_phi = torch.exp(-(((phi - phi_target) / sigma_phi) ** 2))
    return r_dir * f_phi


def kick_loft(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    vz_sat: float = 2.5,
) -> torch.Tensor:
    """項7'. Loft = r_direction * clamp(vz / vz_sat, 0, 1)。shape: (N,)

    vz = v_ball_3d_frozen * sin(φ_frozen)。頂点高さは vz²/2g で **vz だけで決まる** ので、
    「はっきり浮かせる」タスクの目的関数として角度 (kick_elevation) より正しい。

    kick_elevation は角度にしか興味がなく速度に無関心なため、「30° のゆるい蹴り」と
    「30° の強烈な蹴り」が同じ満点になる。walk_loop_shoot の実測でも φ≈1-2° の
    ほぼ水平な蹴りに収束した。vz を狙わせれば「強く」と「上に」を同時に要求できる。

    * 飽和 (vz_sat で頭打ち) なので青天井にならない。飽和は Gaussian と違い届かなくても
      勾配が死なない (線形ランプ) ので、vz_sat は狙いたい浮きの値をそのまま置けばよい。
      浮き [m] = vz²/2g なので、vz=2.5 で 0.32 m、vz=3.7 で 0.70 m (K1 の身長相当)。
    * r_direction への乗算・kick_done ゲート・打ち下ろし (φ<0 → sin<0 → clamp 0) の
      扱いは kick_elevation と同じ。踏みつけ exploit 対策の設計原則を維持する。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    vz = state["v_ball_3d_frozen"] * torch.sin(state["phi_frozen"])
    f_loft = torch.clamp(vz / vz_sat, min=0.0, max=1.0)
    return r_dir * f_loft


def kick_plant_foot(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    lon_target: float = -0.03,
    sigma_lon: float = 0.10,
    lat_target: float = 0.19,
    sigma_lat: float = 0.06,
) -> torch.Tensor:
    """項9. Plant Foot (軸足配置) = r_direction * f(lon) * f(lat)。shape: (N,)

    latch 時に凍結した軸足 (蹴っていない方の足) のボール相対位置を、キック方向フレームで
    評価する。狙いは **「蹴る瞬間に軸足がボールの真横に来ている」** こと。
    軸足がボールの後方にあると蹴り足はボールの向こう側の高い位置に当たるため、足裏を
    低く潜らせられず仰角が出ない (``sole_height_at_kick`` が下がらない)。

    * f(lon) = exp(−(lon − lon_target)² / 2σ_lon²) : 前後。lon は kick_dir 成分で + が前。
    * f(lat) = exp(−(lat − lat_target)² / 2σ_lat²) : 左右。lat は絶対値なので左右キック両対応。

    パラメータ既定値は K1 の実寸から決めてある (K1_22dof.xml):

    * ``lat_target = 0.19``: 股関節の横オフセットが ±0.096 なので通常スタンス幅が 0.192。
      軸足を横 0.19 に置くと蹴り足がちょうどキック線上 (横 0) に来る。無理な姿勢を
      要求しない自然な値。
    * ``sigma_lat = 0.06``: 下限側は衝突限界で決まる。ボール半径 0.11 + 足箱の半幅 0.035 =
      **0.145 より内側は軸足がボールに当たる**。0.19±0.06 なら 0.145 で f=0.66、
      0.12 まで入ると f=0.29 と十分冷たくなる。
    * ``lon_target = -0.03``: ``body_pos_w`` が返すのは足リンク原点 (= 足首) だが、足箱の
      中心はそこから前方 +0.026 にある。**足の中心をボール真横に置くには足首を −0.026**。
      チップ気味に「やや後ろ」へ寄せたければ −0.08 程度まで下げる。
    * ``sigma_lon = 0.10``: ±0.1 で半値。歩幅の分解能を考えるとこれ以上締めても追従できない。

    設計上の約束 (kick_elevation / kick_loft と同じ):

    * **r_direction への乗算**であること。加算にすると「方向を無視して軸足だけ置く」で
      報酬が取れてしまう。乗算なら kick_done ゲート・方向精度・胴体の正対を全て通過した
      蹴りにしか払われない。
    * **他のキック報酬とは加算で並べる**。``kick_loft`` に掛けてはいけない。学習初期は
      軸足配置がまず合わないので、掛けると loft の勾配がゼロ付近で死ぬ。
    * **非負** (罰にしない)。外した配置は「罰される」のではなく「報われない」に留める。
      負の dense 払いにすると、_r_direction の NOTE と同じ「外したら早く転んで損切り」の
      抜け道が復活する。

    NOTE: 軸足がボールに接触してしまう解は、この項の σ_lat に加えて
          :func:`extra_ball_touch` (2 回目以降の接触を罰する) が既に塞いでいる。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    f_lon = torch.exp(-((state["plant_lon_frozen"] - lon_target) ** 2) / (2.0 * sigma_lon**2))
    f_lat = torch.exp(-((state["plant_lat_frozen"] - lat_target) ** 2) / (2.0 * sigma_lat**2))
    return r_dir * f_lon * f_lat


def kick_foot_lift(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    vz_foot_sat: float = 2.0,
) -> torch.Tensor:
    """項11. Foot Lift (すくい上げ) = r_direction * clamp(foot_vz / vz_foot_sat, 0, 1)。shape: (N,)

    latch 時に凍結した **蹴り足の鉛直速度** (``foot_vz_frozen``、+ = 上向き) を評価する。
    狙いは **「ボールが浮くメカニズムを反発係数依存から運動学依存へ移す」** こと。

    walk_lob は Isaac Sim では浮くのに MuJoCo・実機では浮かない。原因はボールの反発係数で、
    Isaac の既定 (e≈0.6) では「足を水平に突っ込んでボールを地面との間で弾ませる」だけで
    vz が出てしまうのに対し、MuJoCo・実機 (e≈0) ではその成分が丸ごと消える。
    ``kick_loft`` / ``kick_elevation`` は **結果** (ボールの vz・仰角) だけを見るので、
    どちらの機構で浮いたかを区別できず、シミュレータ固有の解を選んでも満点が出る。

    この項は **原因側** (接触の瞬間に足自身が上へ動いているか) を直接報酬にする。
    足の上向き運動量から移る vz は反発係数に依存しないので、この項で誘導した解は
    e が消える環境でもそのまま残る。``kick_loft`` (結果) と並べて置くことで、
    「上げろ」と「すくい上げで上げろ」を同時に要求する形になる。

    * f_lift = clamp(foot_vz / vz_foot_sat, 0, 1)。**打ち下ろし (foot_vz < 0) は 0**。
      踏みつけ型の解にはこの項から一切払われない。
    * ``vz_foot_sat = 2.0`` [m/s] はボール vz 目標 (walk_lob の ``vz_sat`` = 5.0) に対して
      運動学的に必要な足速度の目安。剛体衝突では質量比と接触法線で伝達率が決まるので
      1:1 では移らないが、飽和型 (線形ランプ) なので厳密な値である必要はない。届かない
      値を置いても勾配は死なず、逆に飽和させると圧力が消える (``kick_loft`` と同じ)。
      実測は ``Metrics/kick_direction/foot_vz`` で見て、飽和しているようなら上げること。

    設計上の約束 (kick_loft / kick_plant_foot と同じ):

    * **r_direction への乗算**であること。加算にすると「方向を無視して足を上に振る」だけで
      報酬が取れてしまう。乗算なら kick_done ゲート・方向精度 (τ_direction)・胴体の正対
      (p_style) を全て通過した蹴りにしか払われない。``sigma_direction`` は同じタスクの
      他のキック報酬と **必ず同じ値** にすること (項ごとに違うと方位を外したときの損得が
      食い違って何を最適化しているのか読めなくなる)。
    * **他のキック報酬とは加算で並べる**。``kick_loft`` に掛けてはいけない。学習初期は
      すくい上げがまず出ないので、掛けると loft の勾配がゼロ付近で死ぬ。
    * **非負** (罰にしない)。すくい上げのない蹴りは「罰される」のではなく「報われない」に
      留める。負の dense 払いにすると、_r_direction の NOTE と同じ「外したら早く転んで
      損切り」の抜け道が復活する。
    * **青天井にしないこと**。飽和 (vz_foot_sat で頭打ち) が「足を全力で上へ振り抜く」
      だけの解を防いでいる (kick_elevation の NOTE と同じ原則)。
    """
    r_dir, state = _r_direction(env, r_stance, alpha, v_thresh, sigma_direction)

    f_lift = torch.clamp(state["foot_vz_frozen"] / vz_foot_sat, min=0.0, max=1.0)
    return r_dir * f_lift


def walk_speed(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_walk: float = 0.5,
    sigma_walk_potential: float = 0.5,
) -> torch.Tensor:
    """項4. Walk Speed = (f(τ_walk) − 0.5) * 2 * p_walk。shape: (N,)

    τ_walk はロボット速度の G 方向成分（符号付き）。f = sigmoid(τ_walk / σ) なので、
    G に向かって進めば正、遠ざかれば負になる。p_walk = exp(−d(robot, G) / σ_pot) は
    G への接近度 (1 = 到達)。G は P_kick で下限クランプされるので、キック立ち位置に
    着いた時点で p_walk が飽和し、この項は self-gate する。凍結しない。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    f_walk = torch.sigmoid(state["tau_walk"] / sigma_walk)
    p_walk = torch.exp(-state["d_to_G"] / sigma_walk_potential)
    return (f_walk - 0.5) * 2.0 * p_walk


def approach_penalty(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_sole: float = 0.35,
    sigma_pose: float = 0.3,
) -> torch.Tensor:
    """項5. Approach Penalty = f(d_soleToBall) * p_kickPose。負の重みで使う。shape: (N,)

    * f(d_soleToBall) = 1 − exp(−(d/σ)²) : 足裏がボールから **遠いほど大きい** (近い=0, 遠い=1)
    * p_kickPose                          : 理想キック姿勢からの **ズレほど大きい** (合致=0, ズレ=1)

    理想（足がボールに近い × 姿勢が P_kick と一致）で 0、最悪（遠い × ズレ）で最大の罰。
    pre-latch のみ有効（kick_done で 0 ゲート）。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    # 遠いほど 1 に近づく
    f_sole = 1.0 - torch.exp(-((state["d_sole_to_ball"] / sigma_sole) ** 2))

    # 理想キック姿勢 = P_kick に立ち、蹴り方向を向いている。合致度が高いほど 0 に近づく。
    pose_match = torch.exp(-((state["d_to_P_kick"] / sigma_pose) ** 2)) * state["p_style"]
    p_kick_pose = 1.0 - pose_match

    return f_sole * p_kick_pose * (~state["kick_done"]).float()


def ball_avoidance(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_sole: float = 0.35,
    sigma_pose: float = 0.3,
) -> torch.Tensor:
    """項5'. Ball Avoidance = f(d_soleToBall) * p_kickPose。負の重みで使う。shape: (N,)

    :func:`approach_penalty` の f_sole を反転した、B-Human 原義 (と推定される) 形。

    * f(d_soleToBall) = exp(−(d/σ)²) : 足裏がボールに **近いほど大きい** (近い=1, 遠い=0)
    * p_kickPose                      : 理想キック姿勢からの **ズレほど大きい**

    「構え (P_kick に立ち蹴り方向を向く) ができるまでボールに寄るな」という抑止。
    姿勢が合えば近づいても罰が消えるので、キック自体は妨げない。全方位サンプリング
    (ψ が大きい = ボールの正面側から接近するエピソード) では、まっすぐ突っ込んで
    足を当てる行動をこの項が直接罰し、誘導なしで回り込みを成立させる。

    approach_penalty (遠いほど罰 = 接近圧) とは逆向きであることに注意。原典ポスターの
    項名が "Ball Avoidance" であること、他の f() が全て 0 で最大の釣鐘型であること、
    著者の証言 (「逆向きに蹴れば報酬なし/ペナルティで大丈夫」) からこの向きと推定した。
    フルペーパーでの式の確認は取れていない。

    pre-latch のみ有効（kick_done で 0 ゲート）。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)

    # 近いほど 1 に近づく (approach_penalty と逆)
    f_sole = torch.exp(-((state["d_sole_to_ball"] / sigma_sole) ** 2))

    pose_match = torch.exp(-((state["d_to_P_kick"] / sigma_pose) ** 2)) * state["p_style"]
    p_kick_pose = 1.0 - pose_match

    return f_sole * p_kick_pose * (~state["kick_done"]).float()


def extra_ball_touch(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
) -> torch.Tensor:
    """項8. 2 回目以降のボール接触。発火したステップだけ 1。負の重みで使う。shape: (N,)

    「まず軽く触って、それから蹴る」を潰すための項。1 回目の接触は無料なので、
    ポリシーが罰を避ける唯一の道は **最初の接触をそのままキックにする** こと。

    NOTE: 現在のポリシーでは「2 回目の接触 = 本命のキック」なので、この項は一見
          キック本体を罰しているように見える。それで正しい。キック報酬 (項1+2 で
          dense 2 秒ぶん ≈ +5) の方が 1 回の余分な接触の罰より十分大きいので、
          「蹴らない」ではなく「1 回目で蹴る」方向に動くはず。逆に重みを上げすぎると
          ボールに触ること自体を避けて kick_rate が落ちるので、Metrics で監視すること。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    return state["extra_touch_event"]


def kick_pose_overshoot(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
) -> torch.Tensor:
    """項6. Kick Pose Overshoot。キック線 R を跨いだ瞬間だけ 1。負の重みで使う。shape: (N,)

    基準側 init_side は開始時の sign(s) ではなく、ロボットが |s| > 閾値 までどちらかの
    側へ寄った時点の符号で確定する (:mod:`.kick_state` の _INIT_SIDE_COMMIT_DIST 参照。
    s≈0 スタートで符号がノイズになり、正しい行動がコイントスで罰されるのを防ぐ)。
    確定側から反対側へ符号が反転したら発火して latch する。戻っても解除せず、
    1 エピソード最大 1 回だけ罰する。未確定の間は発火しない。
    """
    state = kick_state(env, r_stance=r_stance, alpha=alpha, v_thresh=v_thresh)
    return state["overshoot_event"]
