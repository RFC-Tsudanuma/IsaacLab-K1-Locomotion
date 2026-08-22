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

import math
import torch
from typing import TYPE_CHECKING

from ...locomotion.mdp.rewards import (
    feet_close_penalty as _loco_feet_close_penalty,
    knee_close_penalty as _loco_knee_close_penalty,
)
from .kick_state import _foot_body_ids, _foot_forward_xy, kick_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _r_direction(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項3. Kick Velocity Strong = r_direction * v_ball（生の速度）。shape: (N,)"""
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    return r_dir * state["v_ball_frozen"]


def kick_velocity_overshoot(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    margin: float = 0.2,
    overshoot_sat: float = 1.0,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    f_lift = torch.clamp(state["foot_vz_frozen"] / vz_foot_sat, min=0.0, max=1.0)
    return r_dir * f_lift


def kick_contact_height(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    ball_radius: float = 0.11,
    h_sat: float = 0.03,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項12. Contact Height (低い当たり) = r_direction * f_low。shape: (N,)

    latch 時に凍結した **蹴り足の足裏高さ** (``sole_height_at_kick`` [m]) を評価する。
    低いほど良い = ボールの下側に当てているほど良い。

    なぜ要るか (walk_lob 2026-08-16 の実測から)
    -------------------------------------------
    ``kick_apex_height`` 0.425 m はボール中心の絶対高さなので、静止時 0.11 m から
    上昇 0.315 m ⇔ 打ち出し vz ≈ 2.49 m/s。ところが同じ run の ``foot_vz`` は
    0.81 m/s しかない。**すくい上げ (足自身の鉛直速度) では説明が付かない** 量が出て
    いるということで、実際に浮きを作っているのは接触法線の向き — つまり
    「ボール中心より下に、速い水平速度で当てている」ことの方である。

    剛体・反発ゼロの衝突では、ボールは接触点からボール中心へ向かう法線方向に飛ぶ。
    足裏高さ h でボール (半径 R) に当てたときの法線仰角は asin((R − h) / R) なので、

        h = 0.083 (実測) → 14°     h = 0.055 → 30°
        h = 0.032        → 45°     h = 0.011 → 84°

    実測の射出仰角 25° はこの 14° に foot_vz のぶんが乗った値として整合する。
    **仰角を 45-60° まで持っていくには h を 0.03 m 台まで下げる必要がある**、
    というのがこの項の根拠。``kick_loft`` / ``kick_elevation`` は結果 (ボールの vz・
    仰角) しか見ないので「どこに当てて浮かせたか」を指定できず、``kick_foot_lift`` は
    別機構 (足の鉛直速度) を指定する項なので、接触点の高さはどの項からも直接の
    圧力を受けていなかった。

    ``kick_plant_foot`` (軸足を前へ) とは **同じことの表と裏** である。軸足がボールの
    後方に残ったまま蹴ると蹴り足は伸び切った姿勢でボールの向こう側の高い位置に届く
    ので、h は下がりようがない。あちらが原因側 (構え)、こちらが結果側 (当たり所) を
    直接押さえる。両方入れて構わない (walk_lob 系は実際に両方入れる)。

    f_low の形
    ----------
    ``f_low = clamp((ball_radius − h) / (ball_radius − h_sat), 0, 1)``

    * h ≥ ``ball_radius`` (ボール中心以上の高さに当てた) → 0。打ち下ろし気味の
      当たりにはこの項から一切払われない。
    * h ≤ ``h_sat`` → 1 で頭打ち。既定 0.03 は上の表の 45° 相当。
      **飽和させるのは「つま先を地面へめり込ませる」方向へ青天井に引かないため**
      (kick_elevation / kick_foot_lift と同じ原則)。h_sat より下げても得をしないので、
      地面を掻くだけの解に動機が生まれない。

    設計上の約束 (kick_plant_foot / kick_foot_lift と同じ):

    * **r_direction への乗算**であること。加算にすると「方向を無視して足を低く出す」
      だけで報酬が取れてしまう。乗算なら kick_done ゲート・方向精度 (τ_direction)・
      胴体の正対 (p_style) を全て通過した蹴りにしか払われない。``sigma_direction`` は
      同じタスクの他のキック報酬と **必ず同じ値** にすること。
    * **他のキック報酬とは加算で並べる**。``kick_loft`` に掛けてはいけない。学習初期は
      低い当たりがまず出ないので、掛けると loft の勾配がゼロ付近で死ぬ。
    * **非負** (罰にしない)。高い当たりは「罰される」のではなく「報われない」に留める。

    NOTE: ``sole_height_at_kick`` は **latch を起こした接触** (= キック本体) の
          足裏高さで、最初の接触ではない。多重接触があるときに蹴る前の偶発的な接触を
          拾わないための作りで、詳細は :mod:`.kick_state` の該当箇所を参照。

    NOTE: ``touch_count > 0`` でゲートしている。この項は他のキック報酬と違って
          **凍結値の初期値 0.0 が「満点」側に写る** (f_low(0) = 1) ので、接触が
          一度も記録されないまま latch した場合に払ってしまう。``kick_foot_lift``
          (foot_vz=0 → 0 点) や ``kick_plant_foot`` (目標から遠い → 0 点) は初期値が
          自然に無得点側なのでこのゲートが要らない。
          実際には latch のトリガー (ボール速度 > v_thresh) を満たす dv は同じ
          ステップの接触検出も満たすので通常は起きないが、未計測が満点になる向きの
          失敗モードは残さない。

    .. warning::
       **walk_lob でこの項は反証されている (2026-08-18)。現在どのタスクからも
       使われていない。** ``sole_height_at_kick`` は狙いどおり 0.062 → 0.050 に
       下がったが、**同じ run で apex は 0.340 → 0.234 に下がった**。低く当てるには
       立ち位置を詰めるしかなく、それがスイング長 = ボール速度を削るため。
       3 run を並べると apex 上昇 ∝ (ball_vel · sinφ)² にほぼ完全に乗り、仰角より
       ボール速度の変動の方が支配的だった (詳細は
       :mod:`~...walk_lob_rough.walk_lob_rough_env_cfg` のモジュール docstring)。
       **「浮かせたい」目的でこの項を足すときは、ボール速度が落ちていないかを
       ``kick_vel_ratio`` で必ず確認すること。** 上の「なぜ要るか」の推論は
       接触法線の幾何としては正しいが、速度とのトレードオフを勘定に入れていない。

    Args:
        ball_radius: ボール半径 [m]。0 点になる足裏高さ (= ボール中心の高さ)。
        h_sat: 満点になる足裏高さ [m]。既定 0.03 は法線仰角 45° 相当。
    """
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    span = max(ball_radius - h_sat, 1e-6)
    f_low = torch.clamp((ball_radius - state["sole_height_at_kick"]) / span, min=0.0, max=1.0)
    measured = (state["touch_count"] > 0.0).float()
    return r_dir * f_low * measured


def walk_speed(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_walk: float = 0.5,
    sigma_walk_potential: float = 0.5,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項4. Walk Speed = (f(τ_walk) − 0.5) * 2 * p_walk。shape: (N,)

    τ_walk はロボット速度の G 方向成分（符号付き）。f = sigmoid(τ_walk / σ) なので、
    G に向かって進めば正、遠ざかれば負になる。p_walk = exp(−d(robot, G) / σ_pot) は
    G への接近度 (1 = 到達)。G は P_kick で下限クランプされるので、キック立ち位置に
    着いた時点で p_walk が飽和し、この項は self-gate する。凍結しない。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項5. Approach Penalty = f(d_soleToBall) * p_kickPose。負の重みで使う。shape: (N,)

    * f(d_soleToBall) = 1 − exp(−(d/σ)²) : 足裏がボールから **遠いほど大きい** (近い=0, 遠い=1)
    * p_kickPose                          : 理想キック姿勢からの **ズレほど大きい** (合致=0, ズレ=1)

    理想（足がボールに近い × 姿勢が P_kick と一致）で 0、最悪（遠い × ズレ）で最大の罰。
    pre-latch のみ有効（kick_done で 0 ゲート）。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

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
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    # 近いほど 1 に近づく (approach_penalty と逆)
    f_sole = torch.exp(-((state["d_sole_to_ball"] / sigma_sole) ** 2))

    pose_match = torch.exp(-((state["d_to_P_kick"] / sigma_pose) ** 2)) * state["p_style"]
    p_kick_pose = 1.0 - pose_match

    return f_sole * p_kick_pose * (~state["kick_done"]).float()


def ball_avoidance_exec(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    d_contact: float = 0.18,
    d_sat: float = 0.45,
    sigma_pose: float = 0.3,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項5''. Ball Avoidance (execution 解釈) = f(d_mean) * p_kickPose。負の重みで使う。shape: (N,)

    B-Human ポスターの Ball Avoidance ``f(d_soleToBall)·p_kickPose`` (weight −3) を、
    :func:`approach_penalty` / :func:`ball_avoidance` とは **第 3 の向き** で読んだもの。
    ユーザーとの議論で確定した解釈 (2026-08-17):

    * ``f(d) = clamp((d − d_contact) / (d_sat − d_contact), 0, 1)``
      : **遠いほど大きい** (接触距離で厳密に 0、d_sat 以遠で 1)
    * ``p_kickPose = exp(−(d_to_P_kick/σ_pose)²) · p_style``
      : **構えの一致度** (1 = P_kick に立ち蹴り方向を向いている)。
      ``p_style`` / ``p_walk`` と同じ自然な極性で、``approach_penalty`` /
      :func:`ball_avoidance` が使う「ズレほど大きい」反転版ではない。

    積を負の重みで払うので、罰されるのは **「構えは完成しているのに足がボールから
    遠い」** 状態だけになる。名前どおりの「ボールを避けろ」ではなく、
    「構えたなら実行しろ (蹴り切れ)」という督促として効く。

    核心は **キック接触の瞬間に距離側が 0 になり罰が消える** こと。足がボールに触れる
    位置まで詰めれば f = 0 なので、構えが完璧でも罰は残らない。つまりこの項は
    「構えて止まったまま」だけを罰し、蹴り抜けた瞬間に自分で消える。

    d は **両足の平均** (:data:`~.kick_state` の ``d_sole_to_ball_mean``)
    ----------------------------------------------------------------------
    片足 min (``d_sole_to_ball``) だと「軸足を後ろに置いて蹴り足だけ突き出す」退行解が、
    綺麗なインサイドキック (両足ともボール近傍、平均 ≈ 0.17-0.20 m) と同じ値になり
    区別できない。平均なら退行解は ≈ 0.32 m で分離する。

    パラメータ
    ----------
    * ``d_contact = 0.18``: ボール半径 0.11 (中心 z = 0.11) と足リンク原点
      (接地時 z ≈ 0.038 = :data:`~.kick_state._SOLE_OFFSET`) の鉛直差 0.072 に、
      接触時の水平距離を足したもの。綺麗なインサイドの構えでの両足平均 ≈ 0.17-0.20 に
      当たる。ここで f が 0 に張り付くので「接触したら罰ゼロ」が成立する。
    * ``d_sat = 0.45``: 突き出し退行解 (平均 ≈ 0.32) で f ≈ 0.5、それ以遠は飽和。
      青天井にしないことで、遠方 (接近中) の罰が構えの一致度ぶんに抑えられる。

    ``f`` を線形クランプにしてあるのは、リポジトリ既存の ``f_phi`` / ``f_loft`` /
    ``f_lift`` (いずれも clamp 形式) と揃えるためと、Gaussian や exp では接触時に
    厳密な 0 にならないため。

    NOTE: 命名は「非負の値を返し、負の重みで使う」既存の罰項の規約
          (:func:`approach_penalty` / :func:`ball_avoidance` /
          :func:`kick_velocity_overshoot` / :func:`kick_pose_overshoot`) に従う。
    NOTE: pre-latch のみ有効 (kick_done で 0 ゲート)。latch 後はボールが飛んでいくので
          距離側が意味を失う。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    # 遠いほど 1。d_contact 以下 (= 接触している) で厳密に 0。
    f_sole = torch.clamp(
        (state["d_sole_to_ball_mean"] - d_contact) / (d_sat - d_contact), min=0.0, max=1.0
    )

    # 構えの一致度。1 = P_kick に立ち、蹴り方向を向いている (自然な極性)。
    pose_match = torch.exp(-((state["d_to_P_kick"] / sigma_pose) ** 2)) * state["p_style"]

    return f_sole * pose_match * (~state["kick_done"]).float()


def extra_ball_touch(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
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
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    return state["extra_touch_event"]


def kick_latch_bonus(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項9. Latch 後の定額ボーナス。post-latch の間ずっと 1。正の重みで使う。shape: (N,)

    目的: 「1 エピソード = 1 キック」構成が生む **長生きバイアス** の相殺
    ------------------------------------------------------------------
    このタスクは latch から ``kick_finished`` の ``delay_steps`` (既定 100 step = 2 秒)
    でエピソードを打ち切る。一方 dense な歩行系の正報酬 (feet_phase 等) は生きている間
    ずっと入るので、**蹴った瞬間に「残り時間ぶんの歩行収入」を没収される**。

    k1_walk_kick_ball_avoid の初回 run (iter 1600) で実測された経済:

    * 蹴らずに歩き続ける dense 収入 ≈ **+1.6 / 秒**
    * 蹴ると残り 4-5 秒 ≈ **+6〜8** を失う
    * 学習初期の蹴り 1 回の実収入 ≈ **+0.3** (方向・速度が未熟で満額 7.8 の 4%)

    差し引き「蹴る = 約 −6 の取引」で、キックは iter 300-400 に一度立ち上がった
    (kick_rate 0.19) 後 iter 1300 以降 0.00 に消滅した。その間 mean_reward は
    −4.4 → +16 と単調増加しており、勾配死ではなく **「蹴らない方が儲かる」を正しく
    学習した** 結果である。

    旧 :func:`approach_penalty` は「ボール近傍に居ないこと」への恒常税だったので、この
    バイアスを偶然相殺していた。:func:`ball_avoidance_exec` は「構えたときだけ課税」する
    ので、その仕事を引き継いでいない。そこで没収ぶんを **キックの成否によらない定額**
    で払い戻し、「蹴る/蹴らない」の選択を収支中立に戻す。方向・速度の巧拙は項1-3 が
    見るので、この項は意図的に品質を問わない。

    ``kick_done`` は latch からエピソード終了まで 1 を返し、RewardManager は
    ``value = func * weight * dt`` で払うので、1 キックあたりの総額は
    **``weight × delay_steps × dt``** になる (weight=4.0, 100 step, dt=0.02 なら +8)。

    NOTE: 総額が ``kick_finished`` の ``delay_steps``
          (``..walk_kick_env_cfg._KICK_DELAY_STEPS``) に比例するので、猶予窓を変えたら
          weight も見直すこと。項1-3 が ``_KICK_W_SCALE`` で自動的に割り戻されるのと
          違い、こちらは手動である。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    return state["kick_done"].float()


def kick_pose_overshoot(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項6. Kick Pose Overshoot。キック線 R を跨いだ瞬間だけ 1。負の重みで使う。shape: (N,)

    基準側 init_side は開始時の sign(s) ではなく、ロボットが |s| > 閾値 までどちらかの
    側へ寄った時点の符号で確定する (:mod:`.kick_state` の _INIT_SIDE_COMMIT_DIST 参照。
    s≈0 スタートで符号がノイズになり、正しい行動がコイントスで罰されるのを防ぐ)。
    確定側から反対側へ符号が反転したら発火して latch する。戻っても解除せず、
    1 エピソード最大 1 回だけ罰する。未確定の間は発火しない。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    return state["overshoot_event"]


# --------------------------------------------------------------------------- #
# インサイドキック (足の内側の面で当てる) 用の項
#
# 既存のキック報酬は「どこへ・どれだけ速く飛んだか」しか見ないので、当たり所は
# 自由だった。実測では 9/9 の run がトーキック (つま先で突く形) に収束している。
# つま先はボールとの接触面積が小さく、実機では当たり所のわずかなずれが方向誤差に
# 直結するので、面の広いインサイドで当てさせたい。
#
# 誘導の入口は 2 つあり、この 2 つで組になっている:
#   1. 逆風を外す  … p_style の帯 (:mod:`.kick_state` の ``style_halfwidth``)。
#      「胴体を蹴り方向へ正対させろ」= トーキックの構えの要求を、帯の中では消す。
#   2. 順風を入れる … :func:`kick_inside_contact`。接触の幾何 (足の向き・当たった面)
#      を直接採点する。
# --------------------------------------------------------------------------- #


def kick_inside_contact(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    d_sat: float = 0.34,
    side_margin: float = 0.0,
    side_sat: float = 0.035,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項13. Inside Contact (右足インサイドで当てたか) = r_direction * f_perp * f_side。shape: (N,)

    値 latch を起こした接触 (= キック本体) の幾何を凍結値で採点する。

    * ``f_perp = clamp((1 − |foot_kick_dot_frozen|) / (1 − d_sat), 0, 1)``
      : 蹴り足がキック方向に対してどれだけ **横を向いているか**。
      ``foot_kick_dot_frozen`` は足のローカル +x (つま先方向) の水平成分と kick_dir の
      内積なので、1 に近い = つま先が前 = トーキック、0 に近い = 足が真横 = 側面で
      当てている。
    * ``f_side = clamp((ball_side_frozen − side_margin) / side_sat, 0, 1)``
      : ボールが蹴り足の **ローカル +y 側 = 内側** へ、どれだけ深く入ったところで
      当たったか。右足のインサイド面が +y であることは K1_22dof.xml の foot body から
      確認済み (:mod:`.kick_state` の該当コメント参照)。|foot_kick_dot| だけでは
      インサイドとアウトサイドが区別できないので、この符号が要る。
      ``side_sat = 0.035`` は足箱の半幅 [m] で、ボール中心が足のローカル +y へ
      半幅ぶん入ったら満点。**0/1 の階段にしないこと** (下の失敗記録)。
    * ``right = (kick_foot_frozen == 1)``
      : 右足で蹴った latch だけ。kick_state の規約で 1 = 右。**このタスクは右足専用**
      (既存 run が 9/9 右足なので、左右両対応にして探索空間を倍にする理由がない)。
    * ``measured = (touch_count > 0)``
      : 一度も接触が記録されないまま latch した場合のガード。``kick_contact_height``
      とまったく同じ理由で、``foot_kick_dot_frozen`` の初期値 0 は f_perp = 1 =
      **満点側** に写ってしまう。未計測が満点になる向きの失敗モードは残さない。

    なぜ f_perp も f_side も線形クランプなのか (Gaussian・階段にしないこと)
    ----------------------------------------------------------------------
    ``d_sat = 0.34`` は cos70°。足がキック方向から 70° 以上横を向いた当たりで満点、
    そこから連続に落ちて、真正面 (|dot| = 1 = 完全なトーキック) でようやく 0 になる。
    **全域で勾配が残る**のが肝。

    Gaussian にすると、σ の外にいる方策には勾配がまったく無くなる。walk_lob 系 5 run の
    ``kick_plant_foot`` (f_lon が Gaussian) で実際にこれが起きていて、plant_lon は
    −0.36〜−0.43 に居座ったまま一度も目標へ寄らず、σ を広げた版でも gap が開いただけ
    だった。学習の出発点はトーキック (|dot| ≈ 1) の側なので、その位置で値も勾配も
    死んでいる形にしてはいけない。

    f_side も **まったく同じ理由**で線形。0/1 の階段は「境界の手前では動かしても
    報酬が 1 mm も変わらない」形なので、Gaussian の裾と同じく勾配が死ぬ。線形なら
    ボールが足の外側寄り (ball_side が小さい) で当たっている段階から
    「もう少し内側で当てる」方向に値が増え続ける。

    失敗記録: f_side を階段にしていた初回 run (2026-08-21_03-49-24)
    --------------------------------------------------------------
    初回の実装は ``f_side = (ball_side_frozen > side_margin)`` の 0/1 だった。
    キック自体は健全 (kick_rate 0.99、拡大ゲートも歩行も崩れず) だったのに、
    ``kick_inside_contact`` を ``kick_direction`` と右足率で割り戻した
    「内側で当たった割合」は iter 320 の 0.073 から iter 997 の 0.036 へ **半減** し、
    トーキックの方へ戻っていった。当たり所を内側へ寄せる方向の勾配がそもそも
    存在しなかったのだから当然で、f_perp には plant_lon の教訓を適用したのに
    f_side に同じ教訓を適用し損ねたのが原因。

    設計上の約束 (``kick_plant_foot`` / ``kick_contact_height`` と同じ)
    ------------------------------------------------------------------
    * **r_direction への乗算**であること。加算にすると「方向を無視して足を横に向ける」
      だけで報酬が取れてしまう。乗算なら kick_done ゲート・方向精度 (τ_direction)・
      胴体の向き (p_style) を全て通過した蹴りにしか払われない。``sigma_direction`` は
      同じタスクの他のキック報酬と **必ず同じ値** にすること。
    * **他のキック報酬とは加算で並べる** (乗算にしない)。学習初期はインサイドがまず
      出ないので、他項に掛けるとそちらの勾配がゼロ付近で死ぬ。
    * **非負** (罰にしない)。トーキックは「罰される」のではなく「報われない」に留める。
      負の dense 払いにすると ``_r_direction`` の NOTE と同じ「外したら早く転んで
      損切り」の抜け道が復活する。

    TODO: 発見が立ち上がった後に「インサイドで当たっていない蹴りは他のキック報酬も
          減らす」形 (f_perp を r_direction 側のゲートに昇格させる) へ進めたくなったら、
          この項とは別に gate 用の係数関数を足すこと。エスカレーションであって、
          最初から掛けてはいけない (上の「加算で並べる」と同じ理由)。

    NOTE: この項を使っているタスクは **walk_inside_kick だけ** (grep 済み。
          ``_KICK_STATE_REWARD_TERMS`` に名前があるのはパラメータ配布先の名簿で、
          項そのものを足しているのは walk_inside_kick の cfg のみ)。したがって
          f_side の形を変えても他タスクの学習は 1 ステップも変わらない。

    Args:
        d_sat: f_perp が満点になる |foot_kick_dot| の上限。既定 0.34 = cos70°。
        side_margin: f_side の坂が始まる ball_side の値 [m]。0.0 (既定) で
            「足のローカル原点より +y 側に入った時点から加点が始まる」。
            当たり面をもっと内側に限りたければ正の値を入れる。
        side_sat: f_side が満点になるまでの幅 [m]。既定 0.035 = 足箱の半幅で、
            ボール中心が足のローカル +y へ半幅ぶん入ったら 1.0。
            狭くするほど階段に近づく = 勾配が痩せるので、下げるときは
            上の失敗記録を読んでから。
    """
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    span = max(1.0 - d_sat, 1e-6)
    f_perp = torch.clamp((1.0 - state["foot_kick_dot_frozen"].abs()) / span, min=0.0, max=1.0)
    side_span = max(side_sat, 1e-6)
    f_side = torch.clamp((state["ball_side_frozen"] - side_margin) / side_span, min=0.0, max=1.0)
    right = (state["kick_foot_frozen"] == 1.0).float()
    measured = (state["touch_count"] > 0.0).float()

    return r_dir * f_perp * f_side * right * measured


def inside_foot_orient(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_dist: float = 0.3,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項14. Inside Foot Orientation (latch 前の足の向き) 。shape: (N,)

    ``exp(−(d / sigma_dist)²) × (1 − |右足前方向 · kick_dir|)`` を latch 前だけ払う。
    d は右足リンクとボール中心の距離 [m]。

    :func:`kick_inside_contact` は「当たった瞬間」しか見ないので、そもそもインサイドで
    当たる接触が一度も起きなければ勾配が 1 度も立たない。この項はその発見期の呼び水で、
    **接触していなくても、ボールの近くで足が横を向いていれば払う**。

    .. warning::
       **農作 (reward farming) のリスクがある。** 蹴らずにボールの脇で足を横に向けた
       まま滞在すると、この項だけが毎ステップ入り続ける。kick_done ゲートは掛けてある
       ので latch 後は消えるが、**latch させないこと自体が得**という方向の圧力になる。
       このリポジトリの他のキック項が全て「latch を通った蹴りにしか払わない」形に
       してあるのは同じ穴を避けるためで、この項だけがその原則を破っている。

       したがって既定では **weight 0 で寝かせておく**。``kick_inside_contact`` が
       数千 iteration 立ち上がらず、``Metrics/kick_direction/kick_rate`` は健全なのに
       インサイドの当たりがまったく出ない、という状況を確認してから、人間が小さい
       weight で有効化すること。有効にしたら ``kick_rate`` と ``time_out`` の割合を
       必ず並べて見る (滞在解に落ちると kick_rate が下がり time_out が増える)。

    NOTE: 右足だけを見る (:func:`kick_inside_contact` と揃えて右足専用)。
    NOTE: 凍結値ではなく現在値を使うので、kick_state の状態には触らずに
          ロボット・ボール・コマンドから直接測る。kick_state は kick_done ゲートと
          パラメータの一貫性 (同じ step の状態を共有すること) のためだけに呼ぶ。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    robot = env.scene["robot"]
    ball = env.scene["soccer_ball"]
    # _foot_body_ids は [left, right] の順 (kick_state の規約)。
    right_id = _foot_body_ids(env, robot)[1]

    right_pos = robot.data.body_pos_w[:, right_id, :]
    d = (right_pos - ball.data.root_pos_w[:, :3]).norm(dim=-1)

    foot_forward = _foot_forward_xy(robot.data.body_quat_w[:, right_id, :])

    cmd = env.command_manager.get_command("kick_direction")
    kick_dir = torch.stack([cmd[:, 1], cmd[:, 0]], dim=-1)

    f_near = torch.exp(-((d / sigma_dist) ** 2))
    f_perp = 1.0 - (foot_forward * kick_dir).sum(dim=-1).abs()

    return f_near * f_perp * (~state["kick_done"]).float()


def kick_plant_lon(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    lon_target: float = -0.03,
    lon_span: float = 0.45,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項15. Plant Longitudinal (軸足の前後位置だけを線形で誘導) = r_direction * f_lon。shape: (N,)

    latch 時に凍結した軸足 (蹴っていない方の足) の **前後位置** を、キック方向フレームで
    採点する。``plant_lon_frozen`` は ``(軸足 − ボール) · kick_dir`` なので
    **+ = 軸足がボールより前**、負が「軸足がボールより手前 (後ろ)」。

    狙いは巻き込み事故の防止である。実機のインサイドキック (run 2026-08-21_05-00-22、
    foot_kick_dot ≈ 0 / ball_side 0.145 / vel_ratio 0.91 = インサイドの型としては
    成立していた) で「振りが手前すぎてボールを巻き込んで転ぶ」事故が出た。軸足が
    ボールより後ろにあると振り足もボールの手前側を通るので、ボール認識が 3cm ずれた
    だけでボールが足の内側 = 進行方向に残り、そこへ踏み込んで転ぶ。軸足がボールの
    真横にあれば、同じ量ずれてもボールは体の横を抜けていく。

    * ``f_lon = clamp(1 − |plant_lon_frozen − lon_target| / lon_span, 0, 1)``
      : 目標を頂点、半幅 ``lon_span`` で 0 に落ちる **線形のテント**。
    * ``lon_target = -0.03`` [m]: ``body_pos_w`` が返すのは足リンク原点 (= 足首) だが、
      足箱の中心はそこから前方 +0.026 にある。**足の中心をボール真横に置くには
      足首を −0.026**。:func:`kick_plant_foot` とまったく同じ導出で、目標そのものは
      あちらと同じ値。
    * ``lon_span = 0.45`` [m]: 下の「線形テントにする理由」参照。

    なぜ線形テントで、Gaussian にしてはいけないのか
    ----------------------------------------------
    軸足を報酬で誘導する試みは :func:`kick_plant_foot` (f_lon が σ_lon = 0.10 の
    Gaussian) が先行しており、walk_lob 系 5 run で反証されている。ただし **死因は
    特定できている**: Gaussian の σ の外では値も勾配もゼロになる、という形の問題である。
    当時の実測 plant_lon ≈ −0.42 に対し、σ = 0.10 の Gaussian は
    ``exp(−0.39² / (2·0.10²)) ≈ 5e-4``。ポリシーが立っている場所で報酬関数が真っ平ら
    だったのだから、動かないのは当然だった。

    同じタスクの :func:`kick_inside_contact` の f_perp (線形クランプ・発見期から満額) は
    ``foot_kick_dot`` を 0.9 → 0.0 まで動かし切っている。線形なら遠いところでも
    傾きが一定で残るからで、この項もその流儀に合わせる。

    半幅 0.45 での各基準点の値 (これが「勾配が生きている」の中身):

    * plant_lon = −0.42 (lob / middle の実測 = 発見期の最悪値) → f_lon = 0.13
    * plant_lon = −0.23 (inside の収束値、run 2026-08-21_05-00-22) → f_lon = 0.56
    * plant_lon = −0.03 (目標 = 足箱の中心がボール真横) → f_lon = 1.00

    半幅を詰めるほど「正解の近くだけ急」になるが、その代償として発見期に居る
    −0.4 付近が 0 に潰れる。0.45 は **最悪値でもまだ 1 割強が残る** 最小限の広さ。

    なぜ r_stance を詰めるのではなく、足を測る項を足すのか
    ------------------------------------------------------
    先に「報酬を足さずに指令側だけ動かす」を試して失敗している。
    ``r_stance`` (P_kick = 胴体の終着指令をボール後方どれだけに置くか) を
    0.20 → 0.10 に詰め、model_4999 から ``--resume`` で 6700 iteration の
    fine-tune を回した (run 2026-08-21_10-41-17)。結果 **plant_lon は −0.23 のまま
    完全に不動** (他の指標は無傷: vel_ratio 0.94 / dir_error 4.2° / touch 1.03)。

    理由は報酬の流れを追うと分かる。P_kick は報酬にほとんど流れていない:

    * キック報酬 (項1-3 とその乗算項) は **飛翔の凍結値** で採点されるので、
      胴体がどこに立っていたかを一切見ない。
    * ``p_style`` はこのタスクでは帯 (``style_halfwidth`` = 40°) にしてあるので、
      帯の中では胴体の向きも報酬に効かない。
    * P_kick へ体を寄せる圧は ``ball_avoidance`` の pose_match 経由だけで、実測は
      −0.01 / episode しかない。

    過去に見えていた「r_stance → plant_lon」の相関 (0.25 → −0.39、0.20 → −0.23) は
    **ゼロから学習した run 同士の比較**であって、発見期に接近の誘導が型を形作った
    結果である。収束済みポリシーを動かすレバーではない。

    そこで設計原則を分ける: **``r_stance`` は胴体の終着指令であり、その役割は
    「望む構えを妨害しないこと」まで。軸足の誘導は足そのものを測る別の項が持つ。**
    この項がその「別の項」で、``r_stance`` は 0.10 のまま触らない。

    なぜ未計測ガードが要らないのか
    ------------------------------
    :func:`kick_inside_contact` は ``touch_count > 0`` のガードを持つが、この項は
    要らない。pre-latch は ``p_style_frozen`` が 0 なので ``r_direction`` が 0 になり
    (:func:`_r_direction` の kick_done ゲートも掛かる)、そもそも 1 円も払われない。
    latch 後は ``plant_lon_frozen`` が必ず **その蹴りの実測値** で、初期値 0 が
    残っている状態は存在しない (:mod:`.kick_state` が trigger と同じステップで
    現在値を凍結する)。「未計測が満点側に写る」失敗モードがそもそも無い。

    なぜ lat (横) を混ぜないのか
    ----------------------------
    :func:`kick_plant_foot` は f_lon と f_lat の積だったが、この項は **lon だけ**。
    inside の実測 ``plant_lat`` は 0.30 (通常スタンス幅 0.192 よりかなり広い) で、
    これはこれで別の課題だが、いま塞ぎたいのは巻き込み事故であって、その原因は
    前後方向だけである。lat を掛けると「横が外れているあいだ lon の勾配も死ぬ」
    という :func:`kick_plant_foot` の失敗をもう一度作ることになる。
    横を誘導したくなったら、この項に掛けるのではなく **別の項を加算で並べる**こと。

    設計上の約束 (:func:`kick_inside_contact` / :func:`kick_plant_foot` と同じ)
    --------------------------------------------------------------------------
    * **r_direction への乗算**であること。加算にすると「方向を無視して軸足だけ前に
      置く」で報酬が取れてしまう。乗算なら kick_done ゲート・方向精度 (τ_direction)・
      胴体の向き (帯つき p_style) を全て通過した蹴りにしか払われない。
      ``sigma_direction`` は同じタスクの他のキック項と **必ず同じ値** にすること。
    * **他のキック報酬とは加算で並べる** (乗算にしない)。学習初期は軸足がまず合わない
      ので、他項に掛けるとそちらの勾配がゼロ付近で死ぬ。
    * **非負** (罰にしない)。手前すぎる軸足は「罰される」のではなく「報われない」に
      留める。負の dense 払いにすると :func:`_r_direction` の NOTE と同じ
      「外したら早く転んで損切り」の抜け道が復活する。

    NOTE: この項を使っているタスクは **walk_inside_kick だけ**。
          ``_KICK_STATE_REWARD_TERMS`` に名前があるのはパラメータ配布先の名簿で、
          項そのものを足しているのは walk_inside_kick の cfg のみ (他タスクの cfg には
          属性が無いので None ガードで飛ばされる)。

    Args:
        lon_target: 軸足の目標前後位置 [m] (ボール基準、+ が前)。既定 −0.03 で
            足箱の中心がボールの真横。もっと前に踏み込ませたければ 0 側へ、
            チップ気味に後ろへ残したければ下げる。
        lon_span: テントの半幅 [m]。目標からこの距離離れると 0 点。
            **狭くするほど発見期の位置 (−0.4 付近) が 0 に潰れて勾配が死ぬ**ので、
            下げるときは上の「線形テントにする理由」の表を読んでから。
    """
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    span = max(lon_span, 1e-6)
    gap = (state["plant_lon_frozen"] - lon_target).abs()
    f_lon = torch.clamp(1.0 - gap / span, min=0.0, max=1.0)
    return r_dir * f_lon


def kick_plant_grounded(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項17. Plant Grounded (蹴る瞬間に軸足が接地しているか) = r_direction * plant_contact。shape: (N,)

    latch したステップの **軸足 (蹴っていない方の足) の法線接触力** を、片足立ちぶんの
    荷重 (0.5 · m · g) で正規化して 0-1 にしたもの
    (:func:`~.kick_state._plant_contact_normalized`)。1 = 軸足にちゃんと体重が
    乗っている、0 = 軸足も浮いている = **跳びながら蹴っている**。

    狙いは実機で観測された「蹴る瞬間に跳ぶ / 蹴った直後に跳ねる」の抑制である。
    跳んで蹴ると、地面反力を使えないぶん蹴り足の運動量だけでボールを打つことになり、
    威力が出ないうえ着地が乱れる。罰で潰すのではなく **接地している蹴りに上乗せで
    払う** ことで、「跳ねない蹴り」を正解側に置く。

    設計上の約束は :func:`kick_plant_lon` / :func:`kick_plant_yaw` と同じ:

    * **r_direction への乗算**。加算だと「蹴らずに両足で立っている」で報酬が取れる。
      乗算なら kick_done ゲート・方向精度・胴体の向きを通過した蹴りにしか払われない。
      ``sigma_direction`` は同じタスクの他のキック項と **必ず同じ値** にすること。
    * **他のキック報酬とは加算で並べる** (乗算にしない)。
    * **非負** (罰にしない)。跳んだ蹴りは「罰される」のではなく「報われない」に留める。
      負の dense 払いにすると :func:`_r_direction` の NOTE と同じ「外したら早く転んで
      損切り」の抜け道が復活する。
    * **線形** (Gaussian にしない)。0-1 に正規化済みの量をそのまま掛けるので、
      跳んでいる側 (0 付近) でも勾配が一定で残る (:func:`kick_plant_lon` の
      「線形テントにする理由」と同じ)。

    未計測ガードが要らない理由も :func:`kick_plant_lon` と同じ。pre-latch は
    ``p_style_frozen`` が 0 で ``r_direction`` が 0 になるので 1 円も払われず、
    latch 後は ``plant_contact_frozen`` が必ずその蹴りの実測値になっている。

    NOTE: ``contact_forces`` センサから足の body index を解決できない構成では
          ``plant_contact_frozen`` が常に 0 になり、この項は常に 0 を返す
          (:func:`~.kick_state._plant_contact_ids` が起動時に一度だけ警告を出す)。
          「効いていない」と「値が 0」を取り違えないよう、Metrics の
          ``plant_contact`` (log_contact_geometry=True で出る) を必ず並べて見ること。
    """
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    return r_dir * state["plant_contact_frozen"]


def kick_plant_yaw(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    yaw_span: float = math.pi / 2,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項16. Plant Yaw (軸足の **向き** を線形で誘導) = r_direction * f_yaw。shape: (N,)

    latch 時に凍結した軸足 (蹴っていない方の足) の **つま先方向** を採点する。
    ``plant_yaw_dot_frozen`` は「軸足のつま先方向 (上から見た水平成分) · kick_dir」で、
    1 = つま先が蹴り方向、0 = 真横、−1 = かかとが蹴り方向。

    :func:`kick_plant_lon` が軸足の **位置** (前後) を誘導するのに対し、この項は
    同じ軸足の **向き** を誘導する。位置と向きは独立に外れるので項も分ける。

    なぜ軸足の向きを誘導したいのか (実機フィードバック 2 回目)
    ----------------------------------------------------------
    軸足のつま先が蹴り方向を向いていると、骨盤ごと蹴り方向へ正対する側へ寄るので、
    振り足の面 (インサイド面) がキック線に正対しやすくなる。軸足が蹴り方向から
    大きくずれた向きで立つと、振り足はその骨盤の向きに引きずられて斜めに入るので、
    **当たりが薄くなる / 空振りする**。実機で実際に出ている失敗がこれである。

    胴体 (p_style) の側は帯 (``style_halfwidth`` = 40°) にしてあり、インサイドの
    自然な構え = 蹴り方向から 30-45° ずれた向きを **積極的に許している**
    (:mod:`~...walk_inside_kick.walk_inside_kick_env_cfg` の「逆風を外す」節)。
    この項はそれと矛盾しない: **胴体はずれてよいが軸足は蹴り方向を向かせる**、
    という役割分担である。骨盤と足首の間に Hip_Yaw があるので両立できる。
    しかも walk_inside_kick は ``joint_deviation_hip`` の対象から ``.*_Hip_Yaw`` を
    外してある (Hip_Yaw を初期値 0 へ引き戻す圧が無い) ので、軸足のヨーは
    ポリシーが自由に使える自由度になっている。

    なぜ cos ではなく **角度** に対して線形なのか
    --------------------------------------------
    ``theta = acos(clamp(plant_yaw_dot_frozen, -1, 1))`` [rad] を取ってから

        ``f_yaw = clamp(1 − theta / yaw_span, 0, 1)``

    と、**角度に対する線形テント** (頂点 theta = 0、``yaw_span`` で 0) にする。
    ``plant_yaw_dot`` (= cos theta) にそのまま線形クランプを掛けると、角度で見たとき
    非線形になる。特に **0° 付近が真っ平ら** になるのが困る (cos の微分は 0 で 0)。
    たとえば 0° → 20° のずれで cos は 1.00 → 0.94 しか動かないので、「あと 20° 直す」
    という改善に対して報酬がほとんど動かない。同じ 20° でも 70° → 90° では
    cos が 0.34 → 0.00 と大きく動く。**直したい領域 (目標の近く) ほど勾配が細る**
    という、報酬の形として最悪の向きの歪みになる。角度で線形にすれば、どこにいても
    「1° 直せば同じだけ得」になる。

    既定 ``yaw_span = π/2`` (90°) の意味は「真横 (90°) で 0 点、そこから
    つま先が蹴り方向へ向くほど直線的に増える」。90° より外 (かかと側) は 0 で
    飽和する。90° を境にするのは、そこから先が「軸足を後ろ向きに置く」= 誘導したい
    構えの反対側だからで、そこに勾配を残しても向かせたい先が無い。

    なぜ未計測ガードが要らないのか
    ------------------------------
    :func:`kick_plant_lon` とまったく同じ理由。pre-latch は ``p_style_frozen`` が 0 で
    ``r_direction`` が 0 になる (:func:`_r_direction` の kick_done ゲートも掛かる) ので
    1 円も払われず、latch 後は ``plant_yaw_dot_frozen`` が必ずその蹴りの実測値になる
    (:mod:`.kick_state` が trigger と同じステップの現在値を凍結する)。初期値 0 が
    残ったまま払われる状態が存在しないので、「未計測が満点側に写る」失敗モードが無い。
    なお初期値 0 は theta = 90° = f_yaw 0 なので、仮に残っても無得点側である。

    設計上の約束 (:func:`kick_plant_lon` と同じ)
    --------------------------------------------
    * **r_direction への乗算**であること。加算にすると「方向を無視して軸足の向きだけ
      揃える」で報酬が取れてしまう。乗算なら kick_done ゲート・方向精度
      (τ_direction)・胴体の向き (帯つき p_style) を全て通過した蹴りにしか払われない。
      ``sigma_direction`` は同じタスクの他のキック項と **必ず同じ値** にすること。
    * **他のキック報酬とは加算で並べる** (乗算にしない)。学習初期は軸足の向きがまず
      合わないので、他項に掛けるとそちらの勾配がゼロ付近で死ぬ。
    * **非負** (罰にしない)。ずれた軸足は「罰される」のではなく「報われない」に
      留める。負の dense 払いにすると :func:`_r_direction` の NOTE と同じ
      「外したら早く転んで損切り」の抜け道が復活する。
    * **線形テント。Gaussian にしないこと。** σ の外で勾配が完全に死ぬ失敗は
      :func:`kick_plant_foot` で実証済み (:func:`kick_plant_lon` の docstring)。

    NOTE: この項を使っているタスクは **walk_inside_kick だけ**。
          ``_KICK_STATE_REWARD_TERMS`` に名前があるのはパラメータ配布先の名簿で、
          項そのものを足しているのは walk_inside_kick の cfg のみ (他タスクの cfg には
          属性が無いので None ガードで飛ばされる)。

    Args:
        yaw_span: テントの半幅 [rad]。軸足のつま先が蹴り方向からこの角度だけ
            ずれると 0 点。既定 π/2 = 90° (真横)。**狭くするほど、いま居る場所が
            0 に潰れて勾配が死ぬ**ので、絞るときは
            ``Metrics/kick_direction/plant_yaw_dot`` の実測を見てから。
    """
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    span = max(yaw_span, 1e-6)
    # clamp は acos の定義域外 (数値誤差で |dot| がわずかに 1 を超える) 対策。
    theta = torch.acos(torch.clamp(state["plant_yaw_dot_frozen"], min=-1.0, max=1.0))
    f_yaw = torch.clamp(1.0 - theta / span, min=0.0, max=1.0)
    return r_dir * f_yaw


def kick_foot_ceiling(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    sigma_direction: float = 0.35,
    h_cap: float = 0.09,
    h_span: float = 0.08,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """項17. Foot Ceiling (足を上げすぎない **上限だけ**) = r_direction * f_cap。shape: (N,)

    latch 時に凍結した蹴り足の足裏高さ (``sole_height_at_kick`` [m]) に、
    **天井だけ** を課す:

        ``f_cap = clamp(1 − (h − h_cap) / h_span, 0, 1)``

    * ``h ≤ h_cap`` → **f_cap = 1 で満点**。それより低くしても 1 円も増えない。
    * ``h_cap < h < h_cap + h_span`` → 線形に減る。
    * ``h ≥ h_cap + h_span`` → 0。

    :func:`kick_contact_height` との違い (こちらは「上限だけ」)
    ----------------------------------------------------------
    足裏高さを見る項は既に :func:`kick_contact_height` があるが、あちらは
    ``f_low = clamp((R − h) / (R − h_sat), 0, 1)`` = **低いほど得** という形で、
    h_sat (0.03) まで下げ続ける圧を掛ける。**この形は walk_lob で反証されている
    (2026-08-18)**: ``sole_height_at_kick`` は狙いどおり 0.062 → 0.050 に下がったが、
    同じ run で apex が 0.340 → 0.234 に落ちた。低く当てるには立ち位置を詰めるしか
    なく、それがスイング長 = ボール速度を削るためである。

    この項はその失敗を繰り返さないために **下向きの圧を一切持たない**。
    ``h_cap`` 以下はすべて満点なので、「もっと低く」の勾配がそもそも存在しない。
    採りたいのは「上げすぎない」だけで、どこまで低くするかはポリシーの自由に任せる。
    したがって :func:`kick_contact_height` の代わりではなく、**別の目的の項**である
    (両方を同時に入れる理由は無い。入れれば h_cap 以下で f_low の下向きの圧だけが
    残り、結局あちらの反証済みの形になる)。

    ``h_cap = 0.09`` の根拠 (足箱の中心がボール中心以下に来る高さ)
    --------------------------------------------------------------
    足コライダーは足リンク原点から見て z = −0.038 (足裏) 〜 −0.002 (上面) の厚み
    0.036 の箱 (:data:`~.kick_state._SOLE_OFFSET` のコメント)。足裏高さ h のとき
    足箱の中心は h + 0.018 にある。ボール中心は静止時 0.11 (半径) なので、
    **足箱の中心がボール中心以下** = h + 0.018 ≤ 0.11 ⇔ h ≤ 0.092。
    0.09 はそのすぐ内側の丸めた値。これより上で当てると足の重心がボール中心より
    高いところを通ることになり、ボールを上から押さえる / 空振りして足がボールの
    上を越える形になる。

    ``h_span = 0.08`` は 0 点になる高さを h = 0.17 に置く幅。ボール直径 0.22 の
    おおよそ 3/4 で、「足がボールの上半分にしか当たっていない」領域を 0 に落とす。
    span を狭くするほど天井の壁が急になるが、急にすると天井の外に居るあいだ勾配が
    死ぬ (:func:`kick_plant_foot` の死因) ので、いま居る高さからの傾きが残る幅を取る。

    設計上の約束
    ------------
    * **r_direction への乗算**・**他のキック報酬とは加算**・**非負** — 理由は
      :func:`kick_plant_lon` の同名の節と同じ。特に「上げすぎたら罰」にしない
      (負の dense 払いは :func:`_r_direction` の NOTE の損切り exploit を復活させる)。
    * ``touch_count > 0`` のゲートが **要る**。この項は凍結値の初期値 0.0 が
      f_cap(0) = 1 = 満点側に写るので、接触が一度も記録されないまま latch した場合に
      払ってしまう。:func:`kick_contact_height` とまったく同じ理由で、同じゲートを掛ける
      (:func:`kick_plant_lon` / :func:`kick_plant_yaw` が要らないのは、あちらの
      凍結値が latch と同じステップの現在値で必ず埋まるため)。

    Args:
        h_cap: これ以下なら満点になる足裏高さ [m]。既定 0.09 = 足箱の中心が
            ボール中心 (0.11) 以下に来る高さ。
        h_span: 天井を越えてから 0 点になるまでの幅 [m]。既定 0.08 (h = 0.17 で 0)。
    """
    r_dir, state = _r_direction(
        env,
        r_stance,
        alpha,
        v_thresh,
        sigma_direction,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )

    span = max(h_span, 1e-6)
    f_cap = torch.clamp(1.0 - (state["sole_height_at_kick"] - h_cap) / span, min=0.0, max=1.0)
    measured = (state["touch_count"] > 0.0).float()
    return r_dir * f_cap * measured


# --------------------------------------------------------------------------- #
# 脚同士の接近ペナルティの「キック中だけ緩める」版
#
# locomotion 側の feet_close_penalty (足首・二値・しきい値 0.14) と
# knee_close_penalty (膝・連続・しきい値 0.13) は、``enabled_self_collisions=False``
# で sim では脚がすり抜けてしまうぶんを報酬で補う項。無いと交差する歩容を獲得して
# MuJoCo / 実機で実際に脚がぶつかる。
#
# ところがインサイドキックは「軸足をキック線のすぐ脇まで寄せて、蹴り足を体の内側から
# 外へ振る」動きなので、接触の前後で左右の脚が普段の歩行より確実に近づく。しきい値を
# 歩行のまま置くと、**インサイドの構えそのものが罰される**。
#
# そこで「ボールが近い env だけ」しきい値を緩める。
#   * 緩めるだけで **無効化はしない**。膝コライダーの直径は 0.09、足箱の幅は 0.07
#     (半幅 0.035) なので、0.10 はまだ物理的な接触の手前。実機で脚がぶつかる解は
#     引き続き罰される。
#   * ボールが飛んでいけば d_sole_to_ball が開いて自動的に元のしきい値へ戻るので、
#     カリキュラムも時計も要らない。「蹴る瞬間だけ緩い」が距離だけで実現する。
# --------------------------------------------------------------------------- #


def feet_close_penalty_kick_aware(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    feet_distance_threshold: float = 0.14,
    relaxed_threshold: float = 0.10,
    relax_dist: float = 0.5,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """``locomotion.mdp.rewards.feet_close_penalty`` の「ボールが近い間だけ緩める」版。

    ボールまでの距離 (``kick_state`` の ``d_sole_to_ball``) が ``relax_dist`` 未満の
    env だけしきい値を ``relaxed_threshold`` に切り替え、それ以外は locomotion 版と
    **完全に同じ値** を返す。負の重みで使う (locomotion 側と同じ −20)。

    Args:
        feet_distance_threshold: 通常時のしきい値 [m]。locomotion 側の既定と揃える。
        relaxed_threshold: ボールが近いときのしきい値 [m]。足箱の幅が 0.07 なので
            0.10 は左右の足がまだ 3cm 空いている状態。
        relax_dist: しきい値を緩める距離 [m]。0.5 はキックの構えに入る手前。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    near = state["d_sole_to_ball"] < relax_dist

    # locomotion 版をそのまま 2 回呼んで env ごとに選ぶ。しきい値を per-env の
    # テンソルにする形だと locomotion 側の実装を写し取ることになり、あちらを直した
    # ときに片方だけ古くなる。
    normal = _loco_feet_close_penalty(env, feet_distance_threshold=feet_distance_threshold)
    relaxed = _loco_feet_close_penalty(env, feet_distance_threshold=relaxed_threshold)
    return torch.where(near, relaxed, normal)


def knee_close_penalty_kick_aware(
    env: ManagerBasedRLEnv,
    r_stance: float,
    alpha: float,
    v_thresh: float,
    min_distance: float = 0.13,
    relaxed_min_distance: float = 0.10,
    relax_dist: float = 0.5,
    r_max: float | None = None,
    orbit_beta: float = 0.6,
    overshoot_margin: float = 0.0,
    lateral_band: tuple[float, float] | None = None,
    style_halfwidth: float | None = None,
) -> torch.Tensor:
    """``locomotion.mdp.rewards.knee_close_penalty`` の「ボールが近い間だけ緩める」版。

    切り替えの条件も理由も :func:`feet_close_penalty_kick_aware` と同じ。
    膝コライダーは半径 0.045 の円柱なので幾何的な接触は約 0.09、
    ``relaxed_min_distance = 0.10`` はその手前で、緩めても交差は罰され続ける。
    """
    state = kick_state(
        env,
        r_stance=r_stance,
        alpha=alpha,
        v_thresh=v_thresh,
        r_max=r_max,
        orbit_beta=orbit_beta,
        overshoot_margin=overshoot_margin,
        lateral_band=lateral_band,
        style_halfwidth=style_halfwidth,
    )
    near = state["d_sole_to_ball"] < relax_dist

    normal = _loco_knee_close_penalty(env, min_distance=min_distance)
    relaxed = _loco_knee_close_penalty(env, min_distance=relaxed_min_distance)
    return torch.where(near, relaxed, normal)
