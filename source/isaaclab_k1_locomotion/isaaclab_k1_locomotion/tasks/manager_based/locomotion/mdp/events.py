# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 専用のイベント関数。"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_PHASE_FREQ_ATTR = "_phase_freq_per_env"


def randomize_phase_freq(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    base_phase_freq: float,
    offset_range: tuple[float, float] = (-0.1, 0.1),
):
    """環境毎の歩行周波数を ``base_phase_freq + uniform(offset_range)`` でランダム化する。

    結果は ``env._phase_freq_per_env`` (shape ``[num_envs]``) に保持し、
    位相を扱う観測/報酬関数 (``phase_obs``, ``feet_phase``, ``foot_clearance_ji_pen`` 等)
    から :func:`get_phase_freq` 経由で参照する。
    """
    base = float(base_phase_freq)

    buf: torch.Tensor | None = getattr(env, _PHASE_FREQ_ATTR, None)
    if buf is None:
        buf = torch.full((env.num_envs,), base, device=env.device)
        setattr(env, _PHASE_FREQ_ATTR, buf)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    low, high = float(offset_range[0]), float(offset_range[1])
    offsets = torch.empty(env_ids.numel(), device=env.device).uniform_(low, high)
    buf[env_ids] = base + offsets


def get_phase_freq(env: "ManagerBasedEnv", default: float) -> "float | torch.Tensor":
    """環境毎にランダム化された位相周波数があればそれを、無ければスカラー ``default`` を返す。"""
    val = getattr(env, _PHASE_FREQ_ATTR, None)
    if val is None:
        return default
    return val


def reset_prev_high_action(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
):
    """リセットされた env の ``_prev_high_action`` バッファを 0 にする。

    バッファ実体は ``HierarchicalVecEnvWrapper`` が用意するので、本関数は無ければ
    no-op で返す。Observation 計算は ``_reset_idx`` の後に走るので、ここで 0 化
    しておけば新エピソード最初の観測 ``last_high_action`` も 0 になる。
    """
    buf = getattr(env, "_prev_high_action", None)
    if buf is None:
        return
    if env_ids is None:
        buf.zero_()
        return
    buf[env_ids] = 0.0


__all__ = ["randomize_phase_freq", "get_phase_freq", "reset_prev_high_action"]


# ---------------------------------------------------------------------------
# 速度・方向に追従する歩行位相 (位相アキュムレータ)
# ---------------------------------------------------------------------------
#
# ★★★ 2026-08-21 追加。**固定周波数の位相では前進と横移動を両立できない**ことが
#   実測で確定したため。
#
#   同じ位相 1.6Hz で回した横移動ポリシー (k1_gk_lateral/2026-08-21_03-12-47) の実測:
#
#       方向    定常速度     歩幅 (= v / 2f)
#       前進    0.984 m/s    0.308 m    ← 1.6Hz で十分。追従率 98%
#       横      0.582 m/s    0.185 m    ← 1.6Hz が上限。指令 1.3 に対し追従率 45%
#
#   横歩きは足が交差できないので歩幅が前進の 60% しか出ず、**同じ速度を出すのに
#   ケイデンスが 1.7 倍要る**。1.3 m/s の横移動には f = 1.3/(2×0.185) = 3.5Hz が必要。
#   前進を 3.5Hz にすると今度は前進が壊れる (1.6Hz で足りているのに刻ませることになる)。
#
#   位相比 (実歩容の周波数 / 位相指令の周波数) と歩容の質は強く対応する (5 点で確認):
#
#       ポリシー   条件      位相比   f(10mm)  跳躍率   実機振動
#       0524_walk  前進      1.00     0.094    2.3%     なし
#       07-28      前進      1.16     0.061    5.1%     —
#       08-21版    横        0.97     0.097    1.9%     未確認
#       07-28      横        2.32     0.321   12.8%     あり
#       08-20版    横        2.98     0.350   16.3%     あり(悪化)
#
#   → **位相を指令から決めれば、方向によらず位相比 1 を保てる**というのが本実装の狙い。
#
# ☠ 位相は「時間の関数」ではなくなるので、**アキュムレータが要る**
#   (従来は `episode_length_buf × 固定freq` で、いつでも再計算できた)。
#   リセット時は :func:`reset_gait_phase` を EventTerm に登録して 0 に戻すこと。
#
# ☠ 既定は無効 (`adaptive=False`)。指定しない限り従来と完全に同じ挙動になる。

_GAIT_PHASE_ATTR = "_gait_phase_accum"        # (N,) 左足の位相 [rad]
_GAIT_PHASE_STEP_ATTR = "_gait_phase_accum_step"


def adaptive_phase_freq(
    env: "ManagerBasedEnv",
    command_name: str = "base_velocity",
    l_fwd: float = 0.31,
    l_lat: float = 0.11,
    l_back: float = 0.0,
    f_min: float = 1.2,
    f_max: float = 5.0,
    dr_base: float = 1.6,
    use_actual_speed: bool = False,
    cmd_gain: float = 1.0,
    vel_lag_s: float = 0.0,
    vel_noise_std: float = 0.0,
) -> torch.Tensor:
    """速度指令とその向きから、歩行位相の周波数 [Hz] を env ごとに決める (N,)。

    ``f = |v| / (2 L(θ))``。``L(θ)`` は方向による 1 歩の進み幅で、前進側 ``l_fwd`` と
    横側 ``l_lat`` を楕円で内挿する::

        1 / L(θ)² = (cosθ / l_fwd)² + (sinθ / l_lat)²

    純前進で ``L = l_fwd``、純横で ``L = l_lat`` になる。

    検算 (2026-08-21 の実測に基づく既定値):
        前進 1.0 m/s → 1.61 Hz  (現行の 1.6Hz と同じ = 前進の挙動は変わらない)
        横   1.3 m/s → 3.51 Hz
        横   0.5 m/s → 1.35 Hz

    Args:
        l_fwd: 前進の 1 歩あたりの進み幅 [m]。1.6Hz で 0.984 m/s だったので 0.308。
        l_lat: 横移動の 1 歩あたりの進み幅 [m]。1.6Hz で 0.582 m/s だったので 0.185。
        f_min: 下限 [Hz]。低速でも足を出し続けるための床。
        f_max: 上限 [Hz]。名目スイング時間が短くなりすぎるのを防ぐ
            (4.0Hz で 0.125s。実測の実スイング 0.13〜0.14s とほぼ同じ)。
        dr_base: 位相 DR を **相対倍率** として乗せるための基準 [Hz]。
            ``randomize_phase_freq`` の ``base_phase_freq`` と同じ値にすること。
    """
    cmd = env.command_manager.get_command(command_name)[:, :2]
    v = torch.norm(cmd, dim=1)
    if use_actual_speed:
        # ☠ **推論側で真の base_lin_vel が要る。** 実機の LowStateData は motors(q,dq) と
        #   imu(rpy,gyro,acc) だけで線速度を持たないので、この経路は「学習と推論で別の式」
        #   になる。2026-08-23 にそれが実機の引きずりの原因と判明した (位相 3.9Hz で学習 →
        #   実機は固定 1.6Hz)。**新しい学習では cmd_gain 方式を使うこと。**
        robot = env.scene["robot"]
        v = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
    else:
        # ★ 2026-08-23: 指令に **定常追従率** を掛けて実速度を近似する。
        #
        #   これで学習と推論が **完全に同じ式** になり、実機に速度推定が要らなくなる。
        #   素の指令 (cmd_gain=1.0) だと届かない要求と戦って歩幅が縮む問題があったが
        #   (横 1.444 → 0.941 m/s)、実測の追従率 0.87〜0.99 を掛ければその乖離が消える。
        #   検算: 横 1.2 → f 2.98Hz (実速度基準 3.05Hz、誤差 2%)
        #         横 1.5 → f 3.73Hz (実速度基準 3.89Hz、誤差 4%)
        #   誤差 4% は位相 DR の幅と同オーダーで、B2 の実測 (遅れ40ms・ノイズ10%でも
        #   引きずり率が悪化しない) の範囲に収まる。
        v = v * float(cmd_gain)

    # 一次遅れ (推論側の実装と一致させること)。
    #   use_actual_speed=True では「推定器の遅れ」の DR、
    #   cmd_gain 方式では「指令に実速度が追いつくまでの過渡」のモデルになる。
    #   ☠ 位相は積分器なので、ここが学習と推論でずれると位相がずれ続ける。
    if float(vel_lag_s) > 0.0:
        prev = getattr(env, "_phase_vel_filt", None)
        if prev is None or prev.shape[0] != v.shape[0]:
            prev = torch.zeros_like(v)
        alpha = float(env.step_dt) / (float(vel_lag_s) + float(env.step_dt))
        v = prev + alpha * (v - prev)
        env._phase_vel_filt = v.detach()
    # 推定器のばらつき DR (推論側には実装しない。頑健性を上げるためだけの学習時ノイズ)。
    if float(vel_noise_std) > 0.0:
        v = v * (1.0 + torch.randn_like(v) * float(vel_noise_std))
    v = v.clamp(min=0.0)
    safe = torch.norm(cmd, dim=1).clamp(min=1e-6)
    cx = cmd[:, 0] / safe
    cy = cmd[:, 1] / safe
    # 前後方向の 1 歩あたりの進み幅。後退は前進と別の値を使えるようにしてある。
    #
    # ★★ 2026-08-23: ``l_back`` を追加。K1 の足は足首関節を中心に **前後非対称**で、
    #   meshes/Left_Foot.STL の実測は つま先 +0.1195 m / かかと -0.0659 m。
    #   **後退で使える支持余裕は前進の 55%** しかない。同じ歩幅を要求するのが誤り。
    #   後退の歩幅を縮める = ケイデンスが上がる = 1 歩あたりの ZMP 振れが小さくなり、
    #   かかと側の余裕が増える。機構の非対称を歩容の非対称で吸収する。
    #
    #   ☠ 後退はシムでは既にほぼ完璧 (引きずり率 0.1% / 転倒最少) なので、**報酬側に
    #     改善の勾配が残っていない**。実機だけ不安定という sim2real ギャップに対して、
    #     方策の自由度を絞るこの手が数少ない直接のレバーになる。
    #   ☠ 効果は転倒率では見えない (元々ほぼ 0)。ZMP のかかと余裕で測ること。
    #
    #   0.0 (既定) では ``l_fwd`` と同じ = 従来どおりの前後対称な挙動。
    l_b = float(l_back) if float(l_back) > 0.0 else float(l_fwd)
    l_x = torch.where(cx < 0.0, torch.full_like(cx, l_b), torch.full_like(cx, float(l_fwd)))
    inv_l = torch.sqrt((cx / l_x) ** 2 + (cy / float(l_lat)) ** 2)
    f = 0.5 * v * inv_l
    f = torch.clamp(f, float(f_min), float(f_max))

    # 位相 DR は絶対値ではなく **倍率** として乗せる (base 1.6±0.05 → ×0.97〜1.03)
    pf = get_phase_freq(env, float(dr_base))
    if torch.is_tensor(pf):
        f = f * (pf / float(dr_base))
    return f


def get_gait_phase(
    env: "ManagerBasedEnv",
    phase_freq: float,
    adaptive: bool = False,
    **freq_kwargs,
) -> torch.Tensor:
    """左足の歩行位相 [rad] を返す (N,)。右足は ``+ pi``。

    ``adaptive=False`` (既定) では従来どおり ``2π × freq × 経過時間`` を返すので、
    既存タスクの挙動は 1 ビットも変わらない。

    ``adaptive=True`` では :func:`adaptive_phase_freq` の周波数で位相を積分する。
    ☠ 積分は **1 ステップにつき 1 回だけ**。同じステップで観測と報酬の両方から
      呼ばれても二重に進まないよう ``common_step_counter`` で番人を置いている
      (EventTerm の実行順に依存させない設計)。
    """
    if not adaptive:
        t = env.episode_length_buf * env.step_dt
        return 2.0 * torch.pi * get_phase_freq(env, phase_freq) * t

    ph: torch.Tensor | None = getattr(env, _GAIT_PHASE_ATTR, None)
    if ph is None or ph.shape[0] != env.num_envs:
        ph = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _GAIT_PHASE_ATTR, ph)
        setattr(env, _GAIT_PHASE_STEP_ATTR, -1)

    cur = int(getattr(env, "common_step_counter", 0))
    if getattr(env, _GAIT_PHASE_STEP_ATTR, -1) != cur:
        f = adaptive_phase_freq(env, **freq_kwargs)
        ph += 2.0 * torch.pi * f * env.step_dt
        # 数値が育ちすぎないよう [0, 2π) に畳む (sin/cos には影響しないが float32 の
        # 分解能が落ちるのを防ぐ。20s エピソードで最大 4Hz なら 500 周する)
        ph.remainder_(2.0 * torch.pi)
        setattr(env, _GAIT_PHASE_STEP_ATTR, cur)
    return ph


def reset_gait_phase(env: "ManagerBasedEnv", env_ids: torch.Tensor | None):
    """リセットされた env の歩行位相を 0 に戻す (EventTerm 用、mode="reset")。

    ☠ :func:`get_gait_phase` を ``adaptive=True`` で使うなら **必ず登録すること**。
      登録しないと前エピソードの位相を引き継ぎ、リセット直後の歩容が不連続になる。
    """
    ph: torch.Tensor | None = getattr(env, _GAIT_PHASE_ATTR, None)
    if ph is not None:
        if env_ids is None:
            ph.zero_()
        else:
            ph[env_ids] = 0.0

    # ★ 2026-08-23: 位相周波数の一次遅れフィルタ (adaptive_phase_freq の vel_lag_s) の
    #   状態も一緒に戻す。これが無いと新エピソードの先頭が前エピソード終盤の速度から
    #   始まり、位相の立ち上がりが env ごとにばらつく。
    filt: torch.Tensor | None = getattr(env, "_phase_vel_filt", None)
    if filt is not None:
        if env_ids is None:
            filt.zero_()
        else:
            filt[env_ids] = 0.0


def randomize_joint_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float] = (-0.02, 0.02),
    asset_cfg=None,
) -> None:
    """関節ゼロ点 (キャリブレーション) のずれを env ごとにランダム化する。

    ★ 2026-08-23 追加。**既存の DR に唯一欠けていた実機由来のばらつき**。

    実機の関節角は「組み付け + 零点較正の誤差」ぶんだけ真値からずれる。これは
    エピソード中ずっと同じ値の **定常バイアス**で、観測ノイズ (毎ステップ引き直す
    ``Unoise`` ±0.01〜0.03) とは性質がまったく違う ── 白色ノイズは方策が平均して
    消せるが、定常バイアスは消せない。左右で符号が違えば立ち姿が斜めになり、
    そのまま横移動の進行方向が傾く。

    実機で観測された症状との対応 (2026-08-23):
        「右に横移動すると斜め前に出る」。同じ ckpt をシムで左右別に測ると
        (``eval_gk_lateral_lr.py``) ドリフトは左のほうが大きく **向きが逆**だったので、
        学習の残差ではなく実機個体のずれが主因と判断した。個体差を学習側で吸収するには
        「個体差そのものを学習分布に入れる」しかない。

    実装:
        ``JointPositionActionCfg(use_default_offset=True)`` と ``joint_pos_rel`` は
        **どちらも ``robot.data.default_joint_pos`` を基準にしている**ので、ここを
        env ごと関節ごとに ``b`` だけずらすと

            * PD 目標の中立姿勢が ``b`` ぶん動く  (= 実際の立ち姿がずれる)
            * 方策が見る関節角もその中立基準になる (= 較正がずれた機体そのもの)

        の両方が一度に再現できる。

    Args:
        offset_range: 一様分布の範囲 [rad]。既定 ±0.02 rad ≈ ±1.15°。
        asset_cfg: 対象。``joint_ids`` を絞れば脚だけに掛けられる (既定は全関節)。

    ☠ **mode="startup" で使うこと。** ``default_joint_pos`` を書き換えるので、
      mode="reset" にするとリセットのたびにバイアスが**累積して発散する**。
    ☠ ``reset_joints_by_scale`` は ``default_joint_pos`` をスケールしてリセット姿勢を
      作るので、そちらにもこのバイアスが乗る (意図どおり: 較正のずれた機体は
      ずれた姿勢で立ち上がる)。
    """
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or isinstance(joint_ids, slice):
        joint_ids = slice(None)
        n_joints = asset.data.default_joint_pos.shape[1]
    else:
        n_joints = len(joint_ids)

    lo, hi = float(offset_range[0]), float(offset_range[1])
    bias = torch.empty(len(env_ids), n_joints, device=env.device).uniform_(lo, hi)

    default = asset.data.default_joint_pos
    if isinstance(joint_ids, slice):
        default[env_ids] += bias
    else:
        default[env_ids[:, None], torch.as_tensor(joint_ids, device=env.device)] += bias
