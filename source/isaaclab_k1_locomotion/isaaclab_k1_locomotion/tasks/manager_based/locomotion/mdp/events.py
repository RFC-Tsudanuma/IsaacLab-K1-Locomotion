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
    f_min: float = 1.2,
    f_max: float = 5.0,
    dr_base: float = 1.6,
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
    safe = v.clamp(min=1e-6)
    cx = cmd[:, 0] / safe
    cy = cmd[:, 1] / safe
    inv_l = torch.sqrt((cx / float(l_fwd)) ** 2 + (cy / float(l_lat)) ** 2)
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
    if ph is None:
        return
    if env_ids is None:
        ph.zero_()
    else:
        ph[env_ids] = 0.0
