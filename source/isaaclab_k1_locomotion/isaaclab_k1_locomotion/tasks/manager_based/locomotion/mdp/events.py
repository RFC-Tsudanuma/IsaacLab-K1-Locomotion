# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 専用のイベント関数。"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_PHASE_FREQ_ATTR = "_phase_freq_per_env"
_PHASE_FREQ_OFFSET_ATTR = "_phase_freq_offset_per_env"
_GAIT_PHASE_ATTR = "_gait_phase_left"
_GAIT_PHASE_STEP_ATTR = "_gait_phase_last_step"


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


def randomize_phase_freq_offset(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float] = (-0.05, 0.05),
):
    """コマンド依存の基本周波数に加算する per-env オフセット [Hz] をランダム化する。

    結果は ``env._phase_freq_offset_per_env`` (shape ``[num_envs]``) に保持し、
    :func:`compute_cmd_phase_freq` が基本周波数 (速度コマンド依存) に自動で加算する。
    """
    buf: torch.Tensor | None = getattr(env, _PHASE_FREQ_OFFSET_ATTR, None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _PHASE_FREQ_OFFSET_ATTR, buf)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    low, high = float(offset_range[0]), float(offset_range[1])
    buf[env_ids] = torch.empty(env_ids.numel(), device=env.device).uniform_(low, high)


def compute_cmd_phase_freq(
    env: "ManagerBasedEnv",
    command_name: str = "base_velocity",
    low_speed: float = 1.0,
    high_speed: float = 1.8,
    low_freq: float = 1.5,
    high_freq: float = 2.0,
) -> torch.Tensor:
    """速度コマンドに応じた per-env 歩行周波数 [Hz] を返す。

    線速度コマンドのノルム ``s = ||cmd_xy||`` に対して:
      - ``s <= low_speed`` では ``low_freq`` で固定
      - ``s > low_speed`` では ``high_speed`` で ``high_freq`` になる傾きで線形増加
        (``high_speed`` 超もクランプせず同じ傾きで外挿する)
    :func:`randomize_phase_freq_offset` が設定した per-env オフセットがあれば加算する。
    """
    cmd = env.command_manager.get_command(command_name)
    speed = torch.norm(cmd[:, :2], dim=1)
    slope = (high_freq - low_freq) / (high_speed - low_speed)
    freq = low_freq + torch.clamp(speed - low_speed, min=0.0) * slope
    offset = getattr(env, _PHASE_FREQ_OFFSET_ATTR, None)
    if offset is not None:
        freq = freq + offset
    return freq


def get_gait_phase(
    env: "ManagerBasedEnv",
    command_name: str = "base_velocity",
    low_speed: float = 1.0,
    high_speed: float = 1.8,
    low_freq: float = 1.5,
    high_freq: float = 2.0,
) -> torch.Tensor:
    """左足の歩行位相 [rad, 0..2π) を返す (右足は +π)。

    周波数がコマンドに応じて時間変化するため、``2π f t`` の直接計算ではなく
    毎ステップ ``2π f(cmd) dt`` を積分して位相の連続性を保つ (コマンド再サンプル時に
    位相が飛ぶと接地スケジュールが突然変わり学習を乱すため)。

    実装ノート:
      - 1 ステップ内では reward → obs の順に複数回呼ばれるので、``common_step_counter``
        が進んだ最初の呼び出しでのみ積分を進める。
      - IsaacLab のステップ順序は「episode_length_buf 更新 → reward → reset → obs」なので、
        読み出し時に ``episode_length_buf == 0`` の env (リセット直後) を 0 に戻せば、
        reward は旧エピソードの位相を、reset 後の obs は位相 0 を見る。
    """
    phase: torch.Tensor | None = getattr(env, _GAIT_PHASE_ATTR, None)
    if phase is None:
        phase = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _GAIT_PHASE_ATTR, phase)
        setattr(env, _GAIT_PHASE_STEP_ATTR, int(env.common_step_counter))
    step = int(env.common_step_counter)
    if getattr(env, _GAIT_PHASE_STEP_ATTR) != step:
        freq = compute_cmd_phase_freq(env, command_name, low_speed, high_speed, low_freq, high_freq)
        phase.add_(2.0 * math.pi * freq * env.step_dt).remainder_(2.0 * math.pi)
        setattr(env, _GAIT_PHASE_STEP_ATTR, step)
    # リセット直後の env は位相 0 からエピソードを開始する
    phase[env.episode_length_buf == 0] = 0.0
    return phase


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


__all__ = [
    "randomize_phase_freq",
    "get_phase_freq",
    "randomize_phase_freq_offset",
    "compute_cmd_phase_freq",
    "get_gait_phase",
    "reset_prev_high_action",
]
