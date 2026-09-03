# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 専用のイベント関数。"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

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


def randomize_rigid_body_inertia(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    inertia_distribution_params: tuple[float, float] = (0.7, 1.3),
):
    """各リンクの慣性テンソルをリンク毎の一様乱数倍でスケールする (startup 専用)。

    IsaacLab 標準には質量と独立な慣性ランダム化がないため自作。
    「現在の」慣性値に倍率を掛けるので、``randomize_rigid_body_mass``
    (recompute_inertia=True: 慣性 = default × 質量比) の **後** に実行すれば
    質量ランダム化と合成される (最終慣性 = default × 質量比 × 本倍率)。
    テンソル全成分に同一スカラを掛けるため正定値性は保たれる。

    Note:
        現在値に累積で掛かるため mode="startup" (1回のみ) でしか使わないこと。
        reset モードで使うと呼ばれる度に縮小/拡大が複利で効いてしまう。
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    inertias = asset.root_physx_view.get_inertias()  # (E, B, 9), CPU
    lo, hi = float(inertia_distribution_params[0]), float(inertia_distribution_params[1])
    ratios = torch.empty((env_ids.numel(), body_ids.numel(), 1)).uniform_(lo, hi)
    inertias[env_ids[:, None], body_ids] = inertias[env_ids[:, None], body_ids] * ratios
    asset.root_physx_view.set_inertias(inertias, env_ids)


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


def reset_root_state_prone_supine(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    lying_height: float = 0.2,
    prone_prob: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """ロボットを「うつ伏せ (prone)」または「仰向け (supine)」の寝た状態で reset する。

    起き上がり (get-up) 学習用。各 env ごとに確率 ``prone_prob`` でうつ伏せ、残りを仰向けに
    初期化する。胴体を水平に倒す (pitch ±90°) ため、上向き既定姿勢 (立位) の root を
    水平に回転させ、高さを ``lying_height`` に下げて地面に寝かせる。

    Args:
        pose_range: ``x`` / ``y`` / ``z`` (位置オフセット) と ``roll`` / ``yaw`` (向きのばらつき)
            の ``(min, max)``。``z`` は ``lying_height`` への加算オフセット、``pitch`` は
            うつ伏せ/仰向けを決めるので無視される。未指定キーは 0。
        velocity_range: ルート速度の ``(min, max)`` (``x,y,z,roll,pitch,yaw``)。
        lying_height: 寝たときのルート高さ [m] (胴体半分の厚み程度)。地面貫通を避けるため
            少し高めから落として静定させる。
        prone_prob: うつ伏せにする確率。残り (1-prone_prob) は仰向け。
        asset_cfg: 対象アセット。
    """
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    num = len(env_ids)
    device = asset.device

    # --- 位置: 既定 xy + env 原点 + ランダム xy、高さは lying_height (+ ランダム z) ---
    pos_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    pos_ranges = torch.tensor(pos_range_list, device=device)
    rand_pos = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (num, 3), device=device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pos
    positions[:, 2] = env.scene.env_origins[env_ids, 2] + lying_height + rand_pos[:, 2]

    # --- 向き: pitch ±90° で水平に倒す (prone: +90°, supine: -90°) + yaw/roll のばらつき ---
    is_prone = torch.rand(num, device=device) < prone_prob
    pitch = torch.where(
        is_prone,
        torch.full((num,), math.pi / 2, device=device),
        torch.full((num,), -math.pi / 2, device=device),
    )
    roll_range = pose_range.get("roll", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    roll = math_utils.sample_uniform(roll_range[0], roll_range[1], (num,), device=device)
    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (num,), device=device)
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    # --- 速度 ---
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=device)
    rand_vel = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (num, 6), device=device)
    velocities = root_states[:, 7:13] + rand_vel

    # --- 反映 ---
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


__all__ = [
    "randomize_phase_freq",
    "get_phase_freq",
    "randomize_phase_freq_offset",
    "compute_cmd_phase_freq",
    "get_gait_phase",
    "randomize_rigid_body_inertia",
    "reset_prev_high_action",
    "reset_root_state_prone_supine",
]
