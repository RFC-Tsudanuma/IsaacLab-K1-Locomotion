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


__all__ = [
    "randomize_phase_freq",
    "get_phase_freq",
    "randomize_phase_freq_offset",
    "compute_cmd_phase_freq",
    "get_gait_phase",
    "randomize_rigid_body_inertia",
    "reset_prev_high_action",
]


# ---------------------------------------------------------------------------
# ドリブル (Isaac-Velocity-Flat-K1-Dribble-Target-v0) 用: 障害物 (Level6-8)
# ---------------------------------------------------------------------------
# 障害物は静的な円柱 (kinematic、半径 0.2m)。シーンからは削除できないので、
# 障害物の無いレベル / 使わない障害物は env 原点から遠方 (_OBSTACLE_PARK_OFFSET) へ退避させる。
# 配置の流れ (ManagerBasedRLEnv._reset_idx の順序):
#   1. reset イベント reset_obstacle_between_start_and_target が全障害物を退避させる
#   2. コマンドの再サンプリング (WaypointPose2dCommand._resample_command) が経路を作った直後に
#      place_dribble_obstacles を呼び、env._dribble_num_obstacles 本を経路に合わせて置く
# レベルごとの配置方法は dribble_target_range_curriculum がセットする env 属性で決まる:
#   _dribble_num_obstacles / _dribble_corner_obstacles / _dribble_corridor

_OBSTACLE_PARK_OFFSET = (100.0, 0.0)
_OBSTACLE_Z = 0.5


def _write_obstacle_xy(env: ManagerBasedEnv, cfg: SceneEntityCfg, env_ids: torch.Tensor, xy: torch.Tensor):
    asset = env.scene[cfg.name]
    pose = asset.data.default_root_state[env_ids, :7].clone()
    pose[:, :2] = xy
    pose[:, 2] = _OBSTACLE_Z
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def reset_obstacle_between_start_and_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    obstacle_cfgs: list[SceneEntityCfg],
):
    """全障害物を遠方へ退避させ、配置用に障害物の一覧を env に覚えさせる。

    実際の配置はこの後のコマンド再サンプリングで行う (経路が決まってからでないと置けないため)。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._dribble_obstacle_cfgs = list(obstacle_cfgs)
    park = env.scene.env_origins[env_ids, :2] + torch.tensor(_OBSTACLE_PARK_OFFSET, device=env.device)
    for cfg in obstacle_cfgs:
        _write_obstacle_xy(env, cfg, env_ids, park)


def place_dribble_obstacles(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    start: torch.Tensor,
    goal: torch.Tensor,
    waypoints: torch.Tensor,
    sign: torch.Tensor,
    normal: torch.Tensor,
    fraction_range: tuple[float, float] = (0.4, 0.6),
    lateral_offset_range: tuple[float, float] = (-0.3, 0.3),
    corner_inset: float = 0.6,
    corridor_half_gap: float = 0.7,
):
    """経路に合わせて障害物を置く (WaypointPose2dCommand._resample_command から呼ばれる)。

    * 通常 (Level6): S→G の ``fraction_range`` の位置に、横へ ``lateral_offset_range`` ずらして置く。
    * corner_obstacles (Level7、Z字): 各コーナー (中間ウェイポイント) から経路の内側 (S→G の線の側) へ
      ``corner_inset`` [m] 寄せた位置に置く。コーナーのウェイポイントは障害物の外側に残るので、
      外側を回り込む必要がある。
    * corridor (Level8): S→G の中点に、横へ ±``corridor_half_gap`` [m] の 2 本をゲート状に置く
      (中心間 1.4m、障害物半径 0.2m を引いた通過幅は約 1.0m)。
    """
    cfgs = getattr(env, "_dribble_obstacle_cfgs", None)
    num = int(getattr(env, "_dribble_num_obstacles", 0))
    if not cfgs or num <= 0:
        return
    num = min(num, len(cfgs))
    seg = goal - start
    n = len(env_ids)
    positions: list[torch.Tensor] = []
    if getattr(env, "_dribble_corridor", False):
        mid = start + 0.5 * seg
        positions = [mid + corridor_half_gap * normal, mid - corridor_half_gap * normal]
    elif getattr(env, "_dribble_corner_obstacles", False):
        # z_shape のコーナーは start + t*seg ± amp*normal (1 本目が +sign、2 本目が -sign)
        corner_signs = [sign, -sign]
        for i in range(min(num, waypoints.shape[1] - 1)):
            positions.append(waypoints[:, i] - corner_signs[i] * corner_inset * normal)
    else:
        r = torch.empty(n, 1, device=env.device)
        for _ in range(num):
            frac = r.uniform_(*fraction_range).clone()
            lat = torch.empty(n, 1, device=env.device).uniform_(*lateral_offset_range)
            positions.append(start + frac * seg + lat * normal)
    for cfg, xy in zip(cfgs[:num], positions):
        _write_obstacle_xy(env, cfg, env_ids, xy)


# ---------------------------------------------------------------------------
# ドリブル: 評価指標「ボールが body yaw 座標で方位角 ±45° 以内にある時間の割合」
# ---------------------------------------------------------------------------
# robot_facing_ball 報酬は使っていないので、横や後ろにボールを置いたまま運ぶ傾向が
# 出ていないかを見るための指標。毎ステップの集計を interval イベントで行い、
# エピソード終了時に curriculum の log_ball_front_ratio が平均を TensorBoard に出す。


def track_ball_front_ratio(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_azimuth_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_ball_front_steps"):
        env._ball_front_steps = torch.zeros(env.num_envs, device=env.device)
        env._ball_front_total = torch.zeros(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    offset_w = ball.data.root_pos_w[env_ids] - robot.data.root_pos_w[env_ids]
    offset_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(robot.data.root_quat_w[env_ids]), offset_w)
    azimuth = torch.atan2(offset_b[:, 1], offset_b[:, 0]).abs()
    env._ball_front_steps[env_ids] += (azimuth <= math.radians(max_azimuth_deg)).float()
    env._ball_front_total[env_ids] += 1.0


# ---------------------------------------------------------------------------
# ドリブル: エピソードの途中の経路の引き直し
# ---------------------------------------------------------------------------


def replan_dribble_path(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "dribble_target",
    probability: float = 0.3,
    offset_range: tuple[float, float] = (0.2, 0.8),
):
    """interval イベント: 呼ばれた env のうち確率 probability で、未通過の途中のウェイポイントを横にずらす。

    実機の戦略は毎周期経路を計画し直すので、ウェイポイントが途中で動くことがある。その再現。
    最終目標は動かさない (WaypointPose2dCommand.replan 参照)。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    env.command_manager.get_term(command_name).replan(chosen, offset_range)


# ---------------------------------------------------------------------------
# ドリブル: ボールの減衰 (damping) のランダム化 (2026-10-25)
# ---------------------------------------------------------------------------
# IsaacLab には剛体の linear / angular damping を env ごとに変えるイベントが無いので、PhysX 側の damping は 0 にして、
# 制御周期 (20ms) ごとにボールの速度・角速度に exp(-c × dt) を掛ける (PhysX の damping dv/dt = -c v を 50Hz で離散化したもの)。
# NOTE: 最初は「-c × 質量 × 速度」の外力を permanent_wrench_composer でかけたが、composer はワールド→ボール座標の変換に
# 最初の呼び出し時の姿勢を使い続ける (リセットまで更新しない) ため、転がって回転するボールでは力の向きがずれて効かなかった。
# c は env ごとにエピソードの開始時に U(damping_range) で引く。本物のボールの転がり抵抗 (MuJoCo では転がり摩擦) の代わりでもある。


def randomize_ball_damping(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    linear_range: tuple[float, float] = (0.2, 0.8),
    angular_range: tuple[float, float] = (0.2, 0.8),
):
    """reset イベント: ボールの linear / angular damping の係数を env ごとに引き直す。"""
    if not hasattr(env, "_ball_damping"):
        env._ball_damping = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_damping[:, 0] = 0.5 * (linear_range[0] + linear_range[1])
        env._ball_damping[:, 1] = 0.5 * (angular_range[0] + angular_range[1])
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    env._ball_damping[env_ids, 0] = torch.empty(n, device=env.device).uniform_(*linear_range)
    env._ball_damping[env_ids, 1] = torch.empty(n, device=env.device).uniform_(*angular_range)


def apply_ball_damping(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("soccer_ball"),
):
    """interval イベント (毎制御周期): ボールの速度・角速度に exp(-c × step_dt) を掛ける。"""
    if not hasattr(env, "_ball_damping"):
        randomize_ball_damping(env, None)
    ball = env.scene[asset_cfg.name]
    decay = torch.exp(-env._ball_damping * env.step_dt)  # (N, 2): linear, angular
    vel = ball.data.root_vel_w.clone()  # (N, 6): 線速度, 角速度 (ワールド)
    vel[:, :3] *= decay[:, 0:1]
    vel[:, 3:] *= decay[:, 1:2]
    ball.write_root_velocity_to_sim(vel)
