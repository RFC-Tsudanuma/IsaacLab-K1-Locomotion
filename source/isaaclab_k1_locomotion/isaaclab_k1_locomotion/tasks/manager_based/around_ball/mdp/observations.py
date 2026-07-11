# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク専用の観測関数。"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 上位アクション駆動の歩行位相アキュムレータ用バッファ名 (locomotion 側の
# _gait_phase_per_env とは独立させ、base_velocity 駆動の位相と混ざらないようにする)
_HIGH_PHASE_ATTR = "_around_ball_gait_phase"
_HIGH_PHASE_STEP_ATTR = "_around_ball_gait_phase_last_step"

# fixed_freq=None (アキュムレータモード) 用の速度依存ケイデンス則。
# locomotion 側の新規約 (get_gait_phase) と同じ値のローカルコピー。
# NOTE: locomotion 側から import しない — 歩行コードの版によっては存在せず
# ImportError で around_ball 全体が読めなくなるため、意図的に自己完結にしている。
# 新規約の歩行 pt を frozen に使うときは、学習時の値とここが一致しているか確認すること。
_GAIT_FREQ_BASE = 1.7
_GAIT_FREQ_SLOPE = 0.5
_GAIT_FREQ_MIN = 1.7
_GAIT_FREQ_MAX = 2.6
_GAIT_FREQ_YAW_WEIGHT = 0.25
# randomize_phase_freq イベント (存在すれば) が書き込む per-env 周波数オフセットのバッファ名
_PHASE_FREQ_OFFSET_ATTR = "_phase_freq_offset_per_env"


def ball_offset_and_bearing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """ボールの base yaw frame 相対位置 (N, 2) と方位角の絶対値 |bearing| (N,) を返すヘルパ。"""
    ball: Articulation = env.scene["soccer_ball"]
    robot: Articulation = env.scene[asset_cfg.name]
    offset_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    offset_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), offset_w)[:, :2]
    bearing = torch.atan2(offset_b[:, 1], offset_b[:, 0]).abs()
    return offset_b, bearing


def ball_pos_rel_fov(
    env: ManagerBasedRLEnv,
    fov_half_angle_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """視野 (±fov_half_angle_deg) 内にあるときだけ更新されるボール相対位置 (base yaw frame, 2D)。

    実機のカメラ視野を模して、ボールの方位角が視野外のときは「最後に見えたときの値」を
    保持して返す (hold-last-seen)。保持値は見えた時点の base yaw frame 座標のままなので、
    その後ロボットが動くと古い値になる — これも「見失った」状況の近似として意図的。
    バッファはエピソードリセット時に :func:`reset_ball_last_seen` イベントで 0 にする
    (ボールは視野内にスポーンするので、リセット直後の観測計算で即座に真値へ更新される)。
    """
    offset_b, bearing = ball_offset_and_bearing(env, asset_cfg)
    visible = bearing <= math.radians(fov_half_angle_deg)

    buf = getattr(env, "_ball_last_seen_pos_b", None)
    if buf is None or buf.shape != offset_b.shape:
        buf = torch.zeros_like(offset_b)
        env._ball_last_seen_pos_b = buf
    buf[visible] = offset_b[visible]
    # ObsManager のノイズ付加でバッファ本体が汚れないように clone を返す
    return buf.clone()


def ball_in_fov(
    env: ManagerBasedRLEnv,
    fov_half_angle_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ボールが視野 (±fov_half_angle_deg) 内にあるかのフラグ (N, 1)。1=見えている。"""
    _, bearing = ball_offset_and_bearing(env, asset_cfg)
    visible = bearing <= math.radians(fov_half_angle_deg)
    return visible.float().unsqueeze(1)


def _high_action_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """上位ポリシーが frozen に注入した歩行コマンド (vx, vy, wz) を返す。未初期化なら 0。"""
    buf = getattr(env, "_prev_high_action", None)
    if buf is None or buf.shape != (env.num_envs, 3):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return buf


def _high_action_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """上位アクションの速度から周波数を決めて積算した左足の歩行位相 φ∈[0,2π) (per-env)。

    locomotion の :func:`get_gait_phase` と同一のアキュムレータ・周波数則
    (freq = clamp(BASE + SLOPE * speed) + per-env オフセット) だが、速度の出所を
    ``base_velocity`` コマンドではなく ``_prev_high_action`` (frozen が実際に受け取る
    歩行コマンド) にする。階層構成では base_velocity は使われないダミーなので、
    そこから位相を作ると frozen が学習時に見た「コマンド速度と位相テンポの対応」が
    崩れる — 本関数はその整合性を回復する。
    """
    phase = getattr(env, _HIGH_PHASE_ATTR, None)
    if phase is None:
        phase = torch.zeros(env.num_envs, device=env.device)
        setattr(env, _HIGH_PHASE_ATTR, phase)
        setattr(env, _HIGH_PHASE_STEP_ATTR, -1)

    # 同一ステップ内の複数回呼び出し (policy/critic/low_level) では 1 回だけ積算
    if getattr(env, _HIGH_PHASE_STEP_ATTR) != int(env.common_step_counter):
        cmd = _high_action_cmd(env)
        speed = torch.norm(cmd[:, :2], dim=1) + _GAIT_FREQ_YAW_WEIGHT * cmd[:, 2].abs()
        freq = _GAIT_FREQ_BASE + _GAIT_FREQ_SLOPE * speed
        offset = getattr(env, _PHASE_FREQ_OFFSET_ATTR, None)
        if offset is not None:
            freq = freq + offset
        freq = freq.clamp(_GAIT_FREQ_MIN, _GAIT_FREQ_MAX)
        phase = (phase + 2.0 * math.pi * freq * env.step_dt) % (2.0 * math.pi)
        # リセット直後の env は位相 0 から再開 (locomotion と同じ規約)
        phase = torch.where(env.episode_length_buf <= 1, torch.zeros_like(phase), phase)
        setattr(env, _HIGH_PHASE_ATTR, phase)
        setattr(env, _HIGH_PHASE_STEP_ATTR, int(env.common_step_counter))
    return getattr(env, _HIGH_PHASE_ATTR)


def high_action_phase_obs(
    env: ManagerBasedRLEnv,
    cmd_threshold: float = 0.05,
    fixed_freq: float | None = None,
) -> torch.Tensor:
    """上位アクション駆動の歩行位相を sin/cos で返す (左足, 右足の計4次元)。

    locomotion の :func:`phase_obs` と同一フォーマット (frozen 歩行ポリシーの
    ``gait_phase`` スロットにそのまま入る)。停止判定も同じ規約で、上位アクションの
    ノルムが ``cmd_threshold`` 未満なら位相をゼロ埋めして「停止すべき」と伝える。

    ``fixed_freq`` は frozen ポリシーの学習時期に合わせて選ぶ:
        * None (既定): 速度依存周波数のアキュムレータ (2026-07 以降の歩行 pt 用)。
        * 数値 (例 1.6): 旧規約 ``φ = 2π·f·t`` の固定周波数 (0524_walk.pt など
          2026-05 時点の歩行 pt はこちらで学習されている)。
    """
    if fixed_freq is not None:
        t = env.episode_length_buf * env.step_dt
        phase_left = (2.0 * math.pi * fixed_freq * t) % (2.0 * math.pi)
    else:
        phase_left = _high_action_gait_phase(env)
    phase_right = phase_left + math.pi

    phase = torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

    cmd = _high_action_cmd(env)
    cmd_speed = torch.norm(cmd[:, :3], dim=1, keepdim=True)
    is_stopped = cmd_speed < cmd_threshold
    phase = torch.where(is_stopped, torch.zeros_like(phase), phase)

    return phase
