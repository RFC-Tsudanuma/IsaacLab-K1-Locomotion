# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the K1 locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_command_resampling_time_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    resampling_time_range: tuple[float, float],
    num_steps: int,
):
    """指定ステップ数を超えたら、コマンドのリサンプリング時間範囲を変更する。"""
    if env.common_step_counter > num_steps:
        term = env.command_manager.get_term(command_name)
        term.cfg.resampling_time_range = resampling_time_range


class lin_vel_command_curriculum(ManagerTermBase):
    """線速度コマンド範囲(lin_vel_x / lin_vel_y)を段階的に拡げるカリキュラム。

    全環境にわたるトラッキング誤差 ``||cmd_xy - root_lin_vel_b_xy||_2`` の指数移動平均(EMA)が
    ``error_threshold`` を下回ったら次のステージに進む。最終ステージに到達したら以降は据え置く。

    Args:
        stages: 各ステージで使用する ``(min, max)``。 ``lin_vel_x`` と ``lin_vel_y`` の両方に同じ範囲を適用する。
        error_threshold: ステージを進めるためのEMA誤差(m/s)の上限。
        command_name: 対象コマンド名(例: ``"base_velocity"``)。
        asset_name: ロボットのアセット名。
        ema_alpha: EMA の更新係数 (0,1]。大きいほど直近の誤差を強く反映する。
        min_updates: ステージ進行を許可する前に必要な呼び出し回数(EMAを温めるため)。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages: list[tuple[float, float]] = [tuple(s) for s in params["stages"]]
        self._error_threshold: float = float(params["error_threshold"])
        self._command_name: str = params["command_name"]
        self._asset_name: str = params.get("asset_name", "robot")

        self._current_stage: int = 0
        self._error_ema: float | None = None
        self._update_count: int = 0

        # 初期ステージの範囲を即時適用
        self._apply_stage(self._stages[self._current_stage])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: Sequence[Sequence[float]],
        error_threshold: float,
        command_name: str,
        asset_name: str = "robot",
        ema_alpha: float = 0.02,
        min_updates: int = 50,
    ) -> dict[str, float]:
        asset: Articulation = env.scene[asset_name]
        cmd_term = env.command_manager.get_term(command_name)

        cmd_lin_xy = cmd_term.command[:, :2]
        actual_lin_xy = asset.data.root_lin_vel_b[:, :2]
        err = torch.norm(cmd_lin_xy - actual_lin_xy, dim=-1).mean().item()

        if self._error_ema is None:
            self._error_ema = err
        else:
            self._error_ema = (1.0 - ema_alpha) * self._error_ema + ema_alpha * err
        self._update_count += 1

        if (
            self._current_stage < len(self._stages) - 1
            and self._update_count >= min_updates
            and self._error_ema < error_threshold
        ):
            self._current_stage += 1
            self._apply_stage(self._stages[self._current_stage])
            self._update_count = 0
            self._error_ema = None

        return {
            "stage": float(self._current_stage),
            "error_ema": float(self._error_ema if self._error_ema is not None else 0.0),
            "lin_vel_max": float(self._stages[self._current_stage][1]),
        }

    def _apply_stage(self, vel_range: tuple[float, float]) -> None:
        cmd_term = self._env.command_manager.get_term(self._command_name)
        cmd_term.cfg.ranges.lin_vel_x = tuple(vel_range)
        cmd_term.cfg.ranges.lin_vel_y = tuple(vel_range)


class ball_max_speed_curriculum(ManagerTermBase):
    """ball_velocity_toward_target の max_speed を段階的に上げるカリキュラム。

    全環境にわたる「目標方向へのボール速度成分」 v_along の EMA が
    ``threshold_ratio * 現在 max_speed`` を超えたら次のステージに進む。
    最終ステージに到達したら据え置く。

    Args:
        stages: 各ステージで使用する ``max_speed`` の値のリスト (昇順)。
        threshold_ratio: ステージを進めるための ``v_along_ema / max_speed`` の下限。
        reward_term_name: 更新対象の reward term 名 (params["max_speed"] を書き換える)。
        command_name: ボールの目標位置を持つコマンド名。
        ball_asset: ボールのシーンエンティティ名。
        ema_alpha: EMA の更新係数 (0,1]。
        min_updates: ステージ進行を許可する前に必要な呼び出し回数。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages: list[float] = [float(s) for s in params["stages"]]
        self._threshold_ratio: float = float(params["threshold_ratio"])
        self._reward_term_name: str = params["reward_term_name"]
        self._command_name: str = params.get("command_name", "target_pos")
        self._ball_asset: str = params.get("ball_asset", "soccer_ball")

        self._current_stage: int = 0
        self._v_along_ema: float | None = None
        self._update_count: int = 0

        self._apply_stage(self._stages[self._current_stage])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: Sequence[float],
        threshold_ratio: float,
        reward_term_name: str,
        command_name: str = "target_pos",
        ball_asset: str = "soccer_ball",
        ema_alpha: float = 0.02,
        min_updates: int = 50,
    ) -> dict[str, float]:
        ball = env.scene[ball_asset]
        ball_pos_w = ball.data.root_pos_w[:, :2]
        ball_vel_w = ball.data.root_com_vel_w[:, :2]
        target_pos_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
        desired_dir = target_pos_w - ball_pos_w
        desired_unit = desired_dir / (torch.norm(desired_dir, dim=1, keepdim=True) + 1e-6)
        v_along = (ball_vel_w * desired_unit).sum(dim=1).clamp(min=0.0)
        v_along_mean = v_along.mean().item()

        if self._v_along_ema is None:
            self._v_along_ema = v_along_mean
        else:
            self._v_along_ema = (1.0 - ema_alpha) * self._v_along_ema + ema_alpha * v_along_mean
        self._update_count += 1

        current_max_speed = self._stages[self._current_stage]
        if (
            self._current_stage < len(self._stages) - 1
            and self._update_count >= min_updates
            and self._v_along_ema > threshold_ratio * current_max_speed
        ):
            self._current_stage += 1
            self._apply_stage(self._stages[self._current_stage])
            self._update_count = 0
            self._v_along_ema = None

        return {
            "stage": float(self._current_stage),
            "v_along_ema": float(self._v_along_ema if self._v_along_ema is not None else 0.0),
            "max_speed": float(self._stages[self._current_stage]),
        }

    def _apply_stage(self, max_speed: float) -> None:
        term_cfg = self._env.reward_manager.get_term_cfg(self._reward_term_name)
        term_cfg.params["max_speed"] = max_speed


class kick_direction_command_curriculum(ManagerTermBase):
    """蹴り方向 (target_pos の pos_y 範囲) を段階的に拡げるカリキュラム。

    動いているボールに対して ``1 - cos(ball_vel, desired_dir)`` の EMA が
    ``error_threshold`` を下回ったら次のステージへ。
    最終ステージに到達したら据え置く。

    Args:
        stages: 各ステージで使用する ``(pos_y_min, pos_y_max)`` のリスト。
            最初のステージを ``(0.0, 0.0)`` にすれば「正面のみ」になる。
        error_threshold: ステージを進めるための ``1 - cos_sim`` の上限。
            例: 0.2 なら cos_sim > 0.8 (≒ 37° 以内) になったら進む。
        command_name: 対象コマンド名 (UniformPose2dCommandCfg)。
        ball_asset: ボールのシーンエンティティ名。
        min_speed: 角度誤差を集計する際のボール速度下限 [m/s]。
            停止中はノイズになるので除外する。
        ema_alpha: EMA の更新係数 (0,1]。
        min_updates: ステージ進行を許可する前に必要な呼び出し回数。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stages: list[tuple[float, float]] = [tuple(s) for s in params["stages"]]
        self._error_threshold: float = float(params["error_threshold"])
        self._command_name: str = params["command_name"]
        self._ball_asset: str = params.get("ball_asset", "soccer_ball")
        self._min_speed: float = float(params.get("min_speed", 0.5))

        self._current_stage: int = 0
        self._error_ema: float | None = None
        self._update_count: int = 0

        self._apply_stage(self._stages[self._current_stage])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: Sequence[Sequence[float]],
        error_threshold: float,
        command_name: str,
        ball_asset: str = "soccer_ball",
        min_speed: float = 0.5,
        ema_alpha: float = 0.02,
        min_updates: int = 50,
    ) -> dict[str, float]:
        ball = env.scene[ball_asset]
        ball_pos_w = ball.data.root_pos_w[:, :2]
        ball_vel_w = ball.data.root_com_vel_w[:, :2]
        target_pos_w = env.command_manager.get_term(command_name).pos_command_w[:, :2]
        desired_dir = target_pos_w - ball_pos_w

        ball_speed = torch.norm(ball_vel_w, dim=1)
        mask = ball_speed > min_speed
        if mask.any():
            cos_sim = torch.nn.functional.cosine_similarity(
                ball_vel_w[mask], desired_dir[mask], dim=1, eps=1e-6
            )
            err = (1.0 - cos_sim).mean().item()
            if self._error_ema is None:
                self._error_ema = err
            else:
                self._error_ema = (1.0 - ema_alpha) * self._error_ema + ema_alpha * err

        self._update_count += 1

        if (
            self._current_stage < len(self._stages) - 1
            and self._update_count >= min_updates
            and self._error_ema is not None
            and self._error_ema < error_threshold
        ):
            self._current_stage += 1
            self._apply_stage(self._stages[self._current_stage])
            self._update_count = 0
            self._error_ema = None

        return {
            "stage": float(self._current_stage),
            "error_ema": float(self._error_ema if self._error_ema is not None else -1.0),
            "pos_y_max": float(self._stages[self._current_stage][1]),
        }

    def _apply_stage(self, y_range: tuple[float, float]) -> None:
        cmd_term = self._env.command_manager.get_term(self._command_name)
        cmd_term.cfg.ranges.pos_y = tuple(y_range)
