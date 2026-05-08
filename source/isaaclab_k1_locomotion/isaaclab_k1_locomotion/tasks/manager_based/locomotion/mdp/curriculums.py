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
