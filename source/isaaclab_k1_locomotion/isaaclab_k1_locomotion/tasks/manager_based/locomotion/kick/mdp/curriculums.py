# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg

import isaaclab_k1_locomotion.tasks.manager_based.locomotion.kick.mdp.rewards as kick_rewards

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class kick_success_rate_curriculum(ManagerTermBase):
    """Success-rate-based stage curriculum for the kick task."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._ball_spawn_pos = tuple(params["ball_spawn_pos"])
        self._success_distance = float(params["success_distance"])
        self._success_speed = float(params["success_speed"])
        self._speed_threshold_stage3 = float(params.get("avg_ball_speed_threshold", 1.2))
        self._contact_rate_threshold = float(params.get("contact_rate_threshold", 0.8))
        self._kick_rate_threshold = float(params.get("kick_rate_threshold", 0.7))
        self._recovery_rate_threshold = float(params.get("recovery_rate_threshold", 0.8))
        self._window_size = int(params.get("window_size", 128))
        self._contact_history: deque[float] = deque(maxlen=self._window_size)
        self._kick_history: deque[float] = deque(maxlen=self._window_size)
        self._recovery_history: deque[float] = deque(maxlen=self._window_size)
        self._peak_speed_history: deque[float] = deque(maxlen=self._window_size)
        self._double_support_history: deque[float] = deque(maxlen=self._window_size)
        self._com_in_support_history: deque[float] = deque(maxlen=self._window_size)
        self._feet_level_history: deque[float] = deque(maxlen=self._window_size)
        self._both_feet_low_history: deque[float] = deque(maxlen=self._window_size)
        self._upright_history: deque[float] = deque(maxlen=self._window_size)
        kick_rewards.set_curriculum_stage(env, kick_rewards.KICK_STAGE_DISCOVERY)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        ball_spawn_pos: tuple[float, float, float],
        success_distance: float,
        success_speed: float,
        window_size: int = 128,
        contact_rate_threshold: float = 0.8,
        kick_rate_threshold: float = 0.7,
        avg_ball_speed_threshold: float = 1.2,
        recovery_rate_threshold: float = 0.8,
    ) -> dict[str, float]:
        return kick_rewards._profile_named_call(
            env,
            "curriculum_kick_stage",
            self._compute_curriculum,
            env,
            env_ids,
            ball_spawn_pos,
            success_distance,
            success_speed,
            window_size,
            contact_rate_threshold,
            kick_rate_threshold,
            avg_ball_speed_threshold,
            recovery_rate_threshold,
        )

    def _compute_curriculum(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        ball_spawn_pos: tuple[float, float, float],
        success_distance: float,
        success_speed: float,
        window_size: int = 128,
        contact_rate_threshold: float = 0.8,
        kick_rate_threshold: float = 0.7,
        avg_ball_speed_threshold: float = 1.2,
        recovery_rate_threshold: float = 0.8,
    ) -> dict[str, float]:
        if isinstance(env_ids, slice):
            done_env_ids = torch.arange(env.num_envs, device=env.device)
        else:
            done_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

        if done_env_ids.numel() > 0:
            stats = kick_rewards.get_curriculum_episode_stats(
                env,
                done_env_ids,
                self._ball_spawn_pos,
                self._success_distance,
                self._success_speed,
            )
            self._contact_history.extend(stats["contact_success"].tolist())
            self._kick_history.extend(stats["kick_success"].tolist())
            self._recovery_history.extend(stats["recovery_success"].tolist())
            self._peak_speed_history.extend(stats["peak_ball_forward_speed"].tolist())
            self._double_support_history.extend(stats["double_support"].tolist())
            self._com_in_support_history.extend(stats["com_in_support"].tolist())
            self._feet_level_history.extend(stats["feet_level"].tolist())
            self._both_feet_low_history.extend(stats["both_feet_low"].tolist())
            self._upright_history.extend(stats["upright"].tolist())

        current_stage = kick_rewards.get_curriculum_stage(env)
        contact_rate = self._history_mean(self._contact_history)
        kick_rate = self._history_mean(self._kick_history)
        recovery_rate = self._history_mean(self._recovery_history)
        avg_peak_speed = self._history_mean(self._peak_speed_history)
        double_support_rate = self._history_mean(self._double_support_history)
        com_in_support_rate = self._history_mean(self._com_in_support_history)
        feet_level_rate = self._history_mean(self._feet_level_history)
        both_feet_low_rate = self._history_mean(self._both_feet_low_history)
        upright_rate = self._history_mean(self._upright_history)

        if current_stage == kick_rewards.KICK_STAGE_DISCOVERY and contact_rate >= self._contact_rate_threshold:
            current_stage = kick_rewards.KICK_STAGE_POWER
        elif current_stage == kick_rewards.KICK_STAGE_POWER and (
            kick_rate >= self._kick_rate_threshold or avg_peak_speed >= self._speed_threshold_stage3
        ):
            current_stage = kick_rewards.KICK_STAGE_RECOVERY
        elif current_stage == kick_rewards.KICK_STAGE_RECOVERY and recovery_rate >= self._recovery_rate_threshold:
            current_stage = kick_rewards.KICK_STAGE_FINAL_POSE

        kick_rewards.set_curriculum_stage(env, current_stage)
        return {
            "stage": float(current_stage),
            "contact_rate": contact_rate,
            "kick_rate": kick_rate,
            "recovery_rate": recovery_rate,
            "double_support_rate": double_support_rate,
            "com_in_support_rate": com_in_support_rate,
            "feet_level_rate": feet_level_rate,
            "both_feet_low_rate": both_feet_low_rate,
            "upright_rate": upright_rate,
            "avg_peak_ball_speed": avg_peak_speed,
            "window_size": float(self._window_size),
        }

    @staticmethod
    def _history_mean(history: deque[float]) -> float:
        if len(history) == 0:
            return 0.0
        return float(sum(history) / len(history))
