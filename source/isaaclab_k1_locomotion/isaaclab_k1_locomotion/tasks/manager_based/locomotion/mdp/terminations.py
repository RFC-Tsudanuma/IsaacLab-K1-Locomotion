# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for K1 locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_after_ball_contact(
    env: ManagerBasedRLEnv,
    time_after_contact: float = 3.0,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """ボールに最初に接触してから一定時間経過した環境を終了する。

    左右足の contact sensor を確認し、初回接触時刻を `_custom_buffers` に記録する。
    現在時刻 - 初回接触時刻 が ``time_after_contact`` を超えたら True を返す。

    リセット検知: episode_length_buf がリセットされると現在時刻が記録時刻より小さく
    なるため、その場合に記録をクリアする (event登録不要で自己復元する)。
    """
    sensor_right = env.scene.sensors["contact_balls_right"]
    sensor_left = env.scene.sensors["contact_balls_left"]
    force_right = torch.norm(sensor_right.data.force_matrix_w[:, 0, 0], p=2, dim=1)
    force_left = torch.norm(sensor_left.data.force_matrix_w[:, 0, 0], p=2, dim=1)
    has_contact = (force_right > contact_threshold) | (force_left > contact_threshold)

    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    buffer_key = "first_ball_contact_time"
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = torch.full(
            (env.num_envs,), -1.0, device=env.device
        )
    first_contact_time = env._custom_buffers[buffer_key]

    current_time = env.episode_length_buf.float() * env.step_dt

    # 環境リセット後は現在時刻が記録時刻より小さくなる → 記録を無効化
    reset_mask = current_time < first_contact_time
    first_contact_time = torch.where(
        reset_mask, torch.full_like(first_contact_time, -1.0), first_contact_time
    )

    # 未接触かつ今 接触 → 初回接触時刻として記録
    new_contact_mask = (first_contact_time < 0) & has_contact
    first_contact_time = torch.where(new_contact_mask, current_time, first_contact_time)

    env._custom_buffers[buffer_key] = first_contact_time

    elapsed = current_time - first_contact_time
    return (first_contact_time >= 0) & (elapsed > time_after_contact)


__all__ = [
    "time_after_ball_contact",
]
