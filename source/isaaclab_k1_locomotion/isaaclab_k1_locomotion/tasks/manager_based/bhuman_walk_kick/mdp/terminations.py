# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""B-Human ``_check_termination`` の移植。判定そのものは :class:`BHumanKickCommand` が持つ。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fallen(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    """胴体が terminate_height より低い状態が terminate_time 続いた / 速度が発散した。"""
    k = env.command_manager.get_term(command_name)
    k.ensure_updated()
    return k.terminated.clone()


def kicks_done(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    """episode_number_of_kicks 回より多く蹴り終えた (B-Human では time_out 扱い)。"""
    k = env.command_manager.get_term(command_name)
    k.ensure_updated()
    return k.kicks_done.clone()


def ball_timeout(env: ManagerBasedRLEnv, command_name: str = "kick") -> torch.Tensor:
    """1 個のボールに ball_timeout_s 以上かかった (B-Human の 30 秒タイムアウト、time_out 扱い)。"""
    k = env.command_manager.get_term(command_name)
    k.ensure_updated()
    return k.ball_timeout.clone()
