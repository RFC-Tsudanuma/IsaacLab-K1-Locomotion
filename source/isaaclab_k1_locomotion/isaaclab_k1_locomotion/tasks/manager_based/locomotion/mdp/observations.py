# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# observations.py

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Helper Functions
##

def phase_obs(env: ManagerBasedRLEnv, phase_freq: float = 1.5) -> torch.Tensor:
    """現在の歩行位相を sin/cos で返す (左足, 右足の計4次元)"""
    t = env.episode_length_buf * env.step_dt
    phase_left = 2.0 * math.pi * phase_freq * t
    phase_right = phase_left + math.pi

    return torch.stack([
        torch.sin(phase_left), torch.cos(phase_left),
        torch.sin(phase_right), torch.cos(phase_right),
    ], dim=1)

##
# Observation Groups (Asymmetric)
##

@configclass
class K1PolicyCfg(ObsGroup):
    """Actor（ポリシー）用：実機で得られる情報のみ。線速度は含めない。"""
    # 線速度(base_lin_vel)は削除
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)
    
    # 整理した位相観測
    gait_phase = ObsTerm(func=phase_obs, params={"phase_freq": 2.0}) # 頻度は適宜調整

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class K1CriticCfg(ObsGroup):
    """Critic（価値関数）用：特権情報（真の線速度など）を含める。"""
    # 基本情報はActorと同じ（ノイズなし）
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # Criticにはこれを入れる
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)
    gait_phase = ObsTerm(func=phase_obs, params={"phase_freq": 2.0})

    def __post_init__(self):
        self.enable_corruption = False # Criticにノイズは不要
        self.concatenate_terms = True