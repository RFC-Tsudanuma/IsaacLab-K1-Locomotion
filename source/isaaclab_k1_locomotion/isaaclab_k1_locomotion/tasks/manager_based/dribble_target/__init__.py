# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""目標点へのドリブル (walk → dribble の直接 fine-tune) タスク。

このパッケージの歩行部分 (rough_env_cfg / flat_env_cfg / history_layout / agents / mdp) は
``feat/inoue_walk_double_encoder`` ブランチ (2026-09-18, 21d53cf) の ``locomotion`` パッケージの
複製。実機の歩行ポリシー (frozen_policies/best_walk) と同じ HistoryActorCritic・同じ歩行 cfg を
使うために、このブランチの ``locomotion`` (walk_kick 系が依存している別系統) とは分けて持つ。
歩行タスクの gym 登録は複製しない (ID が衝突するため)。ドリブルのタスクだけを登録する。

学習の流れ (scripts/rsl_rl/train_dribble_target.sh):
    0. widen_history_checkpoint.py で best_walk (49 次元) を 59 次元に広げる
    1. Stage1 (Isaac-Velocity-Flat-K1-Dribble-Target-WalkPhase-v0): 歩行のみ
    2. Stage2 (Isaac-Velocity-Flat-K1-Dribble-Target-v0): --resume --reset_critic で Stage1 から
"""

import gymnasium as gym

from . import agents

_ENV = f"{__name__}.k1_dribble_target_env_cfg"
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg"

gym.register(
    id="Isaac-Velocity-Flat-K1-Dribble-Target-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:K1DribbleTargetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT}:K1DribbleTargetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Dribble-Target-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:K1DribbleTargetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENT}:K1DribbleTargetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Dribble-Target-ResumeMature-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:K1DribbleTargetEnvCfg_ResumeMature",
        "rsl_rl_cfg_entry_point": f"{_AGENT}:K1DribbleTargetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Dribble-Target-WalkPhase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:K1DribbleTargetWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENT}:K1DribbleTargetWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Dribble-Target-WalkPhase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:K1DribbleTargetWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENT}:K1DribbleTargetWalkPhasePPORunnerCfg",
    },
)
