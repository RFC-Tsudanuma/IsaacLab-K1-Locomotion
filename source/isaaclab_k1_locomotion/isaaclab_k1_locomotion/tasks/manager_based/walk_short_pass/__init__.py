# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# --------------------------------------------------------------------------- #
# Stage 1-3 (履歴入力版)
#
# 中身は共用タスク (Walk-Kick-Walk-Phase / Walk-Loop-Pass / Walk-Loop-Pass-360) と
# 同一で、policy 観測を 50 フレームの履歴にした点だけが違う。共用タスクに履歴を
# 足すと walk_kick / walk_pass / walk_lob / walk_mid_kick / loop_shoot まで
# 道連れになるので、short_pass 系列だけ別 ID に分けている
# (詳細は walk_short_pass_stages_env_cfg のモジュール docstring)。
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_stages_env_cfg:K1WalkShortPassWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_stages_env_cfg:K1WalkShortPassWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_stages_env_cfg:K1WalkShortPassLoopPassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassLoopPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_stages_env_cfg:K1WalkShortPassLoopPassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassLoopPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_stages_env_cfg:K1WalkShortPassLoop360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassLoop360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_stages_env_cfg:K1WalkShortPassLoop360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassLoop360PPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Stage 4 (ショートパス本体)
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_env_cfg:K1WalkShortPassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_short_pass_env_cfg:K1WalkShortPassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkShortPassPPORunnerCfg",
    },
)

# NOTE: 観測項 (55 次元) と行動空間は walk_kick 系と同一だが、actor だけは
#       50 フレームの観測履歴を見る (直近 5 フレームそのまま + 50 フレームの
#       1D-CNN 潜在)。詳細は walk_short_pass_env_cfg の
#       「観測を 50 フレームの履歴にする」節を参照。
#       通し実行 (Stage 1-4) は scripts/rsl_rl/train_walk_short_pass.sh。
#       **共用タスク側の checkpoint からは actor を引き継げない** ので、
#       --load_pretrained には上の Stage 1-3 (履歴入力版) の run を使うこと。
