# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Pass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_pass_env_cfg:K1WalkPassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Pass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_pass_env_cfg:K1WalkPassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkPassPPORunnerCfg",
    },
)

# NOTE: walk phase (stage 1) は Walk-Kick 側の
#       Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0 を共用する。
#       観測 55 次元・行動空間ともに同一で、歩行の学習内容は蹴りの強弱に依存しないため、
#       パス専用の walk phase タスクは用意していない。
