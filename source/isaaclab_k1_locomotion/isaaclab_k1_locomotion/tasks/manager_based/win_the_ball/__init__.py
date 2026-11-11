# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""成功率ゲート付き統合win_the_ballタスクと互換タスクを登録する。"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Flat-K1-Win-The-Ball-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.win_the_ball_env_cfg:K1WinTheBallKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WinTheBallKickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Win-The-Ball-Directional-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.win_the_ball_env_cfg:K1WinTheBallDirectionalEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WinTheBallDirectionalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Win-The-Ball-Directional-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.win_the_ball_env_cfg:K1WinTheBallDirectionalEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WinTheBallDirectionalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Win-The-Ball-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.win_the_ball_env_cfg:K1WinTheBallKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WinTheBallKickPPORunnerCfg",
    },
)