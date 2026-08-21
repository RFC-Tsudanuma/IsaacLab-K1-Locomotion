# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_likelihood_env_cfg:K1WalkKickLikelihoodWalkPhaseEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickLikelihoodWalkPhasePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_likelihood_env_cfg:K1WalkKickLikelihoodWalkPhaseEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickLikelihoodWalkPhasePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Stationary-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_likelihood_env_cfg:K1WalkKickLikelihoodStationaryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickLikelihoodStationaryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Stationary-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_likelihood_env_cfg:K1WalkKickLikelihoodStationaryEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickLikelihoodStationaryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_likelihood_env_cfg:K1WalkKickLikelihoodEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickLikelihoodPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_likelihood_env_cfg:K1WalkKickLikelihoodEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickLikelihoodPPORunnerCfg",
    },
)
