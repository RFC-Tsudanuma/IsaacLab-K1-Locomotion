# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 2 (最終段): 凹凸地形でのロブキック。履歴入力・観測 DR つき。
gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Lob-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_rough_env_cfg:K1WalkLobRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Lob-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_rough_env_cfg:K1WalkLobRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobRoughPPORunnerCfg",
    },
)

# Stage 1: 凹凸地形でボール無しの歩行だけを学習する。履歴入力なので既存の
# walk phase タスク (1 フレーム観測) の checkpoint とは互換性が無く、別 ID で持つ。
gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Lob-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_rough_env_cfg:K1WalkLobRoughWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobRoughWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Lob-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_rough_env_cfg:K1WalkLobRoughWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobRoughWalkPhasePPORunnerCfg",
    },
)
