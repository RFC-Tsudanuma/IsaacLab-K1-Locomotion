# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""歩行 (model_34995) の上に B-Human の walk-kick の学習を載せたタスク。詳細は bhuman_walk_kick_env_cfg.py。"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-K1-BHuman-Walk-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bhuman_walk_kick_env_cfg:K1BHumanWalkKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1BHumanWalkKickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-K1-BHuman-Walk-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bhuman_walk_kick_env_cfg:K1BHumanWalkKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1BHumanWalkKickPPORunnerCfg",
    },
)

for _i in range(2, 8):
    for _suffix, _cls in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"Isaac-K1-BHuman-Walk-Kick-Stage{_i}{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.bhuman_walk_kick_env_cfg:K1BHumanWalkKickStage{_i}EnvCfg{_cls}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1BHumanWalkKickPPORunnerCfg",
            },
        )
