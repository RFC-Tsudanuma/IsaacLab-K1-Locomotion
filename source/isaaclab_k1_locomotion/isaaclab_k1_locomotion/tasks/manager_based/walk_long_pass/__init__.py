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
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_env_cfg:K1WalkLongPassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_env_cfg:K1WalkLongPassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_env_cfg:K1WalkLongPassDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassDRPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-DR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_env_cfg:K1WalkLongPassDREnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassDRPPORunnerCfg",
    },
)

# NOTE: -DR-v0 は **継続学習用**。long_pass の checkpoint から --load_pretrained で
#       始める (カリキュラムは cfg 側で終値に固定済みなので再ランプしない)。
#       通し実行は scripts/rsl_rl/train_walk_long_pass_dr.sh。

# NOTE: loop_pass_360 の checkpoint から --load_pretrained + --reset_noise_std で
#       fine-tune する前提 (通し実行は scripts/rsl_rl/train_walk_long_pass.sh)。
#       観測 55 次元・行動空間は walk_kick 系と同一。
