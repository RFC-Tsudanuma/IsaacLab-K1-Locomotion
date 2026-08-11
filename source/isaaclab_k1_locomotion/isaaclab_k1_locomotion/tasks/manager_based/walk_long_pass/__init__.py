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

# NOTE: 観測項 (55 次元) と行動空間は walk_kick 系と同一だが、actor だけは
#       50 フレームの観測履歴を見る (直近 5 フレームそのまま + 50 フレームの
#       1D-CNN 潜在)。このため loop_pass_360 の checkpoint からは critic と
#       action noise std しか引き継げない (walk_long_pass_env_cfg の
#       「観測を 50 フレームの履歴にする」節を参照)。
#       通し実行は scripts/rsl_rl/train_walk_long_pass.sh。
