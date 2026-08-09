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
    id="Isaac-Velocity-Flat-K1-Walk-Mid-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_mid_kick_env_cfg:K1WalkMidKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMidKickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Mid-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_mid_kick_env_cfg:K1WalkMidKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMidKickPPORunnerCfg",
    },
)

# NOTE: walk_kick_360 の checkpoint から --load_pretrained で fine-tune する前提
#       (通し実行は scripts/rsl_rl/train_walk_mid_kick.sh)。
#       --reset_noise_std は **使わないこと** (理由は env cfg の docstring 参照)。
#       観測 55 次元・行動空間は walk_kick 系と同一。
#
# NOTE: 同じ「5-10 m のキック」を反対側から狙う walk_long_pass (loop_pass_360 を
#       強くする側) と並行運用する。どちらが要件を満たすかは実機で判断すること。
