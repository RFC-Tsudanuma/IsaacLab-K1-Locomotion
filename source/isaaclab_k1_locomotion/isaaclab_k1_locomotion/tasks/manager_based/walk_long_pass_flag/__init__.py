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
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Flag-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_flag_env_cfg:K1WalkLongPassFlagEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFlagPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Flag-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_flag_env_cfg:K1WalkLongPassFlagEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFlagPPORunnerCfg",
    },
)

# NOTE: **観測・行動空間が walk_kick 系 (55/12) と違う唯一のタスク** (56/13, critic 69)。
#       long_pass の checkpoint はそのままでは読めない。必ず
#       scripts/rsl_rl/expand_checkpoint_kick_flag.py でゼロパディングしてから
#       --load_pretrained に渡すこと (生の ckpt を渡すと actor の入力層・出力層が
#       ランダム初期化されてポリシーが壊れる)。通し実行は
#       scripts/rsl_rl/train_walk_long_pass_flag.sh。
#
# NOTE: --reset_noise_std は **付けないこと**。蹴り方は完成品を引き継ぐので、
#       std を戻すと long_pass / walk_mid_kick と同じ失敗 (蹴り方が崩れる) をする。
