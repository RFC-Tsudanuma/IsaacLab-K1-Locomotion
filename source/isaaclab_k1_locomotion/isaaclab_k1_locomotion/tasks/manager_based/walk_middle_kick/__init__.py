# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 2 (middle): 帯 (3.2, 4.5) m/s = 実機 5-10 m 相当のキックを指令どおりに獲得する。
# 中身は walk_weak_kick のレシピ + 帯の張り替え + σ 終点 0.5。観測 55 次元・行動空間は
# walk_kick 系と同一なので、リポジトリ同梱の walk phase checkpoint
# (k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt) から
# --load_pretrained でそのまま始められる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_env_cfg:K1WalkMiddleKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_env_cfg:K1WalkMiddleKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickPPORunnerCfg",
    },
)

# Stage 3 (middle): 全方位版。k1_walk_middle_kick の checkpoint から続ける。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_env_cfg:K1WalkMiddleKick360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKick360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_env_cfg:K1WalkMiddleKick360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKick360PPORunnerCfg",
    },
)

# NOTE: 通し実行は scripts/rsl_rl/train_walk_kick_middle.sh (stage 1 は同梱 checkpoint を再利用)。
#       --reset_noise_std は **使わないこと** (理由は env cfg の docstring 参照)。
# NOTE: カリキュラムが 3000 iteration で終点に着くので、--max_iterations は 3000 以上。
