# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 2 (weak): 限定レンジで「指令どおりの強さのキック」を獲得する。
# 観測 55 次元・行動空間は walk_kick 系と同一なので、リポジトリ同梱の walk phase
# checkpoint (k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt) から
# --load_pretrained でそのまま始められる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Weak-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_env_cfg:K1WalkKickWeakEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWeakPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Weak-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_env_cfg:K1WalkKickWeakEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWeakPPORunnerCfg",
    },
)

# Stage 3 (weak): 全方位版。k1_walk_kick_weak の checkpoint から続ける。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_env_cfg:K1WalkKick360WeakEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_env_cfg:K1WalkKick360WeakEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakPPORunnerCfg",
    },
)

# Stage 4 (weak): 知覚ノイズ+遅延つき。k1_walk_kick_360_weak の checkpoint から続ける。
# stage 3 との差は policy のボール位置観測だけ (エピソードごとランダム遅延 + 30Hz
# サンプル&ホールド + ガウスジッタ σ=6.7cm・クリップ ±20cm)。観測 55 次元・並びは同一なので checkpoint はそのまま載る。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Noisy-Ball-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_env_cfg:K1WalkKick360WeakNoisyBallEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakNoisyBallPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Noisy-Ball-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_env_cfg:K1WalkKick360WeakNoisyBallEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakNoisyBallPPORunnerCfg",
    },
)

# NOTE: 通し実行は scripts/rsl_rl/train_walk_kick_weak.sh (stage 1 は同梱 checkpoint を再利用)。
#       --reset_noise_std は **使わないこと** (理由は env cfg の docstring 参照)。
# NOTE: カリキュラムが 3000 iteration で終点に着くので、--max_iterations は 3000 以上。
