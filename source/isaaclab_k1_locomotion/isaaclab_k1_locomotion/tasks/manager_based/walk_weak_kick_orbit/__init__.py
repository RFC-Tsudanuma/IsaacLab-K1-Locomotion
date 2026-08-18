# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# --------------------------------------------------------------------------- #
# walk_weak_kick の 4 段 (stage 2-5) の写し + 3 つの改造。
#
#   * 回り込み型の目標終端 G (r_max / orbit_beta)
#   * キック線を跨ぐときの遊び (overshoot_margin)
#   * ボールまわりの 4 点 DR (足の反発 / ボール物性 / 初期回転 / 転がり減速)
#
# 値と理由は walk_weak_kick_orbit/orbit_mods.py。弱いキックのレシピ自体は
# walk_weak_kick からそのまま import して使っている (二重定義しない)。
# 観測 55 次元・行動空間は walk_kick 系と同一なので、checkpoint はそのまま載る。
# --------------------------------------------------------------------------- #

# Stage 2 (weak, orbit): 限定レンジで「指令どおりの強さのキック」を獲得する。
# リポジトリ同梱の walk phase checkpoint
# (k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt) から始められる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Weak-Orbit-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKickWeakOrbitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWeakOrbitPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Weak-Orbit-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKickWeakOrbitEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWeakOrbitPPORunnerCfg",
    },
)

# Stage 3 (weak, orbit): 全方位版。k1_walk_kick_weak_orbit の checkpoint から続ける。
# 回り込み G の効果がいちばん出る段。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKick360WeakOrbitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakOrbitPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKick360WeakOrbitEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakOrbitPPORunnerCfg",
    },
)

# Stage 4 (weak, orbit): 知覚ノイズ+遅延つき。stage 3 との差は policy のボール位置観測だけ。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Noisy-Ball-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKick360WeakOrbitNoisyBallEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakOrbitNoisyBallPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Noisy-Ball-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKick360WeakOrbitNoisyBallEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakOrbitNoisyBallPPORunnerCfg",
    },
)

# Stage 5 (weak, orbit): 歩行ポリシーの歩行状態から reset する版。
# 状態プールのパスは環境変数 K1_WALK_STATES_NPZ で渡す。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Noisy-Ball-Walk-Init-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKick360WeakOrbitNoisyBallWalkInitEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakOrbitNoisyBallWalkInitPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Noisy-Ball-Walk-Init-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.walk_weak_kick_orbit_env_cfg:K1WalkKick360WeakOrbitNoisyBallWalkInitEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360WeakOrbitNoisyBallWalkInitPPORunnerCfg",
    },
)

# NOTE: --reset_noise_std は **使わないこと** (理由は env cfg の docstring 参照)。
# NOTE: weak のカリキュラムが 3000 iteration で終点に着くので、--max_iterations は 3000 以上。
