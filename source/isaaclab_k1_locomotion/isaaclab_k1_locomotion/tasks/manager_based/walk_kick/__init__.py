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
    id="Isaac-Velocity-Flat-K1-Walk-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickPPORunnerCfg",
    },
)

# 全方位版: ボール出現・蹴り方向とも 360°、距離 0.5-1.5 m。
# walk_kick の checkpoint から --load_pretrained で fine-tune する前提。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKick360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKick360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360PPORunnerCfg",
    },
)

# Stage 1: 通常の歩行のみを学習する。観測は Walk-Kick と同じ 55 次元なので、
# ここで得た checkpoint を Isaac-Velocity-Flat-K1-Walk-Kick-v0 に引き継げる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWalkPhasePPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Ablation 1: 凹凸地形 (terrain ablation)
#
# 平坦版との差は地形だけ。歩容ごと変わるので fine-tune ではなく
# Rough-...-Walk-Phase → Rough-...-Walk-Kick の 2 段で 0 から学習し直す。
# 観測 55 次元・並びは平坦版と同一。
#
# NOTE: 地形が Flat/Rough を分けるという既存の命名 (Isaac-Velocity-Rough-K1-v0) に
#       合わせて "Rough" 側に置く。walk_kick 系の他タスクは全て "Flat" 側。
# --------------------------------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Kick-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickWalkPhaseRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWalkPhaseRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Kick-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickWalkPhaseRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickWalkPhaseRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKickRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickRoughPPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Ablation 2: 転がるボール (ball velocity ablation)
#
# 地形は平坦のまま、リセット時にボールへランダム水平初速 (0-0.5 m/s) を与える。
# walk_kick_360 の checkpoint から --load_pretrained で fine-tune する前提。
# 観測 55 次元・並びは同一 (ball_vel スロットが元から入っている)。
# --------------------------------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Moving-Ball-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKick360MovingBallEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360MovingBallPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-360-Moving-Ball-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_env_cfg:K1WalkKick360MovingBallEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKick360MovingBallPPORunnerCfg",
    },
)
