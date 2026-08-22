# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 3 段とも観測は policy 55 次元 (walk_kick 素の並び) / critic 61 次元で、actor だけ
# 100 フレームの履歴 (N, 100, 55) を取る。段をまたいで観測が変わらないので、
# checkpoint は stage 1 → 2 → 3 でそのまま繋がる。
#
# 設計と各段の根拠は walk_lob_plant_env_cfg.py のモジュール docstring。
# 通し実行は scripts/rsl_rl/train_walk_lob_plant.sh。
# --resume / --reset_noise_std は **使わないこと** (理由は env cfg の docstring)。

# --------------------------------------------------------------------------- #
# Stage 1: 歩行のみ。共用の歩行 checkpoint
# (k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt) から
# --warm_start_from_single_frame で入る (1 フレーム観測 → 履歴 actor への移植)。
# --------------------------------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Lob-Plant-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_plant_env_cfg:K1WalkLobPlantWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobPlantWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Lob-Plant-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_plant_env_cfg:K1WalkLobPlantWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobPlantWalkPhasePPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Stage 2: 平坦でロブ本体を学習する (本命)。
# 軸足の線形テント (kick_plant_lon / kick_plant_yaw) + 呼び水の kick_velocity_strong
# + loft/elevation の重み倍増。ITER は 8000 以上を推奨 (lon_span の折れ線が
# 4000 iteration で終点、apex はそこから先も伸びうる)。
# --------------------------------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Lob-Plant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_plant_env_cfg:K1WalkLobPlantEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobPlantPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Lob-Plant-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_plant_env_cfg:K1WalkLobPlantEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobPlantPPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Stage 3: 凹凸地形 + ボール物性 DR。地形バリアントは id の "Flat" を "Rough" に
# 置き換える (リポジトリ共通の流儀)。カリキュラムは env cfg 側で終値に固定済みなので、
# 「ITER は 4000 以上」のような下限は無い (既定 3000 は fine-tune としての量)。
# --------------------------------------------------------------------------- #
gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Lob-Plant-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_plant_env_cfg:K1WalkLobPlantRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobPlantRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-K1-Walk-Lob-Plant-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_lob_plant_env_cfg:K1WalkLobPlantRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLobPlantRoughPPORunnerCfg",
    },
)
