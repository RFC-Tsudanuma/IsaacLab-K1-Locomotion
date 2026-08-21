# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 右足インサイドキック。1 段で回す (限定レンジ → 全方位はキック成立率のゲートで拡大、
# ボール観測のノイズ+遅延は最初から入れる)。観測 55 次元・行動空間は walk_kick 系と
# 同一なので、リポジトリ同梱の walk phase checkpoint
# (k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt) から
# --load_pretrained でそのまま始められる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Inside-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_inside_kick_env_cfg:K1WalkInsideKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkInsideKickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_inside_kick_env_cfg:K1WalkInsideKickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkInsideKickPPORunnerCfg",
    },
)

# フォールバック: ボール観測のノイズ+遅延を入れない版。**通常は使わない**。
# インサイドの発見期が立ち上がらなかったときに、原因が観測ノイズなのか報酬設計なのかを
# 切り分けるためだけのタスク (env cfg の docstring 参照)。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_inside_kick_env_cfg:K1WalkInsideKickCleanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkInsideKickCleanPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Clean-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_inside_kick_env_cfg:K1WalkInsideKickCleanEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkInsideKickCleanPPORunnerCfg",
    },
)

# NOTE: 通し実行は scripts/rsl_rl/train_walk_inside_kick.sh (歩行 checkpoint から 1 段)。
#       --resume / --reset_noise_std は **使わないこと** (理由は env cfg の docstring 参照)。
# NOTE: middle のカリキュラム (strong の折れ線 / σ_velocity のアニール) が 3000
#       iteration で終点に着くので、--max_iterations は 3000 以上。拡大ゲートの公称
#       終点も 3000 なので、既定は 5000 にしてある。
