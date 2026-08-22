# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_long_pass_fewa: 47b8863 (fewa/walk_kick_dual_encoder_tune) の walk_long_pass。

**実機で成功した唯一のロングパス系統**をこのブランチへ移植したもの。ID と
experiment_name に ``Fewa`` / ``fewa`` を挟んで、このブランチの ``walk_long_pass``
(1 フレーム観測の別系統。walk_long_pass_history / _orbit / _dr / _flag が import
している) とは完全に分離してある。

通し実行は ``scripts/rsl_rl/train_walk_long_pass_fewa.sh``、
Stage 4 の ablation は ``scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh``。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# --------------------------------------------------------------------------- #
# Stage 1-3 (履歴入力版)
#
# 中身は共用タスク (Walk-Kick-Walk-Phase / Walk-Loop-Pass / Walk-Loop-Pass-360) と
# 同一で、policy 観測を 50 フレームの履歴にした点だけが違う。共用タスクに履歴を
# 足すと walk_kick / walk_pass / walk_lob / walk_mid_kick / loop_shoot まで
# 道連れになるので、long_pass 系列だけ別 ID に分けている
# (詳細は walk_long_pass_fewa_stages_env_cfg のモジュール docstring)。
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_stages_env_cfg:K1WalkLongPassFewaWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_stages_env_cfg:K1WalkLongPassFewaWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-Pass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_stages_env_cfg:K1WalkLongPassFewaLoopPassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaLoopPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-Pass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_stages_env_cfg:K1WalkLongPassFewaLoopPassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaLoopPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_stages_env_cfg:K1WalkLongPassFewaLoop360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaLoop360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_stages_env_cfg:K1WalkLongPassFewaLoop360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaLoop360PPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Stage 4 (ロングパス本体)
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_env_cfg:K1WalkLongPassFewaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_env_cfg:K1WalkLongPassFewaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaPPORunnerCfg",
    },
)

# NOTE: 観測項 (55 次元) と行動空間は walk_kick 系と同一だが、actor だけは
#       50 フレームの観測履歴を見る (直近 5 フレームそのまま + 50 フレームの
#       1D-CNN 潜在)。詳細は walk_long_pass_fewa_env_cfg の
#       「観測を 50 フレームの履歴にする」節を参照。
#       通し実行 (Stage 1-4) は scripts/rsl_rl/train_walk_long_pass_fewa.sh。
#       **共用タスク側の checkpoint からは actor を引き継げない** ので、
#       --load_pretrained には上の Stage 1-3 (履歴入力版) の run を使うこと。

# --------------------------------------------------------------------------- #
# Stage 4 の ablation (一晩で並列に回して翌日に良かったものを実機へ載せる)
#
# いずれも Stage 4 (Fewa-v0) を継承して **1 箇所だけ** 変えたもの。観測・行動空間は
# 基底と同一なので、出発 checkpoint は全変種で共有できる。
# 通し実行は scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh。
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_ablation_env_cfg:K1WalkLongPassFewaBand6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaBand6PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_ablation_env_cfg:K1WalkLongPassFewaBand6EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaBand6PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Calm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_ablation_env_cfg:K1WalkLongPassFewaCalmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaCalmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Calm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_ablation_env_cfg:K1WalkLongPassFewaCalmEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaCalmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6Calm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_ablation_env_cfg:K1WalkLongPassFewaBand6CalmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaBand6CalmPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6Calm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_fewa_ablation_env_cfg:K1WalkLongPassFewaBand6CalmEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassFewaBand6CalmPPORunnerCfg",
    },
)
