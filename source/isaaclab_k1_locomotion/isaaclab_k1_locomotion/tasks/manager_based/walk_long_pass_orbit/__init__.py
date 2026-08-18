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
# コミット 47b8863 時点の walk_long_pass 一式の移植 + 3 つの改造。
#
#   * 回り込み型の目標終端 G (r_max / orbit_beta)
#   * キック線を跨ぐときの遊び (overshoot_margin)
#   * ボールまわりの 4 点 DR (足の反発 / ボール物性 / 初期回転 / 転がり減速)
#
# 値と理由は walk_long_pass_orbit/orbit_mods.py。master 側の walk_long_pass とは
# 観測 (履歴入力) も報酬も別系統なので、タスク ID も experiment_name も分けてある。
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Stage 1-3 (履歴入力版)
#
# 中身は共用タスク (Walk-Kick-Walk-Phase / Walk-Loop-Pass / Walk-Loop-Pass-360) と
# 同一で、policy 観測を 50 フレームの履歴にした点だけが違う。共用タスクに履歴を
# 足すと walk_kick / walk_pass / walk_lob / walk_mid_kick / loop_shoot まで
# 道連れになるので、long_pass 系列だけ別 ID に分けている
# (詳細は walk_long_pass_stages_env_cfg のモジュール docstring)。
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_stages_env_cfg:K1WalkLongPassOrbitWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_stages_env_cfg:K1WalkLongPassOrbitWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Loop-Pass-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_stages_env_cfg:K1WalkLongPassOrbitLoopPassEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitLoopPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Loop-Pass-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_stages_env_cfg:K1WalkLongPassOrbitLoopPassEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitLoopPassPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Loop-Pass-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_stages_env_cfg:K1WalkLongPassOrbitLoop360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitLoop360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Loop-Pass-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_stages_env_cfg:K1WalkLongPassOrbitLoop360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitLoop360PPORunnerCfg",
    },
)

# --------------------------------------------------------------------------- #
# Stage 4 (ロングパス本体)
# --------------------------------------------------------------------------- #

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_env_cfg:K1WalkLongPassOrbitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_orbit_env_cfg:K1WalkLongPassOrbitEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassOrbitPPORunnerCfg",
    },
)

# NOTE: 観測項 (55 次元) と行動空間は walk_kick 系と同一だが、actor だけは
#       50 フレームの観測履歴を見る (直近 5 フレームそのまま + 50 フレームの
#       1D-CNN 潜在)。詳細は walk_long_pass_env_cfg の
#       「観測を 50 フレームの履歴にする」節を参照。
#       Stage 1-4 は上から順に --load_pretrained で繋いで回す (専用スクリプトは無い)。
#       **共用タスク側の checkpoint からは actor を引き継げない** ので、
#       --load_pretrained には上の Stage 1-3 (履歴入力版) の run を使うこと。
