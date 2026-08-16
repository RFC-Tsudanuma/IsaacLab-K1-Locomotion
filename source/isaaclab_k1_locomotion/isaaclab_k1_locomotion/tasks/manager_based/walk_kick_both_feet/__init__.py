# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 1: 歩行のみ。観測は stage 2 と同じ 55 次元・同じ並びなので、この checkpoint を
# そのまま Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-v0 に引き継げる。
#
# NOTE: 既存の k1_walk_kick_walk_phase の checkpoint は流用しないこと。次元は同じでも
#       スロット 3 の意味が違う (左足裏 → ボール 3D 位置)。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_both_feet_env_cfg:K1WalkKickBothFeetWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBothFeetWalkPhasePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_both_feet_env_cfg:K1WalkKickBothFeetWalkPhaseEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBothFeetWalkPhasePPORunnerCfg"
        ),
    },
)

# Stage 2: キック。walk_kick との差は観測スロット 3 (足裏 → ボール 3D 位置) と
# 歩行位相の初期オフセット ({0, π} をエピソードごとに振る) の 2 点だけ。
# 蹴り足が左右に散ったかは Metrics/kick_direction/kick_foot_right_frac で見る。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_both_feet_env_cfg:K1WalkKickBothFeetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBothFeetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_both_feet_env_cfg:K1WalkKickBothFeetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBothFeetPPORunnerCfg",
    },
)

# NOTE: 通し実行は scripts/rsl_rl/train_walk_kick_both_feet.sh。
#       全方位版 (360) や lob / middle への展開は、それぞれの EnvCfg を
#       K1WalkKickBothFeetEnvCfg 側の観測グループと _apply_phase_offset で
#       組み替えれば同じ 2 点だけを移植できる。
