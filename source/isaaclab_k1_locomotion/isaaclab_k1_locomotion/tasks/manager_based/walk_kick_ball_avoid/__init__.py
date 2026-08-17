# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick の Ball Avoidance (原典解釈) 版のタスク登録。

共用タスク (Isaac-Velocity-Flat-K1-Walk-Kick-*) に変更を足すと walk_pass /
walk_lob / walk_mid_kick / loop_shoot まで道連れになるので、別 ID で登録している。
``experiment_name`` も ``k1_walk_kick_ball_avoid*`` で分けてあり、既存 run と混ざらない。

継承元 :class:`~..walk_kick.walk_kick_env_cfg.K1WalkKickEnvCfg` との差は 2 点:

1. 観測スロット 3 = **現在のボール 3D 位置** (遅延なし、``ball_pos_rel``)。
   元は左足裏 ``sole_pos`` で、原典 B-Human の観測表 "Current Ball 3D Position" の
   読み違えだった。
2. ``approach_penalty`` → ``ball_avoidance_exec``。「構えが完成しているのに足が
   ボールから遠い」ことを罰し、キック接触の瞬間に距離側が 0 になって罰が消える。
   d は **両足の平均** を使う (片足 min だと突き出し退行解と綺麗なインサイドが
   区別できない)。

段構成は **2 段**::

    walk phase → kick

**既存の walk_kick 系 checkpoint は引き継げない** (policy は同じ 55 次元なので
``--load_pretrained`` は形の上では通ってしまうが、スロット 3 の意味が違う)。
Walk-Phase から通しで学習すること。詳細は ``walk_kick_ball_avoid_env_cfg.py`` の
docstring。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 1: 歩行のみ。観測は stage 2 と同じ 55 次元・同じ並びなので、この checkpoint を
# そのまま Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-v0 に引き継げる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_ball_avoid_env_cfg:K1WalkKickBallAvoidWalkPhaseEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBallAvoidWalkPhasePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_ball_avoid_env_cfg:K1WalkKickBallAvoidWalkPhaseEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBallAvoidWalkPhasePPORunnerCfg"
        ),
    },
)

# Stage 2: キック (ボール ±60° / 蹴り ±45°, 0.5-0.8 m)。walk_kick との差は
# 観測スロット 3 と ball_avoidance_exec の 2 点だけ。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_ball_avoid_env_cfg:K1WalkKickBallAvoidEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBallAvoidPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_kick_ball_avoid_env_cfg:K1WalkKickBallAvoidEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickBallAvoidPPORunnerCfg"
        ),
    },
)
