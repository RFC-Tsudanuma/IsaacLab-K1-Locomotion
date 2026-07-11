# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール回り込み (around_ball) タスク。

frozen 歩行ポリシーの上に載せる高レベルポリシーを学習する階層タスク。
学習/再生は dribble と同じ階層スクリプトを使う:

    train_dribble.py --task Isaac-AroundBall-K1-v0 --frozen_checkpoint <歩行 ckpt>
    play_dribble.py  --task Isaac-AroundBall-K1-Play-v0 --frozen_checkpoint <歩行 ckpt>
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-AroundBall-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.around_ball_env_cfg:K1AroundBallEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1AroundBallPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-AroundBall-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.around_ball_env_cfg:K1AroundBallEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1AroundBallPPORunnerCfg",
    },
)
