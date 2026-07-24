# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク。

frozen 歩行ポリシー (0524_walk.pt) の上に載せる高レベルポリシーを学習する階層タスク。
RoboCup HSL 2026 Middle ディビジョン想定。横ステップ移動でゴール (幅 2.5m) を守る。

学習/再生は dribble / around_ball と同じ階層スクリプトを task 引数で使い回す:

    train_dribble.py --task Isaac-Goalkeeper-Stage1-K1-v0 --frozen_checkpoint <歩行 ckpt>
    train_dribble.py --task Isaac-Goalkeeper-K1-v0        --frozen_checkpoint <歩行 ckpt> \\
                     --resume --checkpoint <Stage1 ckpt>
    play_dribble.py  --task Isaac-Goalkeeper-K1-Play-v0   --frozen_checkpoint <歩行 ckpt>
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Goalkeeper-Stage1-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_env_cfg:K1GoalkeeperStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GoalkeeperStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goalkeeper-Stage1-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_env_cfg:K1GoalkeeperStage1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GoalkeeperStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goalkeeper-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_env_cfg:K1GoalkeeperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GoalkeeperPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goalkeeper-Stage3-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_env_cfg:K1GoalkeeperStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GoalkeeperStage3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Goalkeeper-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_env_cfg:K1GoalkeeperEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GoalkeeperPPORunnerCfg",
    },
)

##
# 直接制御版 (goalkeeper_direct_env_cfg.py)
#   12 関節を直接制御する単一ポリシー。凍結歩行の横移動 (0.66 m/s) が
#   セーブ率の頭打ちだったため、横移動そのものを学習対象にした後継。
#   学習は通常の train.py を使う (階層版の train_goalkeeper.py ではない)。
##

gym.register(
    id="Isaac-GoalkeeperDirect-Stage1-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_direct_env_cfg:K1GKDirectStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKDirectPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperDirect-Stage1-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_direct_env_cfg:K1GKDirectStage1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKDirectPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperDirect-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_direct_env_cfg:K1GKDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperDirect-Stage3-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_direct_env_cfg:K1GKDirectStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKDirectStage3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperDirect-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_direct_env_cfg:K1GKDirectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg",
    },
)
