# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ゴールキーパー (goalkeeper) タスク。

frozen 歩行ポリシー (0524_walk.pt) の上に載せる高レベルポリシーを学習する階層タスク。
RoboCup HSL 2026 Middle ディビジョン想定。横ステップ移動でゴール (幅 2.6m) を守る。

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

# ★ 2026-07-31: 旧 "Isaac-GoalkeeperDirect-K1-v0" (初速固定レンジの旧 Stage2) と
#   "Isaac-GoalkeeperDirect-Stage3-K1-v0" (適応カリキュラムの旧 Stage3) を、
#   Stage2 統合後の実態に合わせて下の 1 つに集約した。
#   env cfg 側の K1GKDirectEnvCfg (難易度固定の土台) は Stage2 と Play が継承して
#   いるのでクラスとしては残っている (タスク登録のみ廃止)。
gym.register(
    id="Isaac-GoalkeeperDirect-Stage2-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_direct_env_cfg:K1GKDirectStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg",
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

##
# 階層版 v2 (goalkeeper_hier_env_cfg.py)
#   凍結下位 = k1_gk_direct_stage1/2026-07-28 (速度コマンド追従、横 1.28 m/s、実機実績あり)。
#   上位が歩行コマンド (vx, vy, wz) を学習する。直接制御版 Stage2 が手書き P 制御
#   (task_drive_vector) で埋めていたスロットを、学習したポリシーで置き換えるのが主眼。
#   学習・再生は階層エンジン (train_goalkeeper.py / play_goalkeeper.py) を使う。
##

gym.register(
    id="Isaac-GoalkeeperHier-Stage1-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_hier_env_cfg:K1GKHierStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKHierStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperHier-Stage1-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_hier_env_cfg:K1GKHierStage1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKHierStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperHier-Stage2-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_hier_env_cfg:K1GKHierStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKHierStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperHier-Stage2-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goalkeeper_hier_env_cfg:K1GKHierStage2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1GKHierStage2PPORunnerCfg",
    },
)
