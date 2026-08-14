# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""デュアルヒストリー版ゴールキーパー (試験実装)。

arXiv:2401.16889 (Cassie) の dual I/O history をこのタスク向けに移植したもの。
既存の階層版 v2 (``Isaac-GoalkeeperHier-*``) と**並列に存在する別タスク**で、
既存タスクの挙動には一切影響しない。

    * 短期履歴 0.1s (5 frame @50Hz) → 生のまま MLP
    * 長期履歴 1.0s (50 frame)      → 1D CNN (論文と同じ k6/c32/s3 → k4/c16/s2) で圧縮
    * 履歴の中身はボール知覚 (フィールド座標系・見えていなければ 0) + 検出マスク +
      推定自己位置。詳細は :mod:`.observations` の ``GK_HIST_FRAME_DIM`` を参照。

学習・再生は既存の階層エンジンをそのまま使う (``--task`` を差し替えるだけ)::

    ./scripts/rsl_rl/train_gk_hier_dh_stage1.sh
    STAGE1_CKPT=... ./scripts/rsl_rl/train_gk_hier_dh_stage2.sh

★ 既存階層版の ckpt からは ``--resume`` できない (actor の構造が違う)。Stage1 から回すこと。
"""

import gymnasium as gym

gym.register(
    id="Isaac-GoalkeeperHierDH-Stage1-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKHierDHStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:K1GKHierDHStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperHierDH-Stage1-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKHierDHStage1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:K1GKHierDHStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperHierDH-Stage2-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKHierDHStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:K1GKHierDHStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GoalkeeperHierDH-Stage2-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKHierDHStage2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:K1GKHierDHStage2PPORunnerCfg",
    },
)
