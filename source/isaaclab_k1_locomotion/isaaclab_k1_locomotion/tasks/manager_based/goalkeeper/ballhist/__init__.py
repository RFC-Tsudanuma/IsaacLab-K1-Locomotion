# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ボール履歴版ゴールキーパー (試験実装)。

直接版 (``Isaac-GoalkeeperDirect-*``) と **並列に存在する別タスク**で、
既存タスクの挙動には一切影響しない (dualhist と同じ方針)。

    * 方策の ``velocity_commands`` スロットをゼロ埋め
      → 「どこへどれだけ速く動くか」を決めていた手書きの式を方策から隠す
    * ボール相対位置の履歴 (0.4 秒 / base yaw frame) を観測の末尾に追加
      → 方策がそこから方向と速さを自分で決める

なぜこうするか: 実機で出た不具合3件 (待機時の震え / 静止ボールで横移動 /
起動デッドゲート) は、すべて手書きの指令生成に起因していた。式のパラメータが
性能の上限を決めている状態から抜けるのが目的。

**Stage1 のやり直しは不要**。観測の先頭 59 次元は直接版と同一なので、第1層の
新規列をゼロ初期化すれば初期状態が直接版と数学的に同一になる::

    # 直接版の ckpt を ボール履歴版の形へ拡張する
    python3 scripts/rsl_rl/expand_ckpt_for_ballhist.py \\
        --src logs/rsl_rl/k1_gk_direct_stage2/<run>/model_XXXXX.pt \\
        --dst logs/rsl_rl/k1_gk_ballhist_stage2/seed.pt

    # そこから追加学習
    STAGE1_CKPT=logs/rsl_rl/k1_gk_ballhist_stage2/seed.pt \\
        ./scripts/rsl_rl/train_gk_ballhist.sh --max_iterations 20000
"""

import gymnasium as gym

gym.register(
    id="Isaac-GoalkeeperBallHist-Stage2-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKBallHistStage2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.agents."
            "rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-GoalkeeperBallHist-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKBallHistEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.agents."
            "rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg"
        ),
    },
)

# --- Pure 版: 手書きの式を一切使わない (is_engaged を除く) ---
#   基準版との差は critic 観測と密な報酬だけ。詳細は env_cfg.py の該当セクション。
gym.register(
    id="Isaac-GoalkeeperBallHistPure-Stage2-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKBallHistPureStage2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.agents."
            "rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-GoalkeeperBallHistPure-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKBallHistPureEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_k1_locomotion.tasks.manager_based.goalkeeper.agents."
            "rsl_rl_ppo_cfg:K1GKDirectStage2PPORunnerCfg"
        ),
    },
)
