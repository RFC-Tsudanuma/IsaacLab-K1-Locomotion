# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""履歴 + 1D CNN 版の横移動下位ポリシー (arXiv:2401.16889 の dual I/O history)。

★ 2026-08-23: ``feat/inoue_walk_double_encoder`` (歩行タスクに同じ構造を入れた先行実装)
  から **ネットワーク・エクスポータ・対称変換の方式を移植** した。履歴は IsaacLab 標準の
  ``ObservationGroupCfg.history_length`` で取り、自前のリングバッファは持たない。

  あちらのブランチをマージしなかったのは、``K1FlatEnvCfg`` / ``K1RoughEnvCfg`` が
  master から大きく分岐しており、横移動タスクの土台 (この上に乗っている) ごと
  入れ替わってしまうため。**手法だけ借りてタスクは独立** させている。

☠ 先行実装の記録として、**同じ構造で実機に載せても後退転倒は直らなかった**
  (2026-08-04、flat_env_cfg.py のコメント参照)。原因は「深い履歴が CNN の暗黙状態推定を
  狂わせている疑い」とされていた。ただしあちらも位相定数の同期問題を抱えていた形跡が
  あり (学習 1.8Hz / 実機 1.5Hz)、履歴に gait_phase が入っている以上、同じ不一致が
  100 フレーム分増幅されていた可能性がある。**後退の改善をこの構造だけに期待しないこと。**

``goalkeeper/dualhist/`` (上位ポリシーのボール知覚履歴) とは完全に独立している。
"""

import gymnasium as gym

gym.register(
    id="Isaac-GKLateralDH-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKLateralDHEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:K1GKLateralDHPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-GKLateralDH-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1GKLateralDHEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:K1GKLateralDHPPORunnerCfg",
    },
)
