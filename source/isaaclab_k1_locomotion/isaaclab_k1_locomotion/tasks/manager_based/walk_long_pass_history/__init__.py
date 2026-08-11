# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_history_env_cfg:K1WalkLongPassHistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassHistoryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Long-Pass-History-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_long_pass_history_env_cfg:K1WalkLongPassHistoryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkLongPassHistoryPPORunnerCfg",
    },
)

# NOTE: policy 観測が 55 -> 223 次元に変わる (履歴 5 × 42 + 非履歴 13)。
#       行動 12 と critic 観測 61 は据え置き。long_pass の checkpoint はそのままでは
#       読めないので、必ず scripts/rsl_rl/expand_checkpoint_history.py で列を
#       並べ替え + ゼロ埋めしてから --load_pretrained に渡すこと。
#       walk_long_pass_flag 用の expand_checkpoint_kick_flag.py は **使えない**
#       (あちらは末尾にゼロを足すだけで、途中に挿入される履歴スロットを扱えない)。
#       通し実行は scripts/rsl_rl/train_walk_long_pass_history.sh。
#
# NOTE: --reset_noise_std は **付けないこと**。蹴り方は完成品を引き継ぐので、
#       std を戻すと long_pass / walk_mid_kick と同じ失敗 (蹴り方が崩れる) をする。
