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
#       左足裏だけだった 3 次元スロットはボール 3D 位置へ置換し、
#       policy 223 / critic 61 / action 12 の全ベクトに mirror 写像を定義する。
#       PPO は data augmentation なし、係数 0.5 の mirror loss を使う。
#
#       long_pass の checkpoint はそのままでは読めないので、形状を合わせる場合は
#       scripts/rsl_rl/expand_checkpoint_history.py で列を並べ替え + ゼロ埋めする。
#       ただし左足裏の重み/統計がボール位置に渡るため、これは形状互換な
#       近似初期化に過ぎず、元との挙動一致は保証しない。
#       walk_long_pass_flag 用の expand_checkpoint_kick_flag.py は **使えない**
#       (あちらは末尾にゼロを足すだけで、途中に挿入される履歴スロットを扱えない)。
#       通し実行は scripts/rsl_rl/train_walk_long_pass_history.sh。
#
# NOTE: --reset_noise_std は **付けないこと**。蹴り方は完成品を引き継ぐので、
#       std を戻すと long_pass / walk_mid_kick と同じ失敗 (蹴り方が崩れる) をする。
