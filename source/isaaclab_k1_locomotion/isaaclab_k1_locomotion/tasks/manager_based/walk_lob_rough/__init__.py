# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_lob の履歴入力版 (平坦 3 段 + 凹凸 3 段) のタスク登録。

段の構成と、なぜ 3 段なのかは :mod:`.walk_lob_rough_env_cfg` の docstring を参照。
**まず Flat-* の 3 段を通すこと。** 凹凸 + ボールはこのリポジトリで一度も学習を
通していない組み合わせ。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

_ENTRY = "isaaclab.envs:ManagerBasedRLEnv"
_ENV = f"{__name__}.walk_lob_rough_env_cfg"
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg"

# (gym id, env cfg クラス名, RunnerCfg クラス名)
_TASKS = [
    # -- 平坦 3 段 -----------------------------------------------------------
    ("Isaac-Velocity-Flat-K1-Walk-Lob-Hist-Walk-Phase-v0",
     "K1WalkLobHistWalkPhaseEnvCfg", "K1WalkLobHistWalkPhasePPORunnerCfg"),
    ("Isaac-Velocity-Flat-K1-Walk-Lob-Hist-Kick-v0",
     "K1WalkLobHistKickEnvCfg", "K1WalkLobHistKickPPORunnerCfg"),
    ("Isaac-Velocity-Flat-K1-Walk-Lob-Hist-v0",
     "K1WalkLobHistEnvCfg", "K1WalkLobHistPPORunnerCfg"),
    # -- 凹凸 3 段 -----------------------------------------------------------
    ("Isaac-Velocity-Rough-K1-Walk-Lob-Walk-Phase-v0",
     "K1WalkLobRoughWalkPhaseEnvCfg", "K1WalkLobRoughWalkPhasePPORunnerCfg"),
    ("Isaac-Velocity-Rough-K1-Walk-Lob-Kick-v0",
     "K1WalkLobRoughKickEnvCfg", "K1WalkLobRoughKickPPORunnerCfg"),
    ("Isaac-Velocity-Rough-K1-Walk-Lob-v0",
     "K1WalkLobRoughEnvCfg", "K1WalkLobRoughPPORunnerCfg"),
]

for _id, _env_cls, _runner_cls in _TASKS:
    gym.register(
        id=_id,
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_ENV}:{_env_cls}",
            "rsl_rl_cfg_entry_point": f"{_AGENT}:{_runner_cls}",
        },
    )
    # PLAY 版は env cfg だけ差し替え、RunnerCfg は学習用と共通
    # (ネットワークが違うと checkpoint が載らない)。
    gym.register(
        id=_id.replace("-v0", "-Play-v0"),
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_ENV}:{_env_cls}_PLAY",
            "rsl_rl_cfg_entry_point": f"{_AGENT}:{_runner_cls}",
        },
    )
