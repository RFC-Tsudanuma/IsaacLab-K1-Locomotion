# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_kick の dual 版 (dual encoder + 両足キック用の観測 2 変更) のタスク登録。

移植元は ``fewa/walk_kick_dual_encoder_tune`` の walk_long_pass 系列と
:mod:`..walk_kick_both_feet`。共用タスク (Isaac-Velocity-Flat-K1-Walk-Kick-*) に
これらを足すと walk_pass / walk_lob / walk_mid_kick / loop_shoot まで道連れになるので、
別 ID で登録している。experiment_name も ``k1_walk_kick_dual*`` で分けてあり、
既存 run と混ざらない。

dual に入っているもの: (1) dual encoder = actor 入力を 100 フレームの観測履歴に、
(2) 観測スロット 3 = ボール 3D 位置 (元は左足裏 sole_pos)、(3) 歩行位相の初期
オフセット {0, π}、(4) 着地 shaping 3 項の無効化、(5) 蹴り段の feet_phase weight
2.0 → 0.8、(6) 最終段のボール観測 DR。(2)(3) は :mod:`..walk_kick_both_feet` 由来で、
dual が未学習のうちに直接畳み込んである (both_feet stage 2 実測:
kick_foot_right_frac 1.0 → 0.39、kick_dir_error 4.5°、kick_rate 0.998)。
(4)-(6) は fewa 47b8863 由来。副作用として critic は 58 次元。

段構成は **4 段** (fewa 47b8863 と同じ配置)::

    walk phase → kick → 360 (クリーン) → 360 DR (最終)

**4 段構成と DR の配置は fewa 47b8863 に合わせた。最終段のボール観測 DR は
一様ノイズ + 遅延** (feat 側のガウスパイプライン ``noisy_ball_pos_b`` は不採用、
2026-08-17)。各段に何が載るかの表は ``walk_kick_dual_env_cfg.py`` の docstring。

**引き継ぎ元は both_feet 系の checkpoint に限る。** 旧 sole_pos 系はスロット 3 の
意味が違うので使えない (policy は同じ 55 次元なので形の上では通ってしまう)。

通し実行は ``scripts/rsl_rl/train_walk_kick_dual.sh``。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 1: 歩行のみ。共用の Walk-Phase と中身は同一で、観測が 100 フレーム履歴になる
# だけ。既存の 1 フレーム checkpoint を使うなら --warm_start_from_single_frame。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Walk-Phase-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDualWalkPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDualWalkPhasePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Walk-Phase-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDualWalkPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDualWalkPhasePPORunnerCfg",
    },
)

# Stage 2: 限定レンジ (ボール ±60° / 蹴り ±45°, 0.5-0.8 m) のキック。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDualPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDualEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDualPPORunnerCfg",
    },
)

# Stage 3: 全方位 (クリーン)。観測 DR は入れない。σ_direction のアニール
# (0.35 → 0.15、1500 → 3000 iteration) と ball_avoidance のランプをここで完走させる。
# **ITER は 3000 以上**。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDual360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDual360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDual360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDual360PPORunnerCfg",
    },
)

# Stage 4 (最終): 全方位 + 観測 DR。中身は Stage 3 と同じで、
#   * IMU / エンコーダの遅延 DR (≤ 0.02 s)
#   * ボール観測の遅延 DR (0.02-0.10 s) とノイズ拡大 (位置 ±0.07 m / 速度 ±0.5 m/s)
#   * ランプの全凍結 (キック報酬 + ball_avoidance。fine-tune 前提)
#   * σ_direction を Stage 3 の終値 0.15 で固定
# だけが乗る。地形は平面のまま。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDual360DREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDual360DRPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-DR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_kick_dual_env_cfg:K1WalkKickDual360DREnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkKickDual360DRPPORunnerCfg",
    },
)
