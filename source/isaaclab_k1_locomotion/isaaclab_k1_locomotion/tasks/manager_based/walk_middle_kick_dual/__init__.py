# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""walk_middle_kick の dual 版 (dual encoder + 両足キック用の観測 2 変更) のタスク登録。

id は既存の middle タスクに ``-Dual`` を挿入したもの (walk_kick_dual と同じ位置)。
既存の 1 フレーム版タスクはそのまま残る。experiment_name も
``k1_walk_middle_kick_dual*`` で分けてあり、既存 run と混ざらない。

dual に入っているもの: (1) dual encoder = actor 入力を 100 フレームの観測履歴に、
(2) 観測スロット 3 = ボール 3D 位置 (元は左足裏 sole_pos)、(3) 歩行位相の初期
オフセット {0, π}、(4) 着地 shaping 3 項の無効化、(5) feet_phase weight 2.0 → 0.8、
(6) 最終段のボール観測 DR。(2)(3) は :mod:`..walk_kick_both_feet` 由来で、dual が
未学習のうちに直接畳み込んである (both_feet stage 2 実測: kick_foot_right_frac
1.0 → 0.39、kick_dir_error 4.5°、kick_rate 0.998)。(4)-(6) は fewa 47b8863 由来。
副作用として critic は 58 次元。

段構成は **4 段** (fewa 47b8863 と同じ配置)。Stage 1 は walk_kick_dual と共用::

    walk phase (walk_kick_dual) → middle → 360Middle (クリーン) → 360Middle DR (最終)

**最終段のボール観測 DR は一様ノイズ + 遅延** (feat 側のガウスパイプラインは
不採用、2026-08-17)。旧 ``-Noisy-Ball-v0`` の id はこの系列からは無くなった
(1 フレーム版の ``Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-Noisy-Ball-v0`` は健在)。

**引き継ぎ元は both_feet 系の checkpoint に限る。** 旧 sole_pos 系はスロット 3 の
意味が違うので使えない (policy は同じ 55 次元なので形の上では通ってしまう)。

通し実行は ``scripts/rsl_rl/train_walk_kick_middle_dual.sh``。
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Stage 2 (middle, dual): 帯 (3.2, 4.5) m/s で「5-10 m 相当の指令どおりのキック」。
# 引き継ぎ元は k1_walk_kick_dual_walk_phase (履歴入力版の walk phase)。
# 同梱の 1 フレーム walk phase から始めるなら --warm_start_from_single_frame。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_dual_env_cfg:K1WalkMiddleKickDualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickDualPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_dual_env_cfg:K1WalkMiddleKickDualEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickDualPPORunnerCfg",
    },
)

# Stage 3 (middle, dual): 全方位版 (クリーン)。観測 DR は入れない。σ_direction の
# アニール (0.35 → 0.15、1500 → 3000 iteration) と ball_avoidance のランプを
# ここで完走させる。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-360-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_dual_env_cfg:K1WalkMiddleKickDual360EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickDual360PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-360-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_middle_kick_dual_env_cfg:K1WalkMiddleKickDual360EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickDual360PPORunnerCfg",
    },
)

# Stage 4 (middle, dual, 最終): 全方位 + 観測 DR。中身は Stage 3 と同じで、
#   * IMU / エンコーダの遅延 DR (≤ 0.02 s)
#   * ボール観測の遅延 DR (0.02-0.10 s) とノイズ拡大 (位置 ±0.07 m / 速度 ±0.5 m/s)
#   * ランプの全凍結 (キック報酬 + kick_plant_foot + ball_avoidance。fine-tune 前提)
#   * σ_direction を Stage 3 の終値 0.15 で固定
# だけが乗る。地形は平面のまま。ボール観測 DR は **一様ノイズ + 遅延** で、
# ガウスの認識パイプライン (旧 Noisy-Ball 系) は不採用 (2026-08-17)。
gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-360-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_middle_kick_dual_env_cfg:K1WalkMiddleKickDual360DREnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickDual360DRPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-Dual-360-DR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_middle_kick_dual_env_cfg:K1WalkMiddleKickDual360DREnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkMiddleKickDual360DRPPORunnerCfg"
        ),
    },
)

# NOTE: ITER は **3000 以上**にすること (middle のカリキュラムが 3000 iteration で終点)。
#       指令帯はカリキュラムで動かさない (最初から固定)。
# NOTE: --reset_noise_std は付けないこと (env cfg の docstring 参照)。
