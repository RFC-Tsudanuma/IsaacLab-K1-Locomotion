#!/usr/bin/env bash
# 地形 ablation (凹凸地形) の 3 段学習を通しで実行する。
#
# B-Human のポスターとの差分のうち「bumpy な地面」だけを切り出して検証するための系統で、
# 平坦版 (train_walk_kick_360.sh) と **地形以外は完全に同一** の 3 段レシピをなぞる:
#
#   Stage 1: Isaac-Velocity-Rough-K1-Walk-Kick-Walk-Phase-v0  → k1_walk_kick_walk_phase_rough
#   Stage 2: Isaac-Velocity-Rough-K1-Walk-Kick-v0             → k1_walk_kick_rough
#   Stage 3: Isaac-Velocity-Rough-K1-Walk-Kick-360-v0         → k1_walk_kick_360_rough
#
# 平坦版の checkpoint から fine-tune するのではなく **0 から学習し直す**。地形が変わると
# 歩容そのものが変わるため、平坦で獲得した歩行を出発点にすると「平坦向けの歩容から
# 抜けられない」という交絡が入り、地形の効果を測れなくなる。
#
# 中身は train_walk_kick_360.sh にタスク名と log root を渡しているだけなので、
# python の解決・STAGE・ITER・WALK_ITER・NUM_ENVS・追加引数の扱いは全て同じ。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_360_rough.sh                 # 通しで実行
#   STAGE=23 ./scripts/rsl_rl/train_walk_kick_360_rough.sh        # rough の歩行学習済みなら 2,3 だけ
#   ITER=20000 ./scripts/rsl_rl/train_walk_kick_360_rough.sh      # kick 系を長く回す
#   STAGE=3 KICK_CKPT=logs/rsl_rl/k1_walk_kick_rough/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_360_rough.sh             # 既存の rough kick から 360 だけ
#
# NOTE: 平坦版の run (k1_walk_kick_walk_phase / k1_walk_kick / k1_walk_kick_360) とは
#       experiment_name が別なので、ログも checkpoint も混ざらない。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WALK_TASK="Isaac-Velocity-Rough-K1-Walk-Kick-Walk-Phase-v0" \
KICK_TASK="Isaac-Velocity-Rough-K1-Walk-Kick-v0" \
KICK360_TASK="Isaac-Velocity-Rough-K1-Walk-Kick-360-v0" \
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_walk_phase_rough" \
KICK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_rough" \
    exec "$REPO_ROOT/scripts/rsl_rl/train_walk_kick_360.sh" "$@"
