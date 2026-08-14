#!/usr/bin/env bash
# walk_kick_weak (弱いキックを指令どおりに出す作り直し) の学習を通しで実行する。
#
#   Stage 1: (学習しない) リポジトリ同梱の walk phase checkpoint を再利用する
#            logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-Weak-v0      -> k1_walk_kick_weak
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-v0  -> k1_walk_kick_360_weak
#
# 歩行 (stage 1) は既存タスクとまったく同じなので学習し直す必要がない。作り直すのは
# キック側 (stage 2 以降) だけ。walk_kick_360 のポリシーからの fine-tune にしないのは、
# あちらが「指令を無視して全力で蹴る」に収束していて探索 std も潰れており、報酬を
# 直しても弱いキックを再発見できないため (詳細は env cfg の docstring)。
#
# 2026-08-03 の run を使うのは knee_close_penalty 導入後の学習だから (.gitignore 参照)。
#
# 中身は train_walk_kick_360.sh にタスク名と log root を渡しているだけなので、
# python の解決・STAGE・ITER・NUM_ENVS・追加引数の扱いは全て同じ。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_weak.sh                  # stage 2,3 を通しで実行
#   ITER=5000 ./scripts/rsl_rl/train_walk_kick_weak.sh        # 各段の iteration を指定
#   STAGE=2 ./scripts/rsl_rl/train_walk_kick_weak.sh          # stage 2 だけ
#   STAGE=3 KICK_CKPT=logs/rsl_rl/k1_walk_kick_weak/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_weak.sh              # 既存の weak から 360 だけ
#   WALK_CKPT=logs/.../model_4999.pt ./scripts/rsl_rl/train_walk_kick_weak.sh
#                                                             # 別の walk phase から
#
# NOTE: ITER は **3000 以上**にすること。キック報酬のカリキュラム
#       (strong の立ち上げ→フェードアウト、σ アニール、overshoot 罰のフェードイン) が
#       3000 iteration で終点に着くので、それより短いと「まだ強く蹴った方が得」な
#       途中状態で終わる。既定は 5000。
# NOTE: --reset_noise_std は付けないこと (env cfg の docstring 参照)。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# stage 1 は学習しないので、既定で stage 2,3 だけを回す。
STAGE="${STAGE:-23}" \
ITER="${ITER:-5000}" \
WALK_CKPT="${WALK_CKPT:-logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt}" \
KICK_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-Weak-v0" \
KICK360_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-v0" \
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_walk_phase" \
KICK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_weak" \
    exec "$REPO_ROOT/scripts/rsl_rl/train_walk_kick_360.sh" "$@"
