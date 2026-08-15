#!/usr/bin/env bash
# walk_middle_kick (5-10 m 飛ぶキックを指令どおりに出す) の学習を通しで実行する。
#
#   Stage 1: (学習しない) リポジトリ同梱の walk phase checkpoint を再利用する
#            logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Middle-Kick-v0      -> k1_walk_middle_kick
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-v0  -> k1_walk_middle_kick_360
#
# 中身は walk_weak_kick のレシピ (3 点セット + ボール物性 DR) そのままで、指令帯だけを
# (0.25, 2.0) から (3.2, 4.5) m/s へ張り替えたもの。実機較正 a ≈ 1.0 m/s² と d = v²/2a
# から、3.2 m/s が 5.1 m、4.5 m/s が 10.1 m に対応する。
#
# walk_kick_360 のポリシーからの fine-tune にしないのは weak と同じ理由で、あちらが
# 「指令を無視して全力で蹴る」に収束していて探索 std も潰れているため。歩行 (stage 1) は
# 既存タスクとまったく同じなので学習し直す必要がなく、作り直すのはキック側だけ。
# 2026-08-03 の run を使うのは knee_close_penalty 導入後の学習だから (.gitignore 参照)。
#
# 中身は train_walk_kick_360.sh にタスク名と log root を渡しているだけなので、
# python の解決・STAGE・ITER・NUM_ENVS・追加引数の扱いは全て同じ。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_middle.sh                  # stage 2,3 を通しで実行
#   ITER=5000 ./scripts/rsl_rl/train_walk_kick_middle.sh        # 各段の iteration を指定
#   STAGE=2 ./scripts/rsl_rl/train_walk_kick_middle.sh          # stage 2 だけ
#   STAGE=3 KICK_CKPT=logs/rsl_rl/k1_walk_middle_kick/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_middle.sh              # 既存の middle から 360 だけ
#   WALK_CKPT=logs/.../model_4999.pt ./scripts/rsl_rl/train_walk_kick_middle.sh
#                                                               # 別の walk phase から
#
# NOTE: ITER は **3000 以上**にすること。キック報酬のカリキュラム
#       (strong の立ち上げ→フェードアウト、σ アニール、overshoot 罰のフェードイン) が
#       3000 iteration で終点に着くので、それより短いと「まだ強く蹴った方が得」な
#       途中状態で終わる。既定は 5000。指令帯はカリキュラムで動かさない (最初から固定)。
# NOTE: --reset_noise_std は付けないこと (env cfg の docstring 参照)。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# stage 1 は学習しないので、既定で stage 2,3 だけを回す。
STAGE="${STAGE:-23}" \
ITER="${ITER:-5000}" \
WALK_CKPT="${WALK_CKPT:-logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt}" \
KICK_TASK="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-v0" \
KICK360_TASK="Isaac-Velocity-Flat-K1-Walk-Middle-Kick-360-v0" \
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_walk_phase" \
KICK_LOG_ROOT="logs/rsl_rl/k1_walk_middle_kick" \
    exec "$REPO_ROOT/scripts/rsl_rl/train_walk_kick_360.sh" "$@"
