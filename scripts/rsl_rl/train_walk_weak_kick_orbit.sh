#!/usr/bin/env bash
# walk_weak_kick_orbit の 4 段学習を通しで実行する (Stage 1 は既存 checkpoint を再利用)。
#
#   Stage 1: (学習しない) 共用の歩行 checkpoint k1_walk_kick_walk_phase を使う。
#            観測 55 次元・並びとも weak_orbit と同じなのでそのまま引き継げる。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-Weak-Orbit-v0
#            限定レンジ (ボール±60°/蹴り±45°) で「指令どおりの強さのキック」を獲得。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-v0
#            全方位。回り込み G (r_max=0.5) がいちばん効く段。
#   Stage 4: ...-Noisy-Ball-v0        ボール位置観測に遅延+ジッタ。
#   Stage 5: ...-Noisy-Ball-Walk-Init-v0  歩行状態から reset (要 K1_WALK_STATES_NPZ)。
#
# 元の weak との差は orbit_mods.py の 3 点 (回り込み G / 跨ぎの遊び / ボール物性 DR) と
# キック 4 項の ×3、ball_avoidance の σ_sole 0.20。段によって G の作り方が変わると
# 前段の歩き方・回り込み方が通用しなくなるので、全段に同じ改造を入れてある。
#
# ITER は 3000 以上にすること
# ------------------------------
# weak のカリキュラム (kick_velocity_strong のフェードアウト、σ_velocity の 1.0→0.35、
# overshoot 罰のフェードイン) が 3000 iteration でようやく終点に着く。途中で切ると
# 中途半端な報酬構成のまま止まるので、既定は 5000 にしてある。
#
# --resume は使わないこと
# -----------------------
# common_step_counter が同期され、次段のキック報酬フェードインが「もう終わった」と
# 判定されてランプしなくなる (train_walk_kick.sh 冒頭コメントと同じ理由)。
# --reset_noise_std も付けないこと (0.3 は収束 std の 3-5 倍で歩行が壊れる実測あり)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh            # Stage 2-5 を通しで
#   STAGE=23 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh   # Stage 2,3 だけ
#   STAGE=3 STAGE2_CKPT=logs/rsl_rl/k1_walk_kick_weak_orbit/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh        # 既存 Stage2 から 360 だけ
#   ITER=20000 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh # 仕上げ
#   STAGE1_CKPT=logs/.../model_4999.pt ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh
#   GPUS=4 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh          # 4 GPU で DDP
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh
#   GPUS=2 MASTER_PORT=29600 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh  # 2本同時
#
# NUM_ENVS は GPU 1 枚あたりの数 (合計は NUM_ENVS × GPUS)。詳細は
# _orbit_common.sh のマルチ GPU のコメント参照。
#   K1_WALK_STATES_NPZ=<path.npz> STAGE=5 ./scripts/rsl_rl/train_walk_weak_kick_orbit.sh
#
# 起動直後に必ず見ること: ログの "Loaded N tensors" / "Skipped N tensors"。
# actor.* が Skipped 側に並んでいたら checkpoint が繋がっていないので止めて引数を直す。

source "$(dirname "${BASH_SOURCE[0]}")/_orbit_common.sh"

ITER=${ITER:-5000}

# Stage 1 の歩行 checkpoint。weak_orbit 専用の walk phase は作っていない — 観測が
# 同一なので共用タスクのものをそのまま使う。
#
# NOTE: ここは「最新 run から自動で拾う」を **してはいけない**。共用の
#       k1_walk_kick_walk_phase には中断した run (model_4.pt / model_0.pt /
#       model_400.pt) が混ざっていて、run 名の新しい順に拾うと 400 iteration の
#       中断 run を掴む。train_walk_kick_weak.sh と同じ既知の完走 checkpoint を
#       直に指定する。
STAGE1_CKPT=${STAGE1_CKPT:-"logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt"}

STAGE2_TASK=${STAGE2_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Weak-Orbit-v0"}
STAGE3_TASK=${STAGE3_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-v0"}
STAGE4_TASK=${STAGE4_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Noisy-Ball-v0"}
STAGE5_TASK=${STAGE5_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Orbit-Noisy-Ball-Walk-Init-v0"}

# 次段が checkpoint を拾う先 (= 各 RunnerCfg の experiment_name)。
# タスクを差し替えるときは LOG_ROOT も必ず対で変えること。
STAGE2_LOG_ROOT=${STAGE2_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_weak_orbit"}
STAGE3_LOG_ROOT=${STAGE3_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_360_weak_orbit"}
STAGE4_LOG_ROOT=${STAGE4_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_360_weak_orbit_noisy_ball"}

if should_run 2; then
    if [[ ! -f "$STAGE1_CKPT" ]]; then
        echo "[ERROR] 歩行 checkpoint がありません: $STAGE1_CKPT" >&2
        echo "[ERROR] STAGE1_CKPT=<path> で明示してください。" >&2
        exit 1
    fi
    run_stage "Stage 2/5: weak_orbit (限定レンジ)" \
        "$STAGE2_TASK" "$ITER" "$STAGE1_CKPT" "$@"
fi

if should_run 3; then
    STAGE2_CKPT="${STAGE2_CKPT:-$(find_latest_ckpt "$STAGE2_LOG_ROOT")}"
    run_stage "Stage 3/5: 360_weak_orbit (全方位・回り込み)" \
        "$STAGE3_TASK" "$ITER" "$STAGE2_CKPT" "$@"
fi

if should_run 4; then
    STAGE3_CKPT="${STAGE3_CKPT:-$(find_latest_ckpt "$STAGE3_LOG_ROOT")}"
    run_stage "Stage 4/5: noisy_ball (観測遅延+ジッタ)" \
        "$STAGE4_TASK" "$ITER" "$STAGE3_CKPT" "$@"
fi

if should_run 5; then
    if [[ -z "${K1_WALK_STATES_NPZ:-}" ]]; then
        echo "[ERROR] Stage 5 は K1_WALK_STATES_NPZ (歩行状態プールの npz) が必要です。" >&2
        exit 1
    fi
    STAGE4_CKPT="${STAGE4_CKPT:-$(find_latest_ckpt "$STAGE4_LOG_ROOT")}"
    run_stage "Stage 5/5: walk_init (歩行状態から reset)" \
        "$STAGE5_TASK" "$ITER" "$STAGE4_CKPT" "$@"
fi

echo "[INFO] done."
