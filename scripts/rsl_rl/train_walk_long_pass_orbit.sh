#!/usr/bin/env bash
# walk_long_pass_orbit の 4 段学習を通しで実行する (5-10m の強い転がしパス)。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Walk-Phase-v0
#            ボール無し。歩行だけを 50 フレーム履歴 CNN の観測形で獲得する。
#   Stage 2: ...-Loop-Pass-v0       目標球速 2.0-3.0、限定レンジ。
#   Stage 3: ...-Loop-Pass-360-v0   全方位化。
#   Stage 4: ...-Orbit-v0           帯を 3.2-5.0 へ (kick_rate ゲート式) + 凹凸地形。
#
# Stage 1 から専用タスクで通すこと
# --------------------------------
# この系統の actor は 50 フレーム履歴を 1D-CNN で符号化する (入力 371 次元)。共用タスク
# (1 フレーム 55 次元) の checkpoint とは actor の入力次元も重みの名前も違うので、
# --load_pretrained すると critic と正規化統計と std だけが載り、actor は乱数のまま
# 学習が始まる (実測: actor.{0,2,4,6}.{weight,bias} の 8 本が落ちる)。
# 共用 checkpoint から入る場合だけ --warm_start_from_single_frame を付ける
# (WARM_START=1 で自動的に付く。既定は付けない = 専用タスクで通す前提)。
#
# 失敗時の切り分け順序 (報酬より先に「継承できているか」を見る)
# -------------------------------------------------------------
#   1. 起動ログの "Loaded N tensors" / "Skipped N tensors"。actor.* が Skipped 側に
#      並んでいたら、そこで止めて引数を直す。
#   2. 最初の 20 iteration の Episode_Termination/base_height と mean_episode_length。
#      継承できていれば base_height は 0.03 前後・episode length 200 以上で、
#      kick_rate は 15 iteration ほどで 0.98 に戻る。base_height が 0.8 まで上がるなら
#      それは「歩けていない」のであって「蹴らない」のではない。
#   3. Policy/mean_noise_std。--reset_noise_std は付けないこと (0.3 は収束 std
#      0.06-0.095 の 3-5 倍で、これだけで歩行が壊れる)。
#
# --resume ではなく --load_pretrained を使うこと
# (common_step_counter が同期され、次段のフェードインカリキュラムが立ち上がらない)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass_orbit.sh             # 通しで 4 段
#   STAGE=234 ./scripts/rsl_rl/train_walk_long_pass_orbit.sh   # 歩行学習済みなら 2-4
#   STAGE=4 STAGE3_CKPT=logs/rsl_rl/k1_walk_long_pass_orbit_loop_360/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_orbit.sh         # Stage 4 だけやり直し
#   WARM_START=1 STAGE=4 STAGE3_CKPT=<共用タスクの ckpt> \
#       ./scripts/rsl_rl/train_walk_long_pass_orbit.sh         # 1 フレーム ckpt から移植
#   WALK_ITER=8000 ITER=5000 ./scripts/rsl_rl/train_walk_long_pass_orbit.sh
#   GPUS=4 ./scripts/rsl_rl/train_walk_long_pass_orbit.sh       # 4 GPU で DDP
#     (NUM_ENVS は GPU 1 枚あたり。詳細は _orbit_common.sh のコメント参照)

source "$(dirname "${BASH_SOURCE[0]}")/_orbit_common.sh"

# Stage 1 は歩行の獲得だけなので長めに取る (fewa 実績値)。Stage 2-4 は ITER。
WALK_ITER=${WALK_ITER:-8000}
ITER=${ITER:-5000}

# 1 フレーム観測の checkpoint から履歴入力版へ移すときだけ 1 にする。
WARM_START=${WARM_START:-0}
WARM_ARGS=()
[[ "$WARM_START" == "1" ]] && WARM_ARGS=(--warm_start_from_single_frame)

STAGE1_TASK=${STAGE1_TASK:-"Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Walk-Phase-v0"}
STAGE2_TASK=${STAGE2_TASK:-"Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Loop-Pass-v0"}
STAGE3_TASK=${STAGE3_TASK:-"Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-Loop-Pass-360-v0"}
STAGE4_TASK=${STAGE4_TASK:-"Isaac-Velocity-Flat-K1-Walk-Long-Pass-Orbit-v0"}

# 次段が checkpoint を拾う先 (= 各 RunnerCfg の experiment_name)。
STAGE1_LOG_ROOT=${STAGE1_LOG_ROOT:-"logs/rsl_rl/k1_walk_long_pass_orbit_walk_phase"}
STAGE2_LOG_ROOT=${STAGE2_LOG_ROOT:-"logs/rsl_rl/k1_walk_long_pass_orbit_loop_pass"}
STAGE3_LOG_ROOT=${STAGE3_LOG_ROOT:-"logs/rsl_rl/k1_walk_long_pass_orbit_loop_360"}

if should_run 1; then
    run_stage "Stage 1/4: walk phase (履歴入力・ボール無し)" \
        "$STAGE1_TASK" "$WALK_ITER" "" "$@"
fi

if should_run 2; then
    STAGE1_CKPT="${STAGE1_CKPT:-$(find_latest_ckpt "$STAGE1_LOG_ROOT")}"
    run_stage "Stage 2/4: loop_pass (球速 2.0-3.0・限定レンジ)" \
        "$STAGE2_TASK" "$ITER" "$STAGE1_CKPT" "${WARM_ARGS[@]}" "$@"
fi

if should_run 3; then
    STAGE2_CKPT="${STAGE2_CKPT:-$(find_latest_ckpt "$STAGE2_LOG_ROOT")}"
    run_stage "Stage 3/4: loop_pass_360 (全方位・回り込み)" \
        "$STAGE3_TASK" "$ITER" "$STAGE2_CKPT" "${WARM_ARGS[@]}" "$@"
fi

if should_run 4; then
    STAGE3_CKPT="${STAGE3_CKPT:-$(find_latest_ckpt "$STAGE3_LOG_ROOT")}"
    run_stage "Stage 4/4: long_pass (球速 3.2-5.0 ゲート式 + 凹凸地形)" \
        "$STAGE4_TASK" "$ITER" "$STAGE3_CKPT" "${WARM_ARGS[@]}" "$@"
fi

echo "[INFO] done."
