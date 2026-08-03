#!/usr/bin/env bash
# walk_loop_pass_360 の 3 段階学習を通しで実行する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する (walk_kick と共用)。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0
#            限定レンジ (ボール±60°/蹴り±45°, 0.5-0.8m) で浮かせる蹴りを獲得する。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Loop-Pass-360-v0
#            全方位 (360°/360°, 0.5-1.5m) + 回り込み。
#
# Stage 2 → 3 の checkpoint 継承が実質カリキュラム (蹴り方は既知・回り込みだけ新規)。
# 全 stage とも観測 55 次元・同じ並びなので、--load_pretrained でそのまま引き継げる。
# --resume を使わない理由は train_walk_kick.sh の冒頭コメントと同じ
# (common_step_counter が同期されてキック報酬カリキュラムがランプしなくなる)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_loop_pass_360.sh              # 通しで実行
#   STAGE=23 ./scripts/rsl_rl/train_walk_loop_pass_360.sh     # 歩行学習済みなら 2,3 だけ
#   STAGE=3 LOOP_CKPT=logs/rsl_rl/k1_walk_loop_pass/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_loop_pass_360.sh          # 既存 loop_pass から 360 だけ
#   STAGE=12 ./scripts/rsl_rl/train_walk_loop_pass_360.sh    # loop_pass までで止める
#   NUM_ENVS=2048 ITER=5000 ./scripts/rsl_rl/train_walk_loop_pass_360.sh
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_loop_pass_360.sh   # walk phase だけ延長
#
# iteration 数は walk phase (WALK_ITER, 既定 5000) と kick 系 (ITER, 既定 20000) で
# 別に持つ。walk phase は歩行の獲得だけなので 5000 で足りる。
#
# NOTE: 旧 experiment 名 (logs/rsl_rl/k1_walk_loop) の run から始めるときは
#       自動検出に乗らないので LOOP_CKPT で明示すること。

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_kick.sh と同じ手順)。
# --------------------------------------------------------------------------- #
for _bf in "${BASH_FUNCTIONS:-}" "$HOME/.bash_functions" /home/satoshi/.bash_functions; do
    if [[ -n "$_bf" && -f "$_bf" ]]; then
        # shellcheck disable=SC1090
        source "$_bf"
        break
    fi
done

LAB_PY=""
if [[ -n "${LAB_PYTHON:-}" ]]; then
    LAB_PY="$LAB_PYTHON"
elif type _labpython2 >/dev/null 2>&1; then
    LAB_PY="_labpython2"
else
    for _cand in "$REPO_ROOT/isaaclab.sh" /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
        if [[ -x "$_cand" ]]; then
            case "$_cand" in
                *isaaclab.sh) LAB_PY="$_cand -p" ;;
                *)            LAB_PY="$_cand" ;;
            esac
            break
        fi
    done
    if [[ -z "$LAB_PY" ]]; then
        for _cand in python python3; do
            if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -c "import isaaclab" >/dev/null 2>&1; then
                LAB_PY="$_cand"
                break
            fi
        done
    fi
fi

if [[ -z "$LAB_PY" ]]; then
    echo "[ERROR] IsaacLab の python が見つかりません。LAB_PYTHON で明示してください。" >&2
    exit 1
fi
echo "[INFO] python: $LAB_PY"

NUM_ENVS=${NUM_ENVS:-4096}
ITER=${ITER:-20000}
# walk phase (Stage 1) は歩行を獲得するだけなので 5000 で足りる (実績値)。
# ITER とは別に持ち、通しで実行しても Stage 1 に 20000 かけないようにする。
WALK_ITER=${WALK_ITER:-5000}
STAGE=${STAGE:-all}

WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0"
LOOP_TASK="Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0"
LOOP360_TASK="Isaac-Velocity-Flat-K1-Walk-Loop-Pass-360-v0"
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_walk_phase"
LOOP_LOG_ROOT="logs/rsl_rl/k1_walk_loop_pass"

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
find_latest_ckpt() {
    local latest_run ckpt
    latest_run=$(find "$1" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    if [[ -z "$latest_run" ]]; then
        echo "[ERROR] run が見つかりません: $1" >&2
        return 1
    fi
    ckpt=$(find "$latest_run" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)
    if [[ -z "$ckpt" ]]; then
        echo "[ERROR] checkpoint が見つかりません: $latest_run" >&2
        return 1
    fi
    echo "$ckpt"
}

if should_run 1; then
    echo "=============================================================="
    echo " Stage 1/3: walk phase  (task=$WALK_TASK, iters=$WALK_ITER)"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$WALK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$WALK_ITER" \
        "$@"
fi

if should_run 2; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT")}"
    echo "=============================================================="
    echo " Stage 2/3: loop_pass  (task=$LOOP_TASK, iters=$ITER)"
    echo " pretrained: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$LOOP_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        --load_pretrained "$WALK_CKPT" \
        "$@"
fi

if should_run 3; then
    LOOP_CKPT="${LOOP_CKPT:-$(find_latest_ckpt "$LOOP_LOG_ROOT")}"
    echo "=============================================================="
    echo " Stage 3/3: loop_pass_360  (task=$LOOP360_TASK, iters=$ITER)"
    echo " pretrained: $LOOP_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$LOOP360_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        --load_pretrained "$LOOP_CKPT" \
        "$@"
fi

echo "[INFO] done."
