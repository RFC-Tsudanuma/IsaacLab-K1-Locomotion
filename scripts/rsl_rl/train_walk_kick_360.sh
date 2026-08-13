#!/usr/bin/env bash
# walk_kick_360 の 3 段階学習を通しで実行する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する (全タスク共用)。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-v0
#            限定レンジ (ボール±60°/蹴り±45°, 0.5-0.8m) で地面蹴りを獲得する。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Kick-360-v0
#            全方位 (360°/360°, 0.5-1.5m) + 回り込み。
#
# Stage 2 → 3 の checkpoint 継承が実質カリキュラム (蹴り方は既知・回り込みだけ新規)。
# 全 stage とも観測 55 次元・同じ並びなので、--load_pretrained でそのまま引き継げる。
# --resume を使わない理由は train_walk_kick.sh の冒頭コメントと同じ
# (common_step_counter が同期されてキック報酬カリキュラムがランプしなくなる)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_360.sh              # 通しで実行
#   STAGE=23 ./scripts/rsl_rl/train_walk_kick_360.sh     # 歩行学習済みなら 2,3 だけ
#   STAGE=12 ./scripts/rsl_rl/train_walk_kick_360.sh     # walk_kick までで止める
#   STAGE=3 KICK_CKPT=logs/rsl_rl/k1_walk_kick/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_360.sh          # 既存 walk_kick から 360 だけ
#   ITER=20000 ./scripts/rsl_rl/train_walk_kick_360.sh   # kick 系を長く回す
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_kick_360.sh   # walk phase だけ延長
#   ./scripts/rsl_rl/train_walk_kick_360_rough.sh        # 地形 ablation (凹凸地形) の通し 3 段
#
# iteration 数は walk phase (WALK_ITER) と kick 系 (ITER) で別に持つ。既定はどちらも
# 5000 で、まず一通り通して挙動を確認する想定。仕上げるときは ITER を上げること。

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
# kick 系 (Stage 2/3) の iteration 数。
# NOTE: 詰めたいときは ITER=20000 などに上げること。5000 は「一通り通して挙動を見る」
#       ための既定値。
ITER=${ITER:-5000}
# walk phase (Stage 1) は歩行を獲得するだけなので 5000 で足りる (実績値)。
WALK_ITER=${WALK_ITER:-5000}
STAGE=${STAGE:-all}

# 3 段のタスクと、次段が checkpoint を拾う先の log root。全て上書き可能にしてある。
# 地形 ablation (凹凸地形) は同じ 3 段レシピを rough 版タスクでなぞるので、
# 5 つまとめて差し替えるだけで通しで回せる (専用ラッパ: train_walk_kick_360_rough.sh)。
# タスクを差し替えるときは LOG_ROOT (= RunnerCfg の experiment_name) も必ず対で変えること。
WALK_TASK=${WALK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0"}
KICK_TASK=${KICK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-v0"}
KICK360_TASK=${KICK360_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-360-v0"}
WALK_LOG_ROOT=${WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_walk_phase"}
KICK_LOG_ROOT=${KICK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick"}

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
    echo " Stage 2/3: walk_kick  (task=$KICK_TASK, iters=$ITER)"
    echo " pretrained: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$KICK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        --load_pretrained "$WALK_CKPT" \
        "$@"
fi

if should_run 3; then
    KICK_CKPT="${KICK_CKPT:-$(find_latest_ckpt "$KICK_LOG_ROOT")}"
    echo "=============================================================="
    echo " Stage 3/3: walk_kick_360  (task=$KICK360_TASK, iters=$ITER)"
    echo " pretrained: $KICK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$KICK360_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        --load_pretrained "$KICK_CKPT" \
        "$@"
fi

echo "[INFO] done."
