#!/usr/bin/env bash
# walk_long_pass (5-10 m の強い転がしパス) の学習を実行する。
#
# loop_pass_360 の checkpoint からの fine-tune 一段のみ。歩行 → loop_pass → 360 の
# 3 段はすでに済んでいる前提で、その最終 checkpoint を出発点に速度帯 (3.2-5.0 m/s)
# だけを新しく学習する (蹴り方・回り込みは既知)。
#
# --reset_noise_std を既定で入れる理由: 収束済みの 360 ポリシーは action std が
# 潰れていて、4-5 m/s は探索したことのない速度域。std を戻さないと慣れた 2-3 m/s の
# 蹴り方に貼り付いたまま抜け出せない。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass.sh                # 最新の 360 ckpt から
#   CKPT=logs/rsl_rl/k1_walk_loop_pass_360/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass.sh            # ckpt を明示
#   ITER=10000 ./scripts/rsl_rl/train_walk_long_pass.sh     # 長く回す
#   RESET_NOISE_STD= ./scripts/rsl_rl/train_walk_long_pass.sh   # std リセット無効

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_loop_pass_360.sh と同じ手順)。
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
ITER=${ITER:-5000}
# 空文字を渡すと --reset_noise_std を付けない。
RESET_NOISE_STD=${RESET_NOISE_STD-0.3}

TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-v0"
SRC_LOG_ROOT="logs/rsl_rl/k1_walk_loop_pass_360"

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

CKPT="${CKPT:-$(find_latest_ckpt "$SRC_LOG_ROOT")}"

EXTRA_ARGS=()
if [[ -n "$RESET_NOISE_STD" ]]; then
    EXTRA_ARGS+=(--reset_noise_std "$RESET_NOISE_STD")
fi

echo "=============================================================="
echo " walk_long_pass  (task=$TASK, iters=$ITER)"
echo " pretrained: $CKPT"
echo " reset_noise_std: ${RESET_NOISE_STD:-"(off)"}"
echo "=============================================================="
$LAB_PY scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$ITER" \
    --load_pretrained "$CKPT" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[INFO] done."
