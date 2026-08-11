#!/usr/bin/env bash
# walk_long_pass_dr (ロングパス + ボール物性 DR) の学習を実行する。
#
# **継続学習**。学習済みの walk_long_pass ポリシー
# (run 2026-08-09_11-03-31: kick_rate 0.998 / kick_vel_ratio 0.921 / kick_dir_error 4.1°)
# を出発点に、ボール物性 (摩擦・反発・質量) への頑健性だけを足す。
#
# --resume ではなく --load_pretrained を使う理由: experiment_name が
# k1_walk_long_pass_dr と別なので --resume では元 run を検出できない。代わりに env cfg 側
# (_freeze_curricula_at_final) で継承カリキュラムを全部終値に固定してあるので、
# common_step_counter が 0 でも iter 0 から親タスクの収束状態で始まる。
#
# --reset_noise_std は **付けない**。walk_long_pass / walk_mid_kick の失敗記録参照。
#
# ITER は控えめでよい。報酬もコマンド分布も変えていないので、DR への適応だけなら
# 1000-2000 iteration で kick_vel_ratio が落ち着くはず。既定は余裕を見て 3000。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass_dr.sh              # 最新の long_pass ckpt から
#   CKPT=logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_dr.sh          # ckpt を明示
#   ITER=5000 ./scripts/rsl_rl/train_walk_long_pass_dr.sh    # 長く回す

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
ITER=${ITER:-3000}
# 既定は空 = --reset_noise_std を付けない (上のコメント参照)。
RESET_NOISE_STD=${RESET_NOISE_STD-}

TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-DR-v0"
SRC_LOG_ROOT="logs/rsl_rl/k1_walk_long_pass"

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
echo " walk_long_pass_dr  (task=$TASK, iters=$ITER)"
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
