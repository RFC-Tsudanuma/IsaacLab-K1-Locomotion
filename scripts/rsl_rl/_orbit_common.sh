#!/usr/bin/env bash
# orbit 系 (walk_weak_kick_orbit / walk_long_pass_orbit) の通しスクリプト共通部。
#
# 単体では実行しない。各通しスクリプトから source して使う。
# 提供するもの:
#   * REPO_ROOT へ cd 済み (train.py は logs/ を CWD 基準で作るため)
#   * $LAB_PY        … IsaacLab の python (train_walk_kick_360.sh と同じ解決手順)
#   * should_run N   … STAGE 環境変数に N が含まれるか (STAGE=all なら常に真)
#   * find_latest_ckpt DIR … experiment ディレクトリの最新 run の最終 checkpoint
#   * run_stage ...  … 1 段ぶんの train.py 実行 (下の説明参照)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_kick_360.sh と同じ手順)。
# --------------------------------------------------------------------------- #
for _bf in "${BASH_FUNCTIONS:-}" "$HOME/.bash_functions"; do
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
STAGE=${STAGE:-all}

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
find_latest_ckpt() {
    local latest_run ckpt
    latest_run=$(find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
    if [[ -z "$latest_run" ]]; then
        echo "[ERROR] run が見つかりません: $1" >&2
        echo "[ERROR] 先に前段を回すか、<STAGE名>_CKPT で明示してください。" >&2
        return 1
    fi
    ckpt=$(find "$latest_run" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)
    if [[ -z "$ckpt" ]]; then
        echo "[ERROR] checkpoint が見つかりません: $latest_run" >&2
        return 1
    fi
    echo "$ckpt"
}

# run_stage <見出し> <task> <iters> <checkpoint or ""> [追加引数...]
#
# checkpoint が空文字なら --load_pretrained を付けない (Stage 1 用)。
# EXTRA_ARGS (呼び出し側が "$@" を入れる配列) は最後に展開する。
run_stage() {
    local title="$1" task="$2" iters="$3" ckpt="$4"
    shift 4
    echo "=============================================================="
    echo " $title"
    echo " task=$task  iters=$iters  num_envs=$NUM_ENVS"
    [[ -n "$ckpt" ]] && echo " pretrained: $ckpt"
    echo "=============================================================="

    local -a cmd=("$LAB_PY" scripts/rsl_rl/train.py
        --task "$task" --headless
        --num_envs "$NUM_ENVS" --max_iterations "$iters")
    [[ -n "$ckpt" ]] && cmd+=(--load_pretrained "$ckpt")
    cmd+=("$@")

    "${cmd[@]}"
}
