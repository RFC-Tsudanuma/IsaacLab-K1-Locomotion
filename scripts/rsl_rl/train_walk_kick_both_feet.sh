#!/usr/bin/env bash
# 両足で蹴れる walk_kick の 2 段階学習を通しで実行する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-v0
#            Stage 1 の重みから始めて、ボール追従とキックを学習する。
#
# 既存の walk_kick との差は 2 点だけ (詳細は walk_kick_both_feet_env_cfg.py の docstring):
#   1. 観測スロット 3 が「左足裏の位置」→「ボール 3D 位置 (1 ステップ遅延)」
#   2. 歩行位相の初期オフセットをエピソードごとに {0, π} で振る
# どちらも「蹴り足が右に固定される」構造的な原因を潰すためのもの。
#
# IMPORTANT: **既存の k1_walk_kick_walk_phase の checkpoint は使わないこと。**
#            policy は 55 次元・同じ並びなので --load_pretrained が通ってしまうが、
#            スロット 3 の意味が違うので入力の解釈がずれる。critic は 61 → 58 次元に
#            変わっているのでそもそも形が合わない。Stage 1 から回すこと。
#
# --resume ではなく --load_pretrained を使う理由は train_walk_kick.sh の冒頭コメントと
# 同じ (common_step_counter が同期されてキック報酬カリキュラムがランプしなくなる)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_both_feet.sh                # 通しで実行
#   STAGE=1 ./scripts/rsl_rl/train_walk_kick_both_feet.sh        # walk phase だけ
#   STAGE=2 ./scripts/rsl_rl/train_walk_kick_both_feet.sh        # キックだけ (最新の walk 重みを自動で拾う)
#   STAGE=2 WALK_CKPT=logs/rsl_rl/k1_walk_kick_both_feet_walk_phase/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_both_feet.sh            # 継承元を明示
#   NUM_ENVS=2048 ITER=10000 ./scripts/rsl_rl/train_walk_kick_both_feet.sh
#
# 効果の確認: TensorBoard の Metrics/kick_direction/kick_foot_right_frac
#            (0 = 常に左足, 1 = 常に右足)。0.5 付近に寄れば両足で蹴れている。

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
# キック (Stage 2) の iteration 数。キック報酬のカリキュラムは 500 iteration で
# 立ち上がりきるので、それ以上あれば形にはなる。詰めるなら 20000。
ITER=${ITER:-20000}
# 歩行 (Stage 1) は歩行を獲得するだけなので 5000 で足りる (walk_kick 系の実績値)。
WALK_ITER=${WALK_ITER:-5000}
STAGE=${STAGE:-all}

WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-Walk-Phase-v0"
KICK_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-Both-Feet-v0"
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_both_feet_walk_phase"

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
    echo " Stage 1/2: walk phase  (task=$WALK_TASK, iters=$WALK_ITER)"
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
    echo " Stage 2/2: kick  (task=$KICK_TASK, iters=$ITER)"
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

echo "[INFO] done."
