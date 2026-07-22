#!/usr/bin/env bash
# walk_kick の 2 段階学習を通しで実行する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-v0
#            Stage 1 の重みから始めて、ボール追従とキックを学習する。
#
# 観測は両 stage とも 55 次元・同じ並びなので、入力層と obs normalizer の統計も含めて
# そのまま引き継がれる (--load_pretrained は形の合うテンソルだけをロードする)。
#
# --resume ではなく --load_pretrained を使うのは意図的:
#   --resume は common_step_counter を Stage 1 の到達 iteration に同期させるため、
#   Stage 2 のキック報酬カリキュラム (0 → 500 iteration でフェードイン) が
#   「もう終わった」と判定されて一切ランプしなくなる。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick.sh                  # 通しで実行
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_walk_kick.sh    # env 数を変える
#   ITER=1000 ./scripts/rsl_rl/train_walk_kick.sh        # 短く試す
#   STAGE=2 WALK_CKPT=logs/.../model_19999.pt ./scripts/rsl_rl/train_walk_kick.sh
#                                                        # Stage 2 だけ再実行
#   ./scripts/rsl_rl/train_walk_kick.sh --video          # 追加引数は両 stage に渡る

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する
# (既存の logs/rsl_rl/... はルートにある)。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する。
#
# 環境ごとに置き場所が違う (ホストでは ~/.bash_functions の _labpython2、
# コンテナでは isaac-sim の python.sh など) ので、決め打ちせず順に探す。
# 明示したいときは LAB_PYTHON で上書きする:
#   LAB_PYTHON=/isaac-sim/python.sh ./scripts/rsl_rl/train_walk_kick.sh
# --------------------------------------------------------------------------- #
# bash_functions があれば読む。_labpython2 は関数なので、サブシェルではなく
# このシェルで source しないと後段の呼び出しまで残らない。
for _bf in "${BASH_FUNCTIONS:-}" "$HOME/.bash_functions" /home/satoshi/.bash_functions; do
    if [[ -n "$_bf" && -f "$_bf" ]]; then
        # shellcheck disable=SC1090
        source "$_bf"
        break
    fi
done

LAB_PY=""
if [[ -n "${LAB_PYTHON:-}" ]]; then
    # 1. 明示指定
    LAB_PY="$LAB_PYTHON"
elif type _labpython2 >/dev/null 2>&1; then
    # 2. bash_functions の _labpython2 (ホスト環境)
    LAB_PY="_labpython2"
else
    # 3. IsaacLab / Isaac Sim の標準的な python (コンテナ環境)
    for _cand in "$REPO_ROOT/isaaclab.sh" /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
        if [[ -x "$_cand" ]]; then
            case "$_cand" in
                *isaaclab.sh) LAB_PY="$_cand -p" ;;
                *)            LAB_PY="$_cand" ;;
            esac
            break
        fi
    done
    # 4. isaaclab が import できる python が PATH にあるか
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
    cat >&2 <<'EOF'
[ERROR] IsaacLab の python が見つかりません。
        LAB_PYTHON に明示してから再実行してください。例:
          LAB_PYTHON=/isaac-sim/python.sh ./scripts/rsl_rl/train_walk_kick.sh
          LAB_PYTHON=python               ./scripts/rsl_rl/train_walk_kick.sh
EOF
    exit 1
fi
echo "[INFO] python: $LAB_PY"

NUM_ENVS=${NUM_ENVS:-4096}
ITER=${ITER:-20000}
STAGE=${STAGE:-all}

WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0"
# Stage 2 のタスクは差し替え可能。walk phase (Stage 1) は 3 タスク共通なので、
# パス／ループシュートもこのスクリプトをそのまま使い回せる。
#   KICK_TASK=Isaac-Velocity-Flat-K1-Walk-Pass-v0 ./scripts/rsl_rl/train_walk_kick.sh
#   KICK_TASK=Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0  ./scripts/rsl_rl/train_walk_kick.sh
#   KICK_TASK=Isaac-Velocity-Flat-K1-Walk-Loop-Shoot-v0 ./scripts/rsl_rl/train_walk_kick.sh
KICK_TASK=${KICK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-v0"}
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_walk_phase"

# Stage 1 の最新 run から最終 checkpoint を拾う。
# run ディレクトリ名は YYYY-MM-DD_HH-MM-SS なので辞書順 = 時刻順。
# model_*.pt は model_9.pt < model_10.pt を正しく比べるため sort -V を使う。
find_latest_walk_ckpt() {
    local latest_run
    latest_run=$(find "$WALK_LOG_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    if [[ -z "$latest_run" ]]; then
        echo "[ERROR] Stage 1 の run が見つかりません: $WALK_LOG_ROOT" >&2
        return 1
    fi
    local ckpt
    ckpt=$(find "$latest_run" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)
    if [[ -z "$ckpt" ]]; then
        echo "[ERROR] checkpoint が見つかりません: $latest_run" >&2
        return 1
    fi
    echo "$ckpt"
}

if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo "=============================================================="
    echo " Stage 1/2: walk phase  (task=$WALK_TASK, iters=$ITER)"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$WALK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        "$@"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_walk_ckpt)}"

    echo "=============================================================="
    echo " Stage 2/2: kick phase  (task=$KICK_TASK, iters=$ITER)"
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
