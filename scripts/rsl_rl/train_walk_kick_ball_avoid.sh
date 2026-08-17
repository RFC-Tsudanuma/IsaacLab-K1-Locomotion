#!/usr/bin/env bash
# walk_kick_ball_avoid (Ball Avoidance の原典解釈版) の 2 段階学習を通しで実行する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-v0
#            Stage 1 の重みから始めて、ボール追従とキックを学習する。
#
# 継承元 walk_kick との差は 2 点だけ (詳細は walk_kick_ball_avoid_env_cfg.py の docstring):
#   1. 観測スロット 3 = 現在のボール 3D 位置 (遅延なし ball_pos_rel、元は左足裏 sole_pos)
#   2. approach_penalty → ball_avoidance_exec (接触の瞬間に距離側が 0 になって罰が消える)
#
# 観測は両 stage とも 55 次元・同じ並びなので、入力層と obs normalizer の統計も含めて
# そのまま引き継がれる (--load_pretrained は形の合うテンソルだけをロードする)。
#
# IMPORTANT: 既存の k1_walk_kick_walk_phase / k1_walk_kick の checkpoint は流用しないこと。
#            次元は同じなので --load_pretrained は形の上では通ってしまうが、スロット 3 の
#            意味が違う (左足裏 → ボール 3D 位置)。必ず Stage 1 から通しで回す。
#
# --resume ではなく --load_pretrained を使うのは意図的:
#   --resume は common_step_counter を Stage 1 の到達 iteration に同期させるため、
#   Stage 2 のキック報酬カリキュラム (0 → 500 iteration でフェードイン) が
#   「もう終わった」と判定されて一切ランプしなくなる。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh                 # 通しで実行 (5000/5000)
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh   # env 数を変える
#   ITER=1000 ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh       # 両 stage を短く試す
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh  # stage 1 だけ変える
#   KICK_ITER=8000 ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh  # stage 2 だけ変える
#   STAGE=2 WALK_CKPT=logs/.../model_4999.pt \
#       ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh             # Stage 2 だけ再実行
#   ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh --video         # 追加引数は両 stage に渡る

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
#   LAB_PYTHON=/isaac-sim/python.sh ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh
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
          LAB_PYTHON=/isaac-sim/python.sh ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh
          LAB_PYTHON=python               ./scripts/rsl_rl/train_walk_kick_ball_avoid.sh
EOF
    exit 1
fi
echo "[INFO] python: $LAB_PY"

NUM_ENVS=${NUM_ENVS:-4096}
# ITER で両 stage を一括指定、WALK_ITER / KICK_ITER で個別に上書きできる。
ITER=${ITER:-5000}
WALK_ITER=${WALK_ITER:-$ITER}
KICK_ITER=${KICK_ITER:-$ITER}
STAGE=${STAGE:-all}

WALK_TASK=${WALK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-Walk-Phase-v0"}
KICK_TASK=${KICK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Ball-Avoid-v0"}
# Stage 2 が Stage 1 の checkpoint を拾う先。experiment_name は RunnerCfg 側で決まる
# (k1_walk_kick_ball_avoid_walk_phase)。WALK_TASK を差し替えるときは対で変えること。
WALK_LOG_ROOT=${WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_ball_avoid_walk_phase"}

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
    echo " Stage 1/2: walk phase  (task=$WALK_TASK, iters=$WALK_ITER)"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$WALK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$WALK_ITER" \
        "$@"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_walk_ckpt)}"

    echo "=============================================================="
    echo " Stage 2/2: kick phase (ball avoidance)  (task=$KICK_TASK, iters=$KICK_ITER)"
    echo " pretrained: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$KICK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$KICK_ITER" \
        --load_pretrained "$WALK_CKPT" \
        "$@"
fi

echo "[INFO] done."
