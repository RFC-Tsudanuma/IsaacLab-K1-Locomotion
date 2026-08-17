#!/usr/bin/env bash
# walk_lob_rough（凹凸地形 + 履歴入力のロブキック）の 2 段階学習を通しで実行する。
#
#   Stage 1: Isaac-Velocity-Rough-K1-Walk-Lob-Walk-Phase-v0
#            ボール無し・凹凸地形の上で歩行だけを学習する。履歴入力 (100 フレーム) と
#            内界センサ (IMU / エンコーダ) の遅延 DR はこの段から入る。
#   Stage 2: Isaac-Velocity-Rough-K1-Walk-Lob-v0
#            ボール中心が K1 身長 0.9m を超えるロブキックを獲得する。
#            ボール観測 (視覚) の遅延 DR と、当たり所を下げる 3 つの報酬変更が入る。
#
# **既存 checkpoint は一切流用できない。** この系列は
#   * 観測スロット 3 が左足裏 → ボール 3D 位置 (意味が違う)
#   * actor が ActorCriticHistoryCNN (形が違う)
# の 2 点で walk_lob / walk_kick 系と非互換なので、必ず stage 1 から通すこと。
# stage 2 の引き継ぎ元は k1_walk_lob_rough_walk_phase だけ。
#
# --resume を使わない理由は train_walk_lob.sh の冒頭コメントと同じ
# (common_step_counter が同期されてキック報酬カリキュラムがランプしなくなる)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_lob_rough.sh                     # 通しで実行
#   STAGE=2 ./scripts/rsl_rl/train_walk_lob_rough.sh             # 歩行学習済みなら stage 2 だけ
#   STAGE=1 ./scripts/rsl_rl/train_walk_lob_rough.sh             # walk phase だけ
#   STAGE=2 WALK_CKPT=logs/rsl_rl/k1_walk_lob_rough_walk_phase/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_lob_rough.sh                 # 継承元を明示
#   ITER=25000 ./scripts/rsl_rl/train_walk_lob_rough.sh          # ロブを長く回す
#
# ITER の既定が walk_lob (5000) より大きいのは、stage 2 の kick_plant_foot
# カリキュラム (lon_target −0.42 → −0.03、sigma_lon 0.25 → 0.10) が
# **iteration 4000 まで走る** ため。5000 では目標が動き終わった直後に学習が終わって
# しまい、詰めた軸足配置から apex を伸ばす時間が残らない。最低でも 8000、
# 詰めるなら 20000 以上。
set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_lob.sh と同じ手順)。
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
# lob (Stage 2) の iteration 数。既定が大きい理由は冒頭コメント参照。
ITER=${ITER:-15000}
# walk phase (Stage 1) は歩行を獲得するだけ。凹凸地形と履歴が入るぶん平坦版より
# 収束が遅い想定で、walk_lob の 5000 から少し伸ばしてある。
WALK_ITER=${WALK_ITER:-8000}
STAGE=${STAGE:-all}

WALK_TASK="Isaac-Velocity-Rough-K1-Walk-Lob-Walk-Phase-v0"
LOB_TASK="Isaac-Velocity-Rough-K1-Walk-Lob-v0"
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_lob_rough_walk_phase"

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
    echo " Stage 1/2: walk phase (rough + history)  (task=$WALK_TASK, iters=$WALK_ITER)"
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
    echo " Stage 2/2: lob (rough + history)  (task=$LOB_TASK, iters=$ITER)"
    echo " pretrained: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$LOB_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        --load_pretrained "$WALK_CKPT" \
        "$@"
fi

echo "[INFO] done."
