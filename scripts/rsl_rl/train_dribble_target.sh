#!/usr/bin/env bash
# ドリブル (目標点へ運ぶ) の学習を通しで実行する。歩行ポリシーを直接 fine-tune する。
#
#   Stage 0: 歩行ポリシー frozen_policies/best_walk/model_52500.pt (1 ステップ 49 次元) を
#            ドリブル用の観測 (policy 59 / critic 66 次元) に広げる (widen_history_checkpoint.py)。
#            追加分の重みは 0 なので、広げた直後のポリシーは歩行ポリシーと同じ出力を出す
#            (--verify で変換前後の出力一致を数値で確認する)。iteration は 0 に戻す。
#   Stage 1: Isaac-Velocity-Flat-K1-Dribble-Target-WalkPhase-v0 (experiment k1_dribble_target_walk_phase)
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。ドリブル用の 5 項は zero_obs
#            (ノイズ無しで厳密に 0)。
#   Stage 2: Isaac-Velocity-Flat-K1-Dribble-Target-v0 (experiment k1_dribble_target)
#            Stage 1 の最終 checkpoint から --resume --reset_critic で始める。
#            --reset_critic: critic と critic の正規化器を作り直し、common_step_counter を同期しない
#              (ドリブル報酬の 0 → 本番値フェードインを 0 から回すため)。iteration は Stage 1 から続く。
#            --reestimate_obs_norm_terms: Stage 1 で常に 0 だったドリブル用 5 項の actor 正規化統計を、
#              Stage 2 の環境で集めた統計で上書きする (§9-2)。
#
# 既知の現象: Stage 2 の開始直後、転倒率 (Episode_Termination/base_height) が 0.31 → 0.98 に上がり、
# 4096 env では約 800 iteration (= 約 1.57 億サンプル) で回復した
# (2026-10-10 のラン: Stage2 開始から +500 → 0.70、+1000 → 0.31、+1500 → 0.04、+2000 → 0.035)。
# 2048 env (既定) ではサンプル数で揃えて約 1600 iteration が目安。それより前に「回復しない」と
# 判断して止めないこと。比べるときは iteration ではなく累計サンプル数で比べる:
#   python scripts/rsl_rl/stage2_progress.py logs/rsl_rl/k1_dribble_target/<run>
#
# env 数とミニバッチ数の既定値 (NUM_ENVS=2048, MINI_BATCHES=16) について (2026-10-18):
#   このマシン (RTX 4070 SUPER 12GB) では 4096 env が動かない。観測が履歴 100 ステップ × (actor 59 +
#   critic 66 次元) と大きく、rsl_rl の rollout storage だけで 48 step × 4096 env × 12509 × 4B ≈ 9.8GB に
#   なるため。2048 env でも PPO 更新 (mirror loss) で数百 MB 足りなかったので、ミニバッチ数を
#   8 → 16 にして 1 回の更新のメモリを半分にした (1 iteration のサンプル数は 2048 × 48)。
#   GPU メモリに余裕のあるマシンで過去のラン (4096 env / 8 ミニバッチ) と揃えたいときは
#   NUM_ENVS=4096 MINI_BATCHES=8 で上書きする。
#
# 使い方:
#   ./scripts/rsl_rl/train_dribble_target.sh                     # 通しで実行
#   NUM_ENVS=4096 MINI_BATCHES=8 ./scripts/rsl_rl/train_dribble_target.sh
#                                                                # 大きい GPU で過去のランと同じ設定にする
#   ITER=50 ./scripts/rsl_rl/train_dribble_target.sh             # 短く試す (各 stage 50 iteration)
#   STAGE=2 WALK_CKPT=logs/rsl_rl/k1_dribble_target_walk_phase/<run>/model_19999.pt \
#       ./scripts/rsl_rl/train_dribble_target.sh                 # Stage 2 だけ再実行
#   ./scripts/rsl_rl/train_dribble_target.sh --video             # 追加引数は Stage 1/2 に渡る


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
#   LAB_PYTHON=/isaac-sim/python.sh ./scripts/rsl_rl/train_dribble_target.sh
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
          LAB_PYTHON=/isaac-sim/python.sh ./scripts/rsl_rl/train_dribble_target.sh
          LAB_PYTHON=python               ./scripts/rsl_rl/train_dribble_target.sh
EOF
    exit 1
fi
echo "[INFO] python: $LAB_PY"

NUM_ENVS=${NUM_ENVS:-2048}
MINI_BATCHES=${MINI_BATCHES:-16}
ITER=${ITER:-20000}
STAGE=${STAGE:-all}

BASE_CKPT=${BASE_CKPT:-"frozen_policies/best_walk/model_52500.pt"}
WIDENED_CKPT=${WIDENED_CKPT:-"logs/rsl_rl/k1_dribble_target_walk_phase/best_walk_widened/model_0.pt"}
WALK_TASK="Isaac-Velocity-Flat-K1-Dribble-Target-WalkPhase-v0"
DRIBBLE_TASK="Isaac-Velocity-Flat-K1-Dribble-Target-v0"
WALK_LOG_ROOT="logs/rsl_rl/k1_dribble_target_walk_phase"
DRIBBLE_OBS_TERMS="target_pos_rel,ball_pos_rel,ball_vel,foot_left_ball_pos_rel,foot_right_ball_pos_rel"

# Stage 1 の最新 run から最終 checkpoint を拾う (best_walk_widened は除く)。
find_latest_walk_ckpt() {
    local latest_run
    latest_run=$(find "$WALK_LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -n 1)
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

if [[ "$STAGE" == "all" || "$STAGE" == "0" || "$STAGE" == "1" ]]; then
    if [[ ! -f "$WIDENED_CKPT" || "$STAGE" == "0" ]]; then
        echo "=============================================================="
        echo " Stage 0: widen $BASE_CKPT (49 -> 59 dims)"
        echo "=============================================================="
        $LAB_PY scripts/rsl_rl/widen_history_checkpoint.py \
            --input "$BASE_CKPT" --output "$WIDENED_CKPT" --verify
    fi
fi

if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo "=============================================================="
    echo " Stage 1/2: walk phase  (task=$WALK_TASK, iters=$ITER)"
    echo " init: $WIDENED_CKPT"
    echo "=============================================================="
    # best_walk は action noise std が収束済みなので --reset_noise_std 0.05 が必須
    # (frozen_policies/best_walk/README.md)。広げた checkpoint は optimizer の状態を持たない。
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$WALK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        agent.algorithm.num_mini_batches="$MINI_BATCHES" \
        --max_iterations "$ITER" \
        --resume --checkpoint "$(realpath "$WIDENED_CKPT")" \
        --reset_noise_std 0.05 \
        "$@"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "2" ]]; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_walk_ckpt)}"

    echo "=============================================================="
    echo " Stage 2/2: dribble  (task=$DRIBBLE_TASK, iters=$ITER)"
    echo " init: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$DRIBBLE_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        agent.algorithm.num_mini_batches="$MINI_BATCHES" \
        --max_iterations "$ITER" \
        --resume --checkpoint "$(realpath "$WALK_CKPT")" \
        --reset_critic \
        --reestimate_obs_norm_terms "$DRIBBLE_OBS_TERMS" \
        "$@"
fi

echo "[INFO] done."
