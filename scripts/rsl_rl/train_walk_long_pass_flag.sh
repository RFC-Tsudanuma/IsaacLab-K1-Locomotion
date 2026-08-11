#!/usr/bin/env bash
# walk_long_pass_flag (ロングパス + キック検出フラグ) の学習を実行する。
#
# **継続学習**。学習済みの walk_long_pass ポリシーを出発点に、
# 「このエピソードで既にボールを蹴ったか」を 0/1 で出力する行動 1 次元を足す。
# 蹴り方の報酬・コマンド分布・カリキュラムは一切変えない。
#
# 他の train_*.sh と決定的に違う点: **checkpoint をそのまま渡せない**。
# このタスクだけ観測 55 -> 56 / 行動 12 -> 13 (critic 61 -> 69) と次元が違うので、
# train.py の --load_pretrained (形の合わないテンソルを捨てる実装) に生の ckpt を
# 渡すと actor の入力層と出力層が両方ランダム初期化されてポリシーが壊れる。
# そこで expand_checkpoint_kick_flag.py でゼロパディングしてから渡す。
# パディングした列/行は 0 なので、拡張直後のポリシーは元と挙動が完全に一致する。
#
# --resume ではなく --load_pretrained を使う理由: experiment_name が
# k1_walk_long_pass_flag と別なので --resume では元 run を検出できない。代わりに env cfg 側
# (_freeze_curricula_at_final) で継承カリキュラムを全部終値に固定してあるので、
# common_step_counter が 0 でも iter 0 から親タスクの収束状態で始まる。
#
# --reset_noise_std は **付けない**。walk_long_pass / walk_mid_kick の失敗記録参照。
# フラグ次元だけは expand スクリプトが std=0.5 を入れるので、探索は足りている。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass_flag.sh              # 最新の long_pass ckpt から
#   CKPT=logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_flag.sh          # ckpt を明示
#   SRC=k1_walk_long_pass_dr ./scripts/rsl_rl/train_walk_long_pass_flag.sh  # DR 版から
#   ITER=5000 ./scripts/rsl_rl/train_walk_long_pass_flag.sh    # 長く回す
#
# 途中まで学習した flag run の **続き** を回すとき (詳細は下の RESUME ブロック):
#   RESUME=1 ITER=3100 ./scripts/rsl_rl/train_walk_long_pass_flag.sh
# ITER は「追加で回す数」。expand も --load_pretrained も通らない。
#
# 見るべきもの (TensorBoard):
#   Metrics/kick_direction/flag_accuracy        … **まずこれ**。単純な正解率 (1.0 が満点)
#   Metrics/kick_direction/kick_rate            … 0.99 付近を維持するはず
#   Metrics/kick_direction/flag_pred_final      … kick_rate に追いつけば成功
#   Metrics/kick_direction/flag_pre_latch_pred  … 誤検出。0 に近いほど良い
#   Metrics/kick_direction/flag_err_mean        … 確率の絶対誤差 (正解率とは別物)

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_long_pass_dr.sh と同じ手順)。
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
# 出発点の experiment。DR 版から始めたいときは SRC=k1_walk_long_pass_dr。
SRC=${SRC:-k1_walk_long_pass}
# フラグ次元の初期ノイズ std (expand スクリプトへ渡す)。
FLAG_STD=${FLAG_STD:-0.5}

TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Flag-v0"
SRC_LOG_ROOT="logs/rsl_rl/$SRC"

# --------------------------------------------------------------------------- #
# RESUME=1: 既存の k1_walk_long_pass_flag run の続きを学習する。
#
# **新規学習の経路 (下の expand → --load_pretrained) を通してはいけない。**
# あちらは long_pass の checkpoint を 0 埋めし直して出発点にするので、
# それまでに積んだフラグ学習が丸ごと捨てられる。
#
# --load_pretrained ではなく --resume を使う理由:
#   * experiment_name が同じなので --resume が run を検出できる (他の段と違う点)
#   * optimizer state (Adam のモーメント、adaptive LR の状態) まで引き継げる
#   * checkpoint は既に 56/13/69 なので拡張は不要
#   * このタスクはカリキュラムを全部終値に固定してあるので、--resume が
#     common_step_counter を同期しても何も変わらない (害も利も無い)
#
# **--max_iterations は「追加で回す数」** であって目標値ではない。
# rsl_rl の learn() は total_it = start_it + num_learning_iterations で回すので、
# iteration 1900 から ITER=3100 を指定すると 5000 で止まる。
#
# --reset_noise_std は **付けないこと**。train.py はこれが指定されると
# optimizer state のロードを丸ごと飛ばすので、resume の意味が薄れる。
#
# 使い方:
#   RESUME=1 ITER=3100 ./scripts/rsl_rl/train_walk_long_pass_flag.sh
#   RESUME=1 LOAD_RUN=2026-08-11_03-38-23 ./scripts/rsl_rl/train_walk_long_pass_flag.sh
#
# NOTE: resume でも log_dir は新しいタイムスタンプで作られる (IsaacLab の仕様)。
#       TensorBoard の曲線は run をまたいで分割されるので、まとめて見るときは
#       logs/rsl_rl/k1_walk_long_pass_flag を丸ごと指定すること。
# --------------------------------------------------------------------------- #
RESUME=${RESUME:-0}
if [[ "$RESUME" != "0" ]]; then
    RESUME_ARGS=(--resume)
    if [[ -n "${LOAD_RUN:-}" ]]; then
        RESUME_ARGS+=(--load_run "$LOAD_RUN")
    fi
    echo "=============================================================="
    echo " walk_long_pass_flag [RESUME]  (task=$TASK, +${ITER} iterations)"
    echo " load_run: ${LOAD_RUN:-"(最新)"}"
    echo " NOTE: ITER は追加で回す数。現在の iteration からの相対値です。"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$ITER" \
        "${RESUME_ARGS[@]}" \
        "$@"
    echo "[INFO] done."
    exit 0
fi

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

# --------------------------------------------------------------------------- #
# checkpoint をゼロパディングして 56/13 (critic 69) にする。
# --------------------------------------------------------------------------- #
EXPANDED="${EXPANDED:-logs/rsl_rl/_expanded/$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
mkdir -p "$(dirname "$EXPANDED")"

echo "=============================================================="
echo " expand checkpoint  ($CKPT -> $EXPANDED)"
echo "=============================================================="
$LAB_PY scripts/rsl_rl/expand_checkpoint_kick_flag.py \
    "$CKPT" -o "$EXPANDED" --flag-std "$FLAG_STD"

echo "=============================================================="
echo " walk_long_pass_flag  (task=$TASK, iters=$ITER)"
echo " pretrained: $EXPANDED  (from $CKPT)"
echo "=============================================================="
$LAB_PY scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$ITER" \
    --load_pretrained "$EXPANDED" \
    "$@"

echo "[INFO] done."
