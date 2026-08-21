#!/usr/bin/env bash
# walk_long_pass_history (ロングパス + 短期 I/O 履歴) の学習を実行する。
#
# **継続学習**。学習済みの walk_long_pass ポリシーを出発点に、policy 観測の本体状態
# 5 項 (projected_gravity / base_ang_vel / joint_pos / joint_vel / prev_joint_request)
# に 0.1 秒 = 5 ステップの履歴を付ける。arXiv:2401.16889 の short history 相当。
# ネットワーク構造・報酬・コマンド分布・カリキュラム・行動空間は変えない。
# ミラー可能にするため、policy / critic の左足裏 3D スロットはボール位置へ変える。
# PPO には係数 0.5 の mirror loss を追加し、data augmentation は使わない。
#
# 他の train_*.sh と決定的に違う点: **checkpoint をそのまま渡せない**。
# policy 観測が 55 -> 223 次元になるので、train.py の --load_pretrained
# (形の合わないテンソルを捨てる実装) に生の ckpt を渡すと actor の入力層と
# actor_obs_normalizer が落ちてポリシーが壊れる。
#
# しかも **flag 版の expand_checkpoint_kick_flag.py も使えない**。あちらは末尾に
# ゼロを足すだけだが、履歴化は各項をその場で 5 倍に展開するので 55 次元の並びが
# 223 次元の中に散らばる (joint_pos は index 11-22 -> 35-94 へ移動)。
# 専用の expand_checkpoint_history.py が列を並べ替える。元の重みは各履歴ブロックの
# 最新スロットに入り、過去 4 スロットは 0 になる。ただし旧 sole_pos の重みと
# 正規化統計が新しい ball_pos に適用されるため、これは形状互換な近似初期化であり、
# 元のポリシーとの挙動一致は保証しない。
#
# --resume ではなく --load_pretrained を使う理由: experiment_name が
# k1_walk_long_pass_history と別なので --resume では元 run を検出できない。代わりに env cfg
# 側 (_freeze_curricula_at_final) で継承カリキュラムを全部終値に固定してあるので、
# common_step_counter が 0 でも iter 0 から親タスクの収束状態で始まる。
#
# --reset_noise_std は **付けない**。walk_long_pass / walk_mid_kick の失敗記録参照。
# 行動空間は変わっていないので、探索は元の std のままで足りている。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass_history.sh              # 最新の long_pass ckpt から
#   CKPT=logs/rsl_rl/k1_walk_long_pass/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_history.sh          # ckpt を明示
#   SRC=k1_walk_long_pass_dr ./scripts/rsl_rl/train_walk_long_pass_history.sh  # DR 版から
#   ITER=5000 ./scripts/rsl_rl/train_walk_long_pass_history.sh    # 長く回す
#
# 見るべきもの (TensorBoard):
#   Metrics/kick_direction/kick_rate       … 観測変更と mirror loss 導入の過渡を監視
#   Metrics/kick_direction/kick_vel_ratio  … 履歴で改善するか
#   Metrics/kick_direction/kick_dir_error_deg … 同上
#   Train/mean_episode_length              … 転倒が減れば伸びる

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_long_pass_flag.sh と同じ手順)。
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
# 履歴スロット数。env cfg の _HISTORY_LEN と必ず揃えること。
HISTORY_LEN=${HISTORY_LEN:-5}

TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-History-v0"
SRC_LOG_ROOT="logs/rsl_rl/$SRC"

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
# checkpoint の actor 入力層を 55 -> 223 に並べ替える (critic/行動は据え置き)。
# --------------------------------------------------------------------------- #
EXPANDED="${EXPANDED:-logs/rsl_rl/_expanded/history_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}"
mkdir -p "$(dirname "$EXPANDED")"

echo "=============================================================="
echo " expand checkpoint  ($CKPT -> $EXPANDED)"
echo "=============================================================="
$LAB_PY scripts/rsl_rl/expand_checkpoint_history.py \
    "$CKPT" -o "$EXPANDED" --history-len "$HISTORY_LEN"

echo "=============================================================="
echo " walk_long_pass_history  (task=$TASK, iters=$ITER)"
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
