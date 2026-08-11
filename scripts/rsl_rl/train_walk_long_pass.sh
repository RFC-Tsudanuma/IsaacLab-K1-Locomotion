#!/usr/bin/env bash
# walk_long_pass (5-10 m の強い転がしパス) を **歩行から通しで** 学習する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する (walk_kick と共用)。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0
#            限定レンジ (ボール±60°/蹴り±45°, 0.5-0.8m) で浮かせる蹴りを獲得する。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Loop-Pass-360-v0
#            全方位 (360°/360°, 0.5-1.5m) + 回り込み。
#   Stage 4: Isaac-Velocity-Flat-K1-Walk-Long-Pass-v0
#            速度帯を (2.0,3.0) → (3.2,5.0) へ引き上げる。
#
# Stage 1-3 は train_walk_loop_pass_360.sh と同一の内容。このスクリプトはそこに
# Stage 4 を継ぎ足して 1 本にしたもの。全 stage とも観測 55 次元・行動 12 次元・
# 同じ並びなので、--load_pretrained でそのまま引き継げる。
#
# 各段の checkpoint 継承が実質カリキュラム:
#   歩行 → (蹴り方を覚える) → (回り込みを覚える) → (強く蹴れるようになる)
# 一段飛ばすと前段が獲得した挙動を再発見するところからになるので、順番は変えないこと。
#
# --resume ではなく --load_pretrained を使う理由: experiment_name が段ごとに違うので
# --resume では前段の run を検出できない。加えて --resume は common_step_counter を
# 同期してしまい、各段のキック報酬カリキュラムがランプしなくなる
# (詳細は train_walk_kick.sh の冒頭コメント)。
#
# --reset_noise_std は **Stage 4 だけ** に入れる。収束済みの 360 ポリシーは action std が
# 潰れていて、4-5 m/s は探索したことのない速度域。std を戻さないと慣れた 2-3 m/s の
# 蹴り方に貼り付いたまま抜け出せない。Stage 2/3 では不要 (むしろ蹴り方を壊す)。
#
# ITER は 5000 未満にしないこと。特に Stage 4 は速度帯のカリキュラムが
# 500 → 3000 iteration で (2.0,3.0) → (3.2,5.0) を動かし、その後の収束に
# 2000 iteration を見込んでいる。途中で止めると帯が目標に届いていない中途半端な
# ポリシーになる。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass.sh                  # 4 段を通しで実行
#   STAGE=34 ./scripts/rsl_rl/train_walk_long_pass.sh         # loop_pass まで済みなら 3,4 だけ
#   STAGE=4 ./scripts/rsl_rl/train_walk_long_pass.sh          # 最新の 360 ckpt から Stage 4 だけ
#   STAGE=4 LOOP360_CKPT=logs/rsl_rl/k1_walk_loop_pass_360/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass.sh              # ckpt を明示
#   ITER=10000 ./scripts/rsl_rl/train_walk_long_pass.sh       # 全 kick 段を長く回す
#   LONG_ITER=10000 ./scripts/rsl_rl/train_walk_long_pass.sh  # Stage 4 だけ延長
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_long_pass.sh   # Stage 1 だけ延長
#   RESET_NOISE_STD= ./scripts/rsl_rl/train_walk_long_pass.sh # Stage 4 の std リセット無効
#
# 通しで繋がるかだけ先に確かめる (各段 10 iteration / 64 env):
#   SMOKE=1 ./scripts/rsl_rl/train_walk_long_pass.sh
#   SMOKE=1 SMOKE_ITER=20 ./scripts/rsl_rl/train_walk_long_pass.sh
# 終了時に作られた run の削除コマンドを表示するので、本番の前に必ず消すこと
# (残すと find_latest_ckpt が 10 iteration の checkpoint を出発点に選んでしまう)。
#
# NOTE: 4 段合計 20000 iteration。4096 env で 1 段あたり数時間かかるので、
#       途中で落ちたときは STAGE で残りだけ再開できるようにしてある。

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

# --------------------------------------------------------------------------- #
# SMOKE モード: 各段を数 iteration だけ回して「4 段が通しで繋がるか」だけ確かめる。
#
# rsl_rl は save_interval とは別に **学習終了時に必ず最終 checkpoint を保存する**
# (max_iterations=5000 の run に model_4999.pt が残っているのがその証拠) ので、
# 10 iteration でも model_9.pt ができて次の段へ引き継げる。
#
# 注意: スモークで作った run も logs/rsl_rl/<experiment>/ に残り、日時順で **最新**
#       になる。放置すると次に本番を回したとき find_latest_ckpt が 10 iteration の
#       ゴミを出発点に選んでしまうので、終了時に削除コマンドを表示する。
# --------------------------------------------------------------------------- #
SMOKE=${SMOKE:-0}
if [[ "$SMOKE" != "0" ]]; then
    _DEF_ITER=${SMOKE_ITER:-10}
    _DEF_ENVS=64
    echo "[INFO] SMOKE モード: 各段 ${_DEF_ITER} iteration / ${_DEF_ENVS} env で通し確認します。"
else
    _DEF_ITER=5000
    _DEF_ENVS=4096
fi

NUM_ENVS=${NUM_ENVS:-$_DEF_ENVS}
# kick 系 (Stage 2/3/4) の既定 iteration 数。
ITER=${ITER:-$_DEF_ITER}
# 段ごとの上書き。指定が無ければ ITER (Stage 1 だけ独立)。
WALK_ITER=${WALK_ITER:-$_DEF_ITER}
LOOP_ITER=${LOOP_ITER:-$ITER}
LOOP360_ITER=${LOOP360_ITER:-$ITER}
LONG_ITER=${LONG_ITER:-$ITER}
# Stage 4 のみ。空文字を渡すと --reset_noise_std を付けない。
RESET_NOISE_STD=${RESET_NOISE_STD-0.3}
STAGE=${STAGE:-all}

WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Kick-Walk-Phase-v0"
LOOP_TASK="Isaac-Velocity-Flat-K1-Walk-Loop-Pass-v0"
LOOP360_TASK="Isaac-Velocity-Flat-K1-Walk-Loop-Pass-360-v0"
LONG_TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-v0"

WALK_LOG_ROOT="logs/rsl_rl/k1_walk_kick_walk_phase"
LOOP_LOG_ROOT="logs/rsl_rl/k1_walk_loop_pass"
LOOP360_LOG_ROOT="logs/rsl_rl/k1_walk_loop_pass_360"
LONG_LOG_ROOT="logs/rsl_rl/k1_walk_long_pass"

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# --- SMOKE モードで作られた run を記録しておくための小道具 --------------------- #
_NEW_RUNS=()
snapshot_runs() { find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort; }
record_new_runs() {  # $1 = 実行前のスナップショット, $2 = log root
    [[ "$SMOKE" == "0" ]] && return 0
    local d
    while IFS= read -r d; do
        [[ -z "$d" ]] && continue
        grep -qxF -- "$d" <<<"$1" || _NEW_RUNS+=("$d")
    done <<<"$(snapshot_runs "$2")"
}

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
# 直前の stage をこの実行で回した場合、その run が最新になるので自動で繋がる。
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
    _snap=$(snapshot_runs "$WALK_LOG_ROOT")
    echo "=============================================================="
    echo " Stage 1/4: walk phase  (task=$WALK_TASK, iters=$WALK_ITER)"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$WALK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$WALK_ITER" \
        "$@"
    record_new_runs "$_snap" "$WALK_LOG_ROOT"
fi

if should_run 2; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT")}"
    _snap=$(snapshot_runs "$LOOP_LOG_ROOT")
    echo "=============================================================="
    echo " Stage 2/4: loop_pass  (task=$LOOP_TASK, iters=$LOOP_ITER)"
    echo " pretrained: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$LOOP_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$LOOP_ITER" \
        --load_pretrained "$WALK_CKPT" \
        "$@"
    record_new_runs "$_snap" "$LOOP_LOG_ROOT"
fi

if should_run 3; then
    LOOP_CKPT="${LOOP_CKPT:-$(find_latest_ckpt "$LOOP_LOG_ROOT")}"
    _snap=$(snapshot_runs "$LOOP360_LOG_ROOT")
    echo "=============================================================="
    echo " Stage 3/4: loop_pass_360  (task=$LOOP360_TASK, iters=$LOOP360_ITER)"
    echo " pretrained: $LOOP_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$LOOP360_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$LOOP360_ITER" \
        --load_pretrained "$LOOP_CKPT" \
        "$@"
    record_new_runs "$_snap" "$LOOP360_LOG_ROOT"
fi

if should_run 4; then
    _snap=$(snapshot_runs "$LONG_LOG_ROOT")
    # CKPT は旧インターフェース (Stage 4 単体スクリプトだった頃の名前) の別名として残す。
    LOOP360_CKPT="${LOOP360_CKPT:-${CKPT:-$(find_latest_ckpt "$LOOP360_LOG_ROOT")}}"

    EXTRA_ARGS=()
    if [[ -n "$RESET_NOISE_STD" ]]; then
        EXTRA_ARGS+=(--reset_noise_std "$RESET_NOISE_STD")
    fi

    echo "=============================================================="
    echo " Stage 4/4: long_pass  (task=$LONG_TASK, iters=$LONG_ITER)"
    echo " pretrained: $LOOP360_CKPT"
    echo " reset_noise_std: ${RESET_NOISE_STD:-"(off)"}"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$LONG_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$LONG_ITER" \
        --load_pretrained "$LOOP360_CKPT" \
        "${EXTRA_ARGS[@]}" \
        "$@"
    record_new_runs "$_snap" "$LONG_LOG_ROOT"
fi

echo "[INFO] done."

if [[ "$SMOKE" != "0" ]]; then
    echo
    echo "=============================================================="
    echo " SMOKE 完了: 4 段が通しで繋がることを確認しました。"
    echo "=============================================================="
    if [[ ${#_NEW_RUNS[@]} -eq 0 ]]; then
        echo "[WARN] 新しい run が検出できませんでした。checkpoint の保存を確認してください。"
    else
        echo " 作られた run (本番の前に消すこと。残すと find_latest_ckpt が"
        echo " この ${SMOKE_ITER:-10} iteration の checkpoint を出発点に選んでしまいます):"
        printf '   %s\n' "${_NEW_RUNS[@]}"
        echo
        echo " まとめて削除:"
        echo "   rm -rf ${_NEW_RUNS[*]}"
    fi
fi
