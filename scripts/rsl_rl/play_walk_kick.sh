#!/usr/bin/env bash
# checkpoint のパスからタスク名を推測して play.py を回し、onnx を書き出す。
#
#   ./scripts/rsl_rl/play_walk_kick.sh logs/rsl_rl/k1_walk_kick_360_weak_noisy_ball/<run>/model_4300.pt
#
# は
#
#   ../isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
#       --task Isaac-Velocity-Flat-K1-Walk-Kick-360-Weak-Noisy-Ball-Play-v0 \
#       --headless --checkpoint <上のパス> --num_envs 1
#
# と等価。タスク名の解決は scripts/rsl_rl/resolve_task.py が
# logs/rsl_rl/<experiment_name>/... の <experiment_name> をタスク登録と突き合わせて行う
# (IsaacLab を import しないので一瞬で終わる)。
#
# onnx は play.py が書き出す。出力先は checkpoint と同じ run の exported/ 配下で、
# ファイル名は <experiment_name>_<run 名>.onnx。最後にそのパスを表示する。
#
# 使い方:
#   ./scripts/rsl_rl/play_walk_kick.sh <checkpoint>              # headless, num_envs 1
#   ./scripts/rsl_rl/play_walk_kick.sh <checkpoint> --num_envs 32
#   NO_HEADLESS=1 ./scripts/rsl_rl/play_walk_kick.sh <checkpoint>  # GUI で見る
#   TASK=<明示> ./scripts/rsl_rl/play_walk_kick.sh <checkpoint>    # 推測を上書き
#   EXPORT_ONLY=0 ./scripts/rsl_rl/play_walk_kick.sh <checkpoint>  # onnx 後も回し続ける
#
# 既定 (EXPORT_ONLY=1) は **onnx を書き出した時点で play.py を落とす**。play.py は
# 書き出しの後 ``while simulation_app.is_running()`` で回り続け、--video 以外に
# 終了条件が無いため、onnx が目的だと放っておくと終わらない。挙動を見たいときは
# EXPORT_ONLY=0 にすること。
#
# 追加の引数はそのまま play.py へ渡る。--num_envs や --task を自分で足した場合は
# 既定値を出さないので、二重指定にはならない。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <checkpoint> [play.py への追加引数...]" >&2
    echo "       TASK=<タスク名> で推測を上書き、--list で対応表" >&2
    exit 1
fi

CKPT="$1"; shift
[[ -f "$CKPT" ]] || { echo "[ERROR] checkpoint が無い: $CKPT" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_lob_hist.sh と同じ手順)
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
    for _cand in "$REPO_ROOT/../isaaclab/isaaclab.sh" "$REPO_ROOT/isaaclab.sh" \
                 /workspace/isaaclab/isaaclab.sh /isaac-sim/python.sh; do
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
                LAB_PY="$_cand"; break
            fi
        done
    fi
fi
[[ -n "$LAB_PY" ]] || { echo "[ERROR] IsaacLab の python が見つかりません。LAB_PYTHON で明示してください。" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# resolve_task.py 用の python を探す
#
# **コンテナ内には python3 が PATH に無い** (isaac-lab-base イメージは
# /isaac-sim/kit/python/bin/python3 しか持たない) ので、素直に python3 を呼ぶと
# command not found になる。resolve_task.py は stdlib しか使わないので、
# 見つかったどれで動かしてもよい。最後の手段として $LAB_PY を使う (遅いが確実)。
# --------------------------------------------------------------------------- #
HELPER_PY=""
for _c in python3 python /isaac-sim/kit/python/bin/python3 /usr/bin/python3; do
    if command -v "$_c" >/dev/null 2>&1 || [[ -x "$_c" ]]; then HELPER_PY="$_c"; break; fi
done
[[ -n "$HELPER_PY" ]] || HELPER_PY="$LAB_PY"

# --------------------------------------------------------------------------- #
# タスク名の解決
# --------------------------------------------------------------------------- #
if [[ -n "${TASK:-}" ]]; then
    TASK_NAME="$TASK"
    echo "[INFO] task (TASK で指定): $TASK_NAME"
else
    # 失敗したら resolve_task.py が stderr に候補を出すので、そのまま見せて止める。
    TASK_NAME="$($HELPER_PY scripts/rsl_rl/resolve_task.py "$CKPT")" || {
        echo "[ERROR] タスク名を推測できませんでした。TASK=<タスク名> で指定してください。" >&2
        exit 1
    }
    echo "[INFO] task (パスから推測): $TASK_NAME"
fi

# --------------------------------------------------------------------------- #
# 既定引数。ユーザーが同じものを渡していたら足さない (二重指定を避ける)
# --------------------------------------------------------------------------- #
EXTRA=("$@")
# 空配列を "$@" で渡しても安全 (bash 4.4+)。空文字列 1 個に化ける ${arr[@]:-} は
# 使わないこと。play.py に空の引数が渡ると hydra がオーバーライド指定として解釈して
# "mismatched input '<EOF>'" で落ちる (2026-08-18 に実際に踏んだ)。
has_flag() { local f="$1"; shift; for a in "$@"; do [[ "$a" == "$f" || "$a" == "$f="* ]] && return 0; done; return 1; }

DEFAULTS=()
has_flag --num_envs "${EXTRA[@]}" || DEFAULTS+=(--num_envs 1)
if [[ -z "${NO_HEADLESS:-}" ]]; then
    has_flag --headless "${EXTRA[@]}" || DEFAULTS+=(--headless)
fi

echo "[INFO] python:     $LAB_PY"
echo "[INFO] checkpoint: $CKPT"
echo "[INFO] export_only: ${EXPORT_ONLY:-1}"
echo "=============================================================="

# --------------------------------------------------------------------------- #
# play.py は onnx / jit を **シミュレーションループに入る前** に書き出したあと、
# ``while simulation_app.is_running()`` で回り続ける (--video 以外に終了条件が無い)。
# onnx を取るのが目的のときは書き出した時点で用は済んでいるので、目印の行を見たら
# 落とす。EXPORT_ONLY=0 でこの挙動を切れば、従来どおり回り続ける。
#
# setsid でプロセスグループを分けてから、グループごと TERM する。Isaac Sim は
# 子プロセスを持つので、親だけ kill すると残る。
# --------------------------------------------------------------------------- #
MARKER="Exported policy to:"

if [[ "${EXPORT_ONLY:-1}" == "0" ]]; then
    $LAB_PY scripts/rsl_rl/play.py \
        --task "$TASK_NAME" --checkpoint "$CKPT" "${DEFAULTS[@]}" "${EXTRA[@]}"
else
    LOG="$(mktemp -t play_walk_kick.XXXXXX.log)"
    trap 'rm -f "$LOG"' EXIT
    setsid $LAB_PY scripts/rsl_rl/play.py \
        --task "$TASK_NAME" --checkpoint "$CKPT" "${DEFAULTS[@]}" "${EXTRA[@]}" \
        >"$LOG" 2>&1 &
    PLAY_PID=$!
    tail -f -n +1 "$LOG" & TAIL_PID=$!

    EXPORTED=0
    while kill -0 "$PLAY_PID" 2>/dev/null; do
        if grep -q "$MARKER" "$LOG"; then
            EXPORTED=1
            echo "[INFO] onnx を書き出したので play.py を終了します (EXPORT_ONLY=0 で継続)"
            kill -TERM -"$PLAY_PID" 2>/dev/null || kill -TERM "$PLAY_PID" 2>/dev/null || true
            break
        fi
        sleep 2
    done
    wait "$PLAY_PID" 2>/dev/null || true
    sleep 1; kill "$TAIL_PID" 2>/dev/null || true
    if [[ "$EXPORTED" == "0" ]]; then
        echo "[ERROR] onnx が書き出される前に play.py が終了しました。上のログを確認してください。" >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------------- #
# 書き出された onnx を教える (play.py は run ディレクトリ配下の exported/ に置く)
# --------------------------------------------------------------------------- #
RUN_DIR="$(dirname "$CKPT")"
ONNX="$(find "$RUN_DIR" -name '*.onnx' -newer "$CKPT" 2>/dev/null | sort | tail -n 1)"
[[ -n "$ONNX" ]] || ONNX="$(find "$RUN_DIR" -name '*.onnx' 2>/dev/null | sort | tail -n 1)"
if [[ -n "$ONNX" ]]; then
    echo "[INFO] onnx: $ONNX"
else
    echo "[WARN] onnx が見つかりません。play.py の出力を確認してください。" >&2
fi
