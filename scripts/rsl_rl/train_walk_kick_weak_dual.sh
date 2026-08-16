#!/usr/bin/env bash
# walk_weak_kick の dual encoder 版 (actor が 100 フレームの観測履歴を見る) を通しで学習する。
#
#   Stage 1: (学習しない) 履歴入力版の walk phase checkpoint を再利用する
#            logs/rsl_rl/k1_walk_kick_dual_walk_phase/<run>/model_<N>.pt
#            (= train_walk_kick_dual.sh の Stage 1。無ければ both_feet 系の
#             1 フレーム版 k1_walk_kick_both_feet_walk_phase にフォールバックし、
#             --warm_start_from_single_frame が自動で付く)
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Weak-v0
#            -> k1_walk_kick_dual_weak
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-Weak-v0
#            -> k1_walk_kick_dual_360_weak   (全方位・**クリーン**。観測 DR は無し)
#            (σ_direction のアニール 0.35 → 0.15 はこの段で回る)
#   Stage 4: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-Weak-DR-v0
#            -> k1_walk_kick_dual_360_weak_dr  (**最終 stage**。観測 DR)
#
# 移植元は fewa/walk_kick_dual_encoder_tune (walk_long_pass の dual encoder 化)。
# **4 段構成と DR の配置は fewa の 47b8863 に合わせてある。**
# コマンド・ボール配置・終了条件は既存の weak タスクと 1 バイトも変えていない。差は
#   * 全 stage: policy 観測が 100 フレームの履歴 (actor = 直近 5 フレーム + CNN 潜在)
#   * 全 stage: 着地 shaping 3 項 (feet_landing_impact / feet_landing_vel /
#               feet_heel_strike) を外す + feet_phase の weight 2.0 → 0.8
#               (fewa 実測: 未対処だと「蹴らずに歩く」が最適解になり kick_rate が
#                0.19-0.28 で 4000 iteration 停滞した)
#   * Stage 3 のみ: σ_direction のアニール (方向の採点を 0.35 → 0.15 rad に締める)
#   * Stage 4 のみ: 観測 DR = IMU/エンコーダの遅延 (≤0.02 s) + **ボール観測の遅延
#                   (0.02-0.10 s) と一様ノイズの拡大 (位置 ±0.07 m / 速度 ±0.5 m/s)**
#                   + ランプの全凍結 + σ_direction を 0.15 で固定 (地形は平面のまま)
# (source/.../walk_weak_kick_dual/walk_weak_kick_dual_env_cfg.py 参照)。
#
# ボール観測 DR は **一様ノイズ + 遅延** (fewa 準拠)。ガウスの認識パイプライン
# (旧 Noisy-Ball 系) は fewa の実機実績に合わせて不採用 (2026-08-17)。
#
# NOTE: 位置ノイズは ±0.07 m を超えないこと。fewa は一度 ±0.1 m まで上げて critic が
#       繰り返し発散した (10000 iteration 中 value loss > 50 が 230 回、最大 1.26e12)。
#
# 歩行 (stage 1) は walk_kick_dual とまったく同じなので学習し直さない。作り直すのは
# キック側 (stage 2 以降) だけ。walk_kick_360 のポリシーからの fine-tune にしないのは、
# あちらが「指令を無視して全力で蹴る」に収束していて探索 std も潰れているため
# (詳細は walk_weak_kick_env_cfg.py の docstring)。
#
# 既存の 1 フレーム checkpoint から始める場合
# -------------------------------------------
# checkpoint に actor.0.weight があれば 1 フレーム観測版と判定し、
# --warm_start_from_single_frame を自動で付ける (旧 actor を履歴 actor の
# 「最新フレームの列」へ移植するので、学習開始時点の挙動が旧ポリシーと一致する)。
#
# **使えるのは both_feet 系の checkpoint だけ** (例:
# logs/rsl_rl/k1_walk_kick_both_feet_walk_phase/<run>/model_*.pt、
# logs/rsl_rl/k1_walk_kick_both_feet/<run>/model_*.pt)。dual の観測は both_feet 版
# (スロット 3 = ボール 3D 位置、critic 58 次元) なので、**旧 sole_pos 系
# (k1_walk_kick_walk_phase など) は不可**: policy は同じ 55 次元なので
# --load_pretrained は形の上では通ってしまうが、スロット 3 の意味が違うので
# 入力の解釈がずれる (critic は次元も合わない)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_weak_dual.sh                 # stage 2,3,4 を通しで実行
#   ITER=5000 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh       # stage 2,3 の iteration
#   DR_ITER=6000 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh    # stage 4 だけ変える
#   STAGE=2 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh         # stage 2 だけ
#   STAGE=34 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh        # stage 3,4 だけ
#   STAGE=4 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh         # stage 4 (DR) だけ
#   STAGE=3 KICK_CKPT=logs/rsl_rl/k1_walk_kick_dual_weak/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_weak_dual.sh             # 既存の dual weak から 360 だけ
#   WALK_CKPT=logs/.../model_4999.pt ./scripts/rsl_rl/train_walk_kick_weak_dual.sh
#                                                                 # 別の walk phase から
#
# NOTE: **Stage 3 の ITER は 3000 以上**にすること。σ_direction のアニールが
#       1500 → 3000 iteration で終点に着く。kick_rate が 1.0 から落ちたら σ が
#       深すぎるサイン (env cfg のコメント参照)。加えてキック報酬のカリキュラム
#       (strong の立ち上げ→フェードアウト、σ アニール、overshoot 罰のフェードイン) が
#       3000 iteration で終点に着くので、それより短いと「まだ強く蹴った方が得」な
#       途中状態で終わる。既定は 5000。
# NOTE: Stage 4 (DR) の既定は 10000 (fewa は Stage 1-3 各 5000 + Stage 4 10000 で
#       通し検証し、ベストは model_6600)。DR に慣れるまで kick_rate が一度下がるので、
#       短く切ると「ノイズを避けて蹴らない」状態で終わりかねない。
# NOTE: --reset_noise_std は付けないこと (env cfg の docstring 参照)。
# NOTE: 履歴長 H = 100 を変えると checkpoint 同士が繋がらない。実機側の
#       リングバッファ長も (1, H, 55) に合わせること。

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_kick_dual.sh と同じ手順)。
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
# 蹴り段 (Stage 2/3) の既定 iteration。段別に上書きできる。
ITER=${ITER:-5000}
KICK_ITER=${KICK_ITER:-$ITER}
KICK360_ITER=${KICK360_ITER:-$ITER}
# Stage 4 (DR) だけ既定が長い (上の NOTE 参照)。
DR_ITER=${DR_ITER:-10000}
# stage 1 は学習しないので、既定で stage 2,3,4 だけを回す。
STAGE=${STAGE:-234}

KICK_TASK=${KICK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Weak-v0"}
KICK360_TASK=${KICK360_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-Weak-v0"}
DR_TASK=${DR_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-Weak-DR-v0"}

# 履歴入力版の log ルート。既存の 1 フレーム版 (k1_walk_kick_weak など) とは別なので、
# 既存 run が誤って引き継ぎ元に選ばれることはない。
WALK_LOG_ROOT=${WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_dual_walk_phase"}
KICK_LOG_ROOT=${KICK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_dual_weak"}
KICK360_LOG_ROOT=${KICK360_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_dual_360_weak"}

# 履歴入力版の walk phase が無いときのフォールバック (both_feet 系の 1 フレーム観測)。
# --warm_start_from_single_frame が自動で付くのでそのまま使える。
#
# 旧 sole_pos 系 (k1_walk_kick_walk_phase) は **フォールバック先にしない**。
# スロット 3 の意味が違うので warm start 元として不適 (冒頭コメント参照)。
FALLBACK_WALK_LOG_ROOT=${FALLBACK_WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_both_feet_walk_phase"}

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
find_latest_ckpt() {
    local root="$1" hint="${2:-}" latest_run ckpt
    if [[ ! -d "$root" ]]; then
        echo "[ERROR] 前段の run がありません: $root" >&2
        [[ -n "$hint" ]] && echo "[ERROR] $hint" >&2
        return 1
    fi
    latest_run=$(find "$root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
    if [[ -z "$latest_run" ]]; then
        echo "[ERROR] 前段の run がありません: $root" >&2
        [[ -n "$hint" ]] && echo "[ERROR] $hint" >&2
        return 1
    fi
    ckpt=$(find "$latest_run" -maxdepth 1 -name 'model_*.pt' 2>/dev/null | sort -V | tail -n 1)
    if [[ -z "$ckpt" ]]; then
        echo "[ERROR] checkpoint が見つかりません: $latest_run" >&2
        echo "[ERROR] (run はあるが model_*.pt が無い。前段が起動直後に落ちた可能性)" >&2
        [[ -n "$hint" ]] && echo "[ERROR] $hint" >&2
        return 1
    fi
    echo "$ckpt"
}

# checkpoint が 1 フレーム観測 (素の ActorCritic) で学習されたものかを判定する。
# 履歴入力版は actor.mlp.0.weight を、1 フレーム版は actor.0.weight を持つ。
is_single_frame_ckpt() {
    $LAB_PY - "$1" <<'PY' >/dev/null 2>&1
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = sd.get("model_state_dict", sd)
sys.exit(0 if "actor.0.weight" in sd else 1)
PY
}

# 1 フレーム観測の checkpoint なら --warm_start_from_single_frame を自動で付ける。
# 付け忘れると actor が 1 本も引き継がれず、乱数のまま歩行からやり直しになる
# (train.py 側でも止まる)。呼び出し側で配列 EXTRA_ARGS を用意しておくこと。
maybe_warm_start() {
    local ckpt="$1"
    if [[ "${SCRIPT_ARGS[*]:-}" == *--warm_start_from_single_frame* ]]; then
        return 0
    fi
    if is_single_frame_ckpt "$ckpt"; then
        echo "[INFO] 引き継ぎ元が 1 フレーム観測の checkpoint なので"
        echo "[INFO] --warm_start_from_single_frame を自動で付けます。"
        EXTRA_ARGS+=(--warm_start_from_single_frame)
    fi
}

SCRIPT_ARGS=("$@")

if should_run 2; then
    if [[ -z "${WALK_CKPT:-}" ]]; then
        # 履歴入力版の walk phase を優先。無ければ both_feet 系の 1 フレーム版へ。
        WALK_CKPT="$(find_latest_ckpt "$WALK_LOG_ROOT" 2>/dev/null || echo "")"
        if [[ -z "$WALK_CKPT" ]]; then
            WALK_CKPT="$(find_latest_ckpt "$FALLBACK_WALK_LOG_ROOT" 2>/dev/null || echo "")"
            if [[ -n "$WALK_CKPT" ]]; then
                echo "[INFO] 履歴入力版の walk phase が無いので both_feet 系の"
                echo "[INFO] 1 フレーム版を使います: $WALK_CKPT"
            fi
        fi
    fi
    if [[ -z "${WALK_CKPT:-}" || ! -f "$WALK_CKPT" ]]; then
        echo "[ERROR] walk phase の checkpoint がありません。" >&2
        echo "[ERROR]   探した先: $WALK_LOG_ROOT, $FALLBACK_WALK_LOG_ROOT" >&2
        echo "[ERROR] 先に STAGE=1 ./scripts/rsl_rl/train_walk_kick_dual.sh を回すか、" >&2
        echo "[ERROR] both_feet 系の 1 フレーム checkpoint を WALK_CKPT=<path> で" >&2
        echo "[ERROR] 指定してください (旧 sole_pos 系は使えません)。" >&2
        exit 1
    fi
    EXTRA_ARGS=()
    maybe_warm_start "$WALK_CKPT"
    echo "=============================================================="
    echo " Stage 2/4: weak kick (dual)  (task=$KICK_TASK, iters=$KICK_ITER)"
    echo " pretrained: $WALK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$KICK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$KICK_ITER" \
        --load_pretrained "$WALK_CKPT" \
        "${EXTRA_ARGS[@]}" \
        "$@"
fi

if should_run 3; then
    KICK_CKPT="${KICK_CKPT:-$(find_latest_ckpt "$KICK_LOG_ROOT" \
        "先に Stage 2 を回すこと: STAGE=2 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh
[ERROR] (checkpoint を明示するなら KICK_CKPT=<path> を渡す)")}"
    EXTRA_ARGS=()
    maybe_warm_start "$KICK_CKPT"
    echo "=============================================================="
    echo " Stage 3/4: weak kick 360 (dual, クリーン)  (task=$KICK360_TASK, iters=$KICK360_ITER)"
    echo " pretrained: $KICK_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$KICK360_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$KICK360_ITER" \
        --load_pretrained "$KICK_CKPT" \
        "${EXTRA_ARGS[@]}" \
        "$@"
fi

if should_run 4; then
    KICK360_CKPT="${KICK360_CKPT:-$(find_latest_ckpt "$KICK360_LOG_ROOT" \
        "先に Stage 3 を回すこと: STAGE=3 ./scripts/rsl_rl/train_walk_kick_weak_dual.sh
[ERROR] (checkpoint を明示するなら KICK360_CKPT=<path> を渡す)")}"
    EXTRA_ARGS=()
    maybe_warm_start "$KICK360_CKPT"
    echo "=============================================================="
    echo " Stage 4/4: weak 360 + 観測 DR (dual, 最終)  (task=$DR_TASK, iters=$DR_ITER)"
    echo " pretrained: $KICK360_CKPT"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$DR_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$DR_ITER" \
        --load_pretrained "$KICK360_CKPT" \
        "${EXTRA_ARGS[@]}" \
        "$@"
fi

echo "[INFO] done."
