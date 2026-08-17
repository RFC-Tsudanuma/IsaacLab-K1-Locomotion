#!/usr/bin/env bash
# walk_kick の dual encoder 版 (actor が 100 フレームの観測履歴を見る) を通しで学習する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-v0
#            限定レンジ (ボール±60°/蹴り±45°, 0.5-0.8m) で地面蹴りを獲得する。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-v0
#            全方位 (360°/360°, 0.5-1.5m) + 回り込み。**クリーン** (観測 DR は無し)。
#            σ_direction のアニール (0.35 → 0.15) と ball_avoidance のランプを
#            ここで完走させる。
#   Stage 4: Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-DR-v0
#            **最終 stage**。Stage 3 と中身は同じで、観測 DR
#            (IMU/エンコーダの遅延 ≤0.02 s + ボール観測の遅延 0.02-0.10 s と
#             一様ノイズの拡大 位置 ±0.07 m / 速度 ±0.5 m/s)、ランプの全凍結、
#            σ_direction の 0.15 固定だけが乗る (地形は平面のまま)。
#
# 移植元は fewa/walk_kick_dual_encoder_tune (walk_long_pass の dual encoder 化) と、
# walk_kick_both_feet の 2 変更 (観測スロット 3 = ボール 3D 位置 / 歩行位相オフセット
# {0, π})。**4 段構成と DR の配置は fewa の 47b8863 に合わせてある。**
# コマンド・ボール配置・終了条件は共用の walk_kick 系とまったく同じ。差は
#   * 全 stage: policy 観測が 100 フレームの履歴 + both_feet 版の観測グループ
#   * 全 stage: 着地 shaping 3 項 (feet_landing_impact / feet_landing_vel /
#               feet_heel_strike) を外す
#   * 蹴り段 (Stage 2-4): feet_phase の weight 2.0 → 0.8
#     (fewa 実測: 未対処だと「蹴らずに歩く」が最適解になり kick_rate が
#      0.19-0.28 で 4000 iteration 停滞した。Stage 1 では歩容をこの項で作るので下げない)
#   * Stage 3 のみ: σ_direction のアニール (方向の採点を 0.35 → 0.15 rad に締める)
#   * Stage 4 のみ: 観測 DR + ランプの全凍結 + σ_direction の 0.15 固定
#
# ボール観測 DR は **一様ノイズ + 遅延** (fewa 準拠)。ガウスの認識パイプライン
# (noisy_ball_pos_b) 方式は fewa の実機実績に合わせて不採用 (2026-08-17)。
#
# both_feet の効果は stage 2 で実測済み: kick_foot_right_frac 1.0 → 0.39、
# kick_dir_error 4.5°、kick_rate 0.998 (方向精度・成功率を落とさずに左右が割れた)。
# dual は未学習なので、別 variant を作らず直接畳み込んである。actor は「直近 5 フレームそのまま + 100 フレームの
# 1D-CNN 潜在」を入力に取るので、前段が 1 フレーム観測だと actor の重みが 1 つも
# 引き継げない (train.py が形の合わないテンソルを黙って捨てる。現在は actor が 0 本なら
# 止まる)。共用タスクに履歴を足すと walk_pass / walk_lob / walk_mid_kick / loop_shoot まで
# 道連れになるため、dual 系列だけ別 ID に分けてある
# (source/.../walk_kick_dual/walk_kick_dual_env_cfg.py 参照)。
#
# NOTE: 位置ノイズは ±0.07 m を超えないこと。fewa は一度 ±0.1 m まで上げて critic が
#       繰り返し発散した (10000 iteration 中 value loss > 50 が 230 回、最大 1.26e12、
#       kick_rate < 0.5 への崩壊が 102 回)。±0.07 では発散 0 回・崩壊 3 回。
#
# 全 stage とも観測 (100, 55)・行動 12 次元・同じ並び (critic 58 次元) なので、
# --load_pretrained でそのまま引き継げる。段が繋がったかは起動ログの "Skipped N tensors" で確認する
# (0 本なら全部引き継げている)。
#
# --resume ではなく --load_pretrained を使う理由: experiment_name が段ごとに違うので
# --resume では前段の run を検出できない。加えて --resume は common_step_counter を
# 同期してしまい、各段のキック報酬カリキュラムがランプしなくなる
# (詳細は train_walk_kick.sh の冒頭コメント)。
#
# 既存の 1 フレーム checkpoint から始める場合
# -------------------------------------------
# **1 フレーム観測**の checkpoint を渡すと、そのままでは actor が引き継がれない。
# このスクリプトは checkpoint に actor.0.weight があるかを見て、あれば
# --warm_start_from_single_frame を自動で付ける (旧 actor を履歴 actor の
# 「最新フレームの列」へ移植するので、学習開始時点の挙動が旧ポリシーと一致する)。
# つまり Stage 1 を回さずに Stage 2 から直行できる:
#
#   STAGE=23 WALK_CKPT=logs/rsl_rl/k1_walk_kick_both_feet_walk_phase/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_dual.sh
#
# **使えるのは both_feet 系の checkpoint だけ** (k1_walk_kick_both_feet_walk_phase /
# k1_walk_kick_both_feet)。dual の観測は both_feet 版 (スロット 3 = ボール 3D 位置、
# critic 58 次元) なので、**旧 sole_pos 系 (k1_walk_kick_walk_phase など) は不可**:
# policy は同じ 55 次元なので --load_pretrained は形の上では通ってしまうが、
# スロット 3 の意味が違うので入力の解釈がずれる (critic は次元も合わない)。
#
# mirror loss (左右対称データ拡張) について
# ----------------------------------------
# 全 stage の RunnerCfg で mirror loss を有効にしてある
# (policy(mirror(obs)) ≈ mirror(policy(obs)) を促す MSE を PPO 損失に加算。
#  鏡像写像は source/.../walk_kick_both_feet/symmetry.py、係数 0.5 は歩行タスクと同じ)。
#
# **mirror loss 導入後は stage 1 (walk phase) から回し直すこと。**
# 既存の both_feet / dual の checkpoint は既に片足に収束していて
# (kick_foot_right_frac が run ごとに 0.99 / 0.01 へ張り付く)、対称化の出発点として
# 不適。そこから掛けると、対称化は「獲得済みの蹴り足を壊す」方向にしか働かない。
#   STAGE=1234 ... で walk phase から通しで回す。
#
# 効果は Metrics/kick_direction/kick_foot_right_frac (0.5 付近なら両足で蹴れている) と、
# rsl_rl のログに出る symmetry loss を併せて見る。kick_dir_error_deg / kick_rate が
# 落ちるようなら係数 0.5 が強すぎるサイン
# (source/.../walk_kick_both_feet/agents/rsl_rl_ppo_cfg.py の _MIRROR_LOSS_COEFF)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_kick_dual.sh              # 4 段を通しで実行
#   STAGE=234 ./scripts/rsl_rl/train_walk_kick_dual.sh    # 歩行学習済みなら 2,3,4 だけ
#   STAGE=34 ./scripts/rsl_rl/train_walk_kick_dual.sh     # 3,4 だけ
#   STAGE=4 ./scripts/rsl_rl/train_walk_kick_dual.sh      # DR 段だけ
#   STAGE=3 KICK_CKPT=logs/rsl_rl/k1_walk_kick_dual/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_kick_dual.sh          # 既存の dual kick から 360 だけ
#   ITER=20000 ./scripts/rsl_rl/train_walk_kick_dual.sh   # kick 段 (2,3) を長く回す
#   DR_ITER=6000 ./scripts/rsl_rl/train_walk_kick_dual.sh # Stage 4 だけ変える
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_kick_dual.sh   # walk phase だけ延長
#
# NOTE: **Stage 3 の ITER は 3000 以上**にすること。σ_direction のアニール
#       (方向の採点を 0.35 → 0.15 rad に締める) が 1500 → 3000 iteration で終点に
#       着くので、それより短いと締め切らないまま終わる。既定は 5000。
#       kick_rate が 1.0 から落ちたら σ が深すぎるサイン (env cfg のコメント参照)。
# NOTE: Stage 4 (DR) の既定は 10000 (fewa は Stage 1-3 各 5000 + Stage 4 10000 で
#       通し検証し、ベストは model_6600)。DR に慣れるまで kick_rate が一度下がるので、
#       短く切ると「ノイズを避けて蹴らない」状態で終わりかねない。
# NOTE: --reset_noise_std は付けないこと。継承元の収束 std は 0.06-0.095 で、
#       0.3 のような値を入れると歩行が壊れる (fewa の実測)。
# NOTE: 履歴長 H = 100 を変えると checkpoint 同士が繋がらない。実機側の
#       リングバッファ長も (1, H, 55) に合わせること。

set -euo pipefail

# train.py は logs/ を CWD 基準で作るので、必ずリポジトリルートで実行する。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# IsaacLab の python を解決する (train_walk_kick_both_feet.sh と同じ手順)。
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

# 履歴観測 (N, 100, 55) × mirror loss は 2 GiB 級の一時テンソルを確保する。断片化で
# OOM しないようアロケータを可変セグメントにする (既に設定済みならそちらを尊重)。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NUM_ENVS=${NUM_ENVS:-4096}
# kick 系 (Stage 2/3) の iteration 数。詰めるときは 20000 まで上げる。
ITER=${ITER:-5000}
KICK_ITER=${KICK_ITER:-$ITER}
KICK360_ITER=${KICK360_ITER:-$ITER}
# walk phase (Stage 1) は歩行を獲得するだけなので 5000 で足りる (実績値)。
WALK_ITER=${WALK_ITER:-5000}
# Stage 4 (DR) だけ既定が長い (上の NOTE 参照)。
DR_ITER=${DR_ITER:-10000}
STAGE=${STAGE:-all}

WALK_TASK=${WALK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-Walk-Phase-v0"}
KICK_TASK=${KICK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-v0"}
KICK360_TASK=${KICK360_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-v0"}
DR_TASK=${DR_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Dual-360-DR-v0"}

# 履歴入力版の log ルート。共用タスク (k1_walk_kick_walk_phase / k1_walk_kick) とは
# 別ディレクトリなので、既存 run が誤って引き継ぎ元に選ばれることはない。
WALK_LOG_ROOT=${WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_dual_walk_phase"}
KICK_LOG_ROOT=${KICK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_dual"}
KICK360_LOG_ROOT=${KICK360_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_dual_360"}

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
# 第 2 引数は「見つからなかったときに何をすればいいか」のヒント。
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
#
# 付け忘れると actor が 1 本も引き継がれず、乱数のまま歩行からやり直しになる。
# train.py 側でも止まるようにしてあるが、既存の 1 フレーム checkpoint から dual 系を
# 始めるのは正当な使い方なので、ここで自動的に正しい引数を組み立てる。
# 呼び出し側で配列 EXTRA_ARGS を用意しておくこと。
# 利用者が自分で --warm_start_from_single_frame を渡していれば二重に付けない。
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

if should_run 1; then
    echo "=============================================================="
    echo " Stage 1/4: walk phase  (task=$WALK_TASK, iters=$WALK_ITER)"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$WALK_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$WALK_ITER" \
        "$@"
fi

if should_run 2; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT" \
        "先に Stage 1 を回すこと: STAGE=1 ./scripts/rsl_rl/train_walk_kick_dual.sh
[ERROR] (共用の 1 フレーム checkpoint から始めるなら WALK_CKPT=<path> を渡す。
[ERROR]  --warm_start_from_single_frame は自動で付く)")}"
    EXTRA_ARGS=()
    maybe_warm_start "$WALK_CKPT"
    echo "=============================================================="
    echo " Stage 2/4: walk_kick (dual)  (task=$KICK_TASK, iters=$KICK_ITER)"
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
        "先に Stage 2 を回すこと: STAGE=2 ./scripts/rsl_rl/train_walk_kick_dual.sh
[ERROR] (checkpoint を明示するなら KICK_CKPT=<path> を渡す)")}"
    EXTRA_ARGS=()
    maybe_warm_start "$KICK_CKPT"
    echo "=============================================================="
    echo " Stage 3/4: walk_kick_360 (dual, クリーン)  (task=$KICK360_TASK, iters=$KICK360_ITER)"
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
        "先に Stage 3 を回すこと: STAGE=3 ./scripts/rsl_rl/train_walk_kick_dual.sh
[ERROR] (checkpoint を明示するなら KICK360_CKPT=<path> を渡す)")}"
    EXTRA_ARGS=()
    maybe_warm_start "$KICK360_CKPT"
    echo "=============================================================="
    echo " Stage 4/4: walk_kick_360 + 観測 DR (dual, 最終)  (task=$DR_TASK, iters=$DR_ITER)"
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
