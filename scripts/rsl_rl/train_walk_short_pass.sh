#!/usr/bin/env bash
# walk_short_pass (0.125-2 m の弱い転がしパス) を **歩行から通しで** 学習する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Short-Pass-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-v0
#            限定レンジ (ボール±60°/蹴り±45°, 0.5-0.8m) で浮かせる蹴りを獲得する。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-360-v0
#            全方位 (360°/360°, 0.5-1.5m) + 回り込み。
#   Stage 4: Isaac-Velocity-Flat-K1-Walk-Short-Pass-v0
#            速度帯を (2.0,3.0) → (0.5,2.0) へ張り替える。
#
# Stage 1-3 の中身は train_walk_loop_pass_360.sh (共用タスク) と同一だが、
# **policy 観測を 50 フレームの履歴にした専用タスク**を使う。actor は
# 「直近 5 フレームそのまま + 50 フレームの 1D-CNN 潜在」を入力に取るので、
# 前段が 1 フレーム観測だと actor の重みが 1 つも引き継げない (train.py が形の
# 合わないテンソルを黙って捨てる)。共用タスクに履歴を足すと walk_kick /
# walk_pass / walk_lob / walk_mid_kick / loop_shoot まで道連れになるため、
# short_pass 系列だけ別 ID に分けてある
# (source/.../walk_short_pass/walk_short_pass_stages_env_cfg.py 参照)。
#
# 全 stage とも観測 (50, 55)・行動 12 次元・同じ並びなので、--load_pretrained で
# そのまま引き継げる。段が繋がったかは起動ログの "Skipped N tensors" で確認する
# (0 本なら全部引き継げている)。
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
# --reset_noise_std は **既定で付けない**。以前は Stage 4 に 0.3 を入れていたが、
# 継承元の収束 std は 0.06-0.095 で、0.3 はその 3-5 倍。実測ではこれだけで歩行が壊れ、
# base_height 終了が 12 iteration で 0.82 まで上がって kick_rate は 0.19 止まりだった。
# std リセットを外すと 14 iteration で kick_rate 0.99 / vel_ratio 0.85 に戻る。
# 「速度域を探索させたい」意図は帯のゲート付きカリキュラムが代替している (下記)。
# 足すとしても継承元の 1.5-2 倍 (RESET_NOISE_STD=0.12) から。
#
# ITER は 5000 未満にしないこと。Stage 4 の速度帯カリキュラムは公称 500 → 3000
# iteration で (2.0,3.0) → (0.5,2.0) を動かし、その後の収束に 2000 iteration を
# 見込んでいる。ただし帯は **kick_rate で開閉するゲート付き** なので、蹴れていない
# 間は進まない。公称より遅れることがあるため、終了時に
# Curriculum/kick_speed_range/alpha が 1.0 に達しているかを必ず確認し、
# 達していなければ SHORT_ITER を伸ばすこと (alpha が長く止まったままなら、
# その speed_max がそのスイングの実質的な上限)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_short_pass.sh                  # 4 段を通しで実行
#   STAGE=34 ./scripts/rsl_rl/train_walk_short_pass.sh         # loop_pass まで済みなら 3,4 だけ
#   STAGE=4 ./scripts/rsl_rl/train_walk_short_pass.sh          # Stage 3 が済んでいれば Stage 4 だけ
#   STAGE=4 LOOP360_CKPT=logs/rsl_rl/k1_walk_short_pass_loop_360/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_short_pass.sh              # ckpt を明示
#   ITER=10000 ./scripts/rsl_rl/train_walk_short_pass.sh       # 全 kick 段を長く回す
#   SHORT_ITER=10000 ./scripts/rsl_rl/train_walk_short_pass.sh # Stage 4 だけ延長
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_short_pass.sh   # Stage 1 だけ延長
#   RESET_NOISE_STD= ./scripts/rsl_rl/train_walk_short_pass.sh # Stage 4 の std リセット無効
#
#   # 共用タスク (1 フレーム観測) の Stage 3 checkpoint しか無い場合に、Stage 1-3 を
#   # 回し直さずに Stage 4 だけ始める。旧 actor を履歴 actor の「最新フレームの列」へ
#   # 移植するので、学習開始時点の挙動が旧ポリシーと一致する (train.py の
#   # --warm_start_from_single_frame。仕組みは remap_single_frame_actor の docstring)。
#   STAGE=4 LOOP360_CKPT=logs/rsl_rl/k1_walk_loop_pass_360/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_short_pass.sh --warm_start_from_single_frame
#
# NOTE: 4 段合計 20000 iteration。4096 env で 1 段あたり数時間かかるので、
#       途中で落ちたときは STAGE で残りだけ再開できるようにしてある。
#
# NOTE: STAGE で途中から始められるのは **前段をこのスクリプトで回してある場合だけ**。
#       各段は直前の段の log ルート (k1_walk_short_pass_*) から checkpoint を拾うので、
#       共用タスクの既存 run (k1_walk_kick_walk_phase / k1_walk_loop_pass_360 など) は
#       引き継ぎ元にならない。1 フレーム観測で学習されていて actor の形が違うため
#       (無理に渡しても actor が初期化されたまま学習が始まる)。

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

NUM_ENVS=${NUM_ENVS:-4096}
# kick 系 (Stage 2/3/4) の既定 iteration 数。
ITER=${ITER:-5000}
# 段ごとの上書き。指定が無ければ ITER (Stage 1 だけ独立)。
WALK_ITER=${WALK_ITER:-5000}
LOOP_ITER=${LOOP_ITER:-$ITER}
LOOP360_ITER=${LOOP360_ITER:-$ITER}
SHORT_ITER=${SHORT_ITER:-$ITER}
# Stage 4 のみ。空文字を渡すと --reset_noise_std を付けない (既定はこちら)。
#
# 既定を 0.3 から「付けない」へ変えた理由 (実測):
#   継承元 (loop_pass_360) の収束 std は 0.06-0.095。0.3 はその 3-5 倍で、
#   蹴りどころか歩行が保たない。warm start を正しく効かせた状態で 0.3 を入れた run は
#   base_height 終了が 12 iteration で 0.82 まで上がり、kick_rate は 0.19 を頭打ちに
#   減り続けた (mean_reward も負のまま)。同じ条件で std リセットだけ外すと
#   14 iteration で kick_rate 0.99 / vel_ratio 0.85 に戻り、継承元の挙動がそのまま残る。
#
# 「速い蹴りは探索したことのない速度域だから std を戻す」という元の意図は、帯を
# ゲート付きカリキュラム (kick_rate_gated_speed_range) で動かすことで代替している。
# 帯は蹴れている間しか進まないので、常に「今の実力の少し上」が指令され続ける。
# std 自体も PPO が学習するパラメータなので、必要なら勝手に上がる (Stage 3 では
# 0.078 → 0.088 に増えている)。
#
# それでも探索を足したいときは、継承元の std の 1.5-2 倍あたりから試すこと:
#   RESET_NOISE_STD=0.12 ./scripts/rsl_rl/train_walk_short_pass.sh
RESET_NOISE_STD=${RESET_NOISE_STD-}
STAGE=${STAGE:-all}

WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Walk-Phase-v0"
LOOP_TASK="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-v0"
LOOP360_TASK="Isaac-Velocity-Flat-K1-Walk-Short-Pass-Loop-Pass-360-v0"
SHORT_TASK="Isaac-Velocity-Flat-K1-Walk-Short-Pass-v0"

# 履歴入力版の log ルート。共用タスク (k1_walk_kick_walk_phase /
# k1_walk_loop_pass / k1_walk_loop_pass_360) とは別ディレクトリなので、
# 既存 run が誤って引き継ぎ元に選ばれることはない。
WALK_LOG_ROOT="logs/rsl_rl/k1_walk_short_pass_walk_phase"
LOOP_LOG_ROOT="logs/rsl_rl/k1_walk_short_pass_loop_pass"
LOOP360_LOG_ROOT="logs/rsl_rl/k1_walk_short_pass_loop_360"

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
# run 名は YYYY-MM-DD_HH-MM-SS (辞書順=時刻順)、model_*.pt は sort -V で数値順。
# 直前の stage をこの実行で回した場合、その run が最新になるので自動で繋がる。
#
# 第 2 引数は「見つからなかったときに何をすればいいか」のヒント。STAGE=<n> で
# 途中の段だけを回そうとして前段が無いのが唯一の失敗パターンなので、素の
# 「run が見つかりません」だけ出して終わると原因が読めない。
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

# 引き継ぎ元が short_pass 系列 (履歴入力) の run かどうかを見て警告する。
#
# 共用タスクの run (k1_walk_kick_walk_phase / k1_walk_loop_pass*) は 1 フレーム
# 観測で学習されており、actor の重みが 1 つも引き継がれない (critic と正規化統計と
# std だけが載る)。--load_pretrained は形の合わないテンソルを黙って捨てるので、
# 明示指定でうっかり混ぜたときに気づけるようにしておく。
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

warn_if_foreign_ckpt() {
    local ckpt="$1" expected_root="$2"
    if [[ "$ckpt" != "$expected_root"/* ]]; then
        echo "[WARN] 引き継ぎ元が $expected_root の run ではありません:" >&2
        echo "[WARN]   $ckpt" >&2
        echo "[WARN] 共用タスク (1 フレーム観測) の checkpoint なら actor は引き継がれません。" >&2
        echo "[WARN] 起動ログの 'Skipped N tensors' を必ず確認すること (0 本なら問題なし)。" >&2
    fi
}

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
        "先に Stage 1 を回すこと: STAGE=1 ./scripts/rsl_rl/train_walk_short_pass.sh
[ERROR] (checkpoint を明示するなら WALK_CKPT=<path> を渡す)")}"
    warn_if_foreign_ckpt "$WALK_CKPT" "$WALK_LOG_ROOT"
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
fi

if should_run 3; then
    LOOP_CKPT="${LOOP_CKPT:-$(find_latest_ckpt "$LOOP_LOG_ROOT" \
        "先に Stage 2 を回すこと: STAGE=2 ./scripts/rsl_rl/train_walk_short_pass.sh
[ERROR] (checkpoint を明示するなら LOOP_CKPT=<path> を渡す)")}"
    warn_if_foreign_ckpt "$LOOP_CKPT" "$LOOP_LOG_ROOT"
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
fi

if should_run 4; then
    # CKPT は旧インターフェース (Stage 4 単体スクリプトだった頃の名前) の別名として残す。
    #
    # 引き継ぎ元は **履歴入力版の Stage 3** ($LOOP360_LOG_ROOT)。共用の
    # logs/rsl_rl/k1_walk_loop_pass_360 は 1 フレーム観測で学習されているので、
    # ここに渡しても actor が引き継がれない (それが欲しいなら明示指定 + 上の警告を了承)。
    LOOP360_CKPT="${LOOP360_CKPT:-${CKPT:-$(find_latest_ckpt "$LOOP360_LOG_ROOT" \
        "先に Stage 3 を回すこと: STAGE=3 ./scripts/rsl_rl/train_walk_short_pass.sh
[ERROR] 共用の logs/rsl_rl/k1_walk_loop_pass_360 は使えない: 1 フレーム観測で学習された
[ERROR] checkpoint なので actor の重みが 1 つも引き継がれない (履歴入力版が必要)。
[ERROR] (checkpoint を明示するなら LOOP360_CKPT=<path> を渡す)")}}"
    warn_if_foreign_ckpt "$LOOP360_CKPT" "$LOOP360_LOG_ROOT"

    EXTRA_ARGS=()
    if [[ -n "$RESET_NOISE_STD" ]]; then
        EXTRA_ARGS+=(--reset_noise_std "$RESET_NOISE_STD")
    fi

    # 1 フレーム観測の checkpoint なら --warm_start_from_single_frame を自動で付ける。
    #
    # 付け忘れると actor が 1 本も引き継がれず、乱数のまま歩行からやり直しになる。
    # train.py 側でも止まるようにしてあるが (--allow_untransferred_actor)、
    # 共用タスクの checkpoint から Stage 4 を始めるのは正当な使い方なので、
    # ここで自動的に正しい引数を組み立てる。
    if [[ "$*" != *--warm_start_from_single_frame* ]] && is_single_frame_ckpt "$LOOP360_CKPT"; then
        echo "[INFO] 引き継ぎ元が 1 フレーム観測の checkpoint なので"
        echo "[INFO] --warm_start_from_single_frame を自動で付けます。"
        EXTRA_ARGS+=(--warm_start_from_single_frame)
    fi

    echo "=============================================================="
    echo " Stage 4/4: short_pass  (task=$SHORT_TASK, iters=$SHORT_ITER)"
    echo " pretrained: $LOOP360_CKPT"
    echo " reset_noise_std: ${RESET_NOISE_STD:-"(off)"}"
    echo "=============================================================="
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$SHORT_TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$SHORT_ITER" \
        --load_pretrained "$LOOP360_CKPT" \
        "${EXTRA_ARGS[@]}" \
        "$@"
fi

echo "[INFO] done."
