#!/usr/bin/env bash
# walk_inside_kick の stage 1 / 2 / 3 を **1 コマンドで通しで** 学習する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Inside-Kick-v0        (1 フレーム観測)
#            = ./scripts/rsl_rl/train_walk_inside_kick.sh
#            インサイドの「型」を発見・収束させる。歩行 checkpoint から始める。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Dual-v0   (100 フレーム履歴 + fewa 束)
#   Stage 3: Isaac-Velocity-Rough-K1-Walk-Inside-Kick-Dual-v0  (凹凸 + ボール物性 DR)
#            = ./scripts/rsl_rl/train_walk_inside_kick_dual.sh の STAGE=2 / STAGE=3
#
# なぜこのラッパーが要るのか
# --------------------------
# 既存の 2 本 (train_walk_inside_kick.sh = stage 1、train_walk_inside_kick_dual.sh =
# stage 2/3) は別コマンドなので、通しで回すには stage 1 の完走を待って手で 2 本目を
# 叩く必要があった。**寝ている間に 3 段を回しきる**ためのラッパー。
#
# 段の間の checkpoint の受け渡しは、既存スクリプトの ``find_latest_ckpt`` に任せず
# **このラッパーが明示的に解決して渡す**。理由は、``find_latest_ckpt`` が
# 「experiment ディレクトリの最新 run」を取る仕様なので、同じマシンで別の run を
# 並行して回していると取り違えるため。ここでは stage 1 が実際に書いた run を
# 掴んで INSIDE_CKPT に固定する。
#
# 使い方
# ------
#   ./scripts/rsl_rl/train_walk_inside_kick_full.sh                 # 3 段通し (既定)
#   nohup ./scripts/rsl_rl/train_walk_inside_kick_full.sh &         # 寝ている間に回す
#
#   FLAT_ITER=8000 ./scripts/rsl_rl/train_walk_inside_kick_full.sh  # stage 1 だけ長く
#   STAGES=23 ./scripts/rsl_rl/train_walk_inside_kick_full.sh       # stage 1 を飛ばす
#   NUM_ENVS=8192 GPUS=2 ./scripts/rsl_rl/train_walk_inside_kick_full.sh
#
# 途中で落ちた場合は STAGES で続きから再開できる (checkpoint は各 experiment
# ディレクトリに残っているので、飛ばした段の分は自動で拾われる)。
#
# ログ
# ----
# 標準出力をそのまま流しつつ、``logs/rsl_rl/_full_run_<timestamp>.log`` にも残す。
# 朝どこで落ちたかを追えるようにするため。LOG_FILE=<path> で変更、LOG_FILE=none で無効。
#
# 既定の iteration とその根拠
# ---------------------------
# * FLAT_ITER=6000 … stage 1。既存スクリプトの既定は 5000 だが、
#   (a) 前回の run (2026-08-22_11-22-36) は 4887 iteration の時点でまだ改善中で
#       (dir_error 5.59 -> 5.27)、頭打ちになっていなかった。
#   (b) 2026-08-25 に指令帯を (3.2, 4.5) -> (1.5, 4.5) へ広げたぶん課題が難しくなった。
#   なお middle 由来のカリキュラム (strong の折れ線 / σ_velocity のアニール / 拡大
#   ゲート) が 3000 iteration で終点に着くので、**stage 1 は 3000 以上が必須**。
# * DUAL_ITER=3000 … stage 2。カリキュラムは終値固定なので下限は無い。
# * ROUGH_ITER=3000 … stage 3。**ここは 3000 を下回らないこと。** 2026-08-25 に
#   σ_direction のアニール (0.35 -> 0.20、窓 500-2500 iteration) を **この段だけに**
#   足したので、短く切ると絞りきる前に終わる
#   (:data:`_INSIDE_SIGMA_DIRECTION_ANNEAL` in walk_inside_kick_env_cfg.py)。
#
# 起動直後に必ず見ること
# ----------------------
# * ログの "Loaded N tensors" / "Skipped N tensors"。actor.* が Skipped 側に並んで
#   いたら checkpoint が繋がっていない。
# * stage 2 の 1 iteration 目。出発点が収束済みなので下の 2 つは基準値のはず:
#     Metrics/kick_direction/foot_kick_dot  ≈ 0.00  (1 へ上がったらトーキック回帰)
#     Metrics/kick_direction/sole_height_at_kick ≈ 0.05 側 (0.087 へ戻ったら
#                                                  実機の巻き込み事故が復活している)
# * **kick_vel_ratio が前回より下がって見えるのは正常**。帯に 1.5-3.2 の低指令が
#   入ったので全 env 平均の比が変わる。低指令域の成否は専用メトリクスで見ること:
#     Metrics/kick_direction/kick_rate_low / kick_vel_ratio_low / kick_low_frac
# * stage 3 は kick_rate を監視する。σ_direction を絞るのは「方向が悪い蹴りの報酬を
#   消す」操作なので、絞りすぎると蹴ること自体を避ける方向に倒れる (0.98 前後なら健全)。
# --------------------------------------------------------------------------- #

set -euo pipefail

# STAGE は _orbit_common.sh が読む変数名なので **使わない** (子スクリプトへ leak して
# 意図しない段が飛ぶ)。このラッパーは STAGES を使い、子には都度 STAGE を渡す。
STAGES=${STAGES:-123}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

FLAT_ITER=${FLAT_ITER:-6000}
DUAL_ITER=${DUAL_ITER:-3000}
ROUGH_ITER=${ROUGH_ITER:-3000}

INSIDE_LOG_ROOT=${INSIDE_LOG_ROOT:-"logs/rsl_rl/k1_walk_inside_kick"}

# --------------------------------------------------------------------------- #
# ログの二重化 (画面 + ファイル)。LOG_FILE=none で無効。
# --------------------------------------------------------------------------- #
LOG_FILE=${LOG_FILE:-"logs/rsl_rl/_full_run_$(date +%Y-%m-%d_%H-%M-%S).log"}
if [[ "$LOG_FILE" != "none" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[INFO] ログ: $LOG_FILE"
fi

_t0=$SECONDS
_elapsed() { printf '%dh%02dm' $(( (SECONDS-_t0)/3600 )) $(( ((SECONDS-_t0)%3600)/60 )); }

banner() {
    echo
    echo "##############################################################"
    echo "# $1"
    echo "# 経過 $(_elapsed)   $(date '+%F %T')"
    echo "##############################################################"
}

# 落ちた場所を朝いちで分かるようにする。
_failed_stage=""
on_err() {
    echo
    echo "[ERROR] ${_failed_stage:-不明な段} で失敗しました (経過 $(_elapsed))。" >&2
    echo "[ERROR] 続きから再開するには STAGES=<残りの段> を付けて再実行してください。" >&2
    [[ "$LOG_FILE" != "none" ]] && echo "[ERROR] ログ: $LOG_FILE" >&2
    exit 1
}
trap on_err ERR

want() { [[ "$STAGES" == *"$1"* ]]; }

# --------------------------------------------------------------------------- #
# Stage 1: インサイドの型を発見する (1 フレーム観測)
# --------------------------------------------------------------------------- #
if want 1; then
    _failed_stage="Stage 1 (inside kick 発見)"
    banner "Stage 1/3: インサイドの型を発見 (1 フレーム観測)  iters=$FLAT_ITER"
    ITER="$FLAT_ITER" "$SCRIPT_DIR/train_walk_inside_kick.sh" "$@"
else
    echo "[INFO] Stage 1 は STAGES=$STAGES によりスキップします。"
fi

# --------------------------------------------------------------------------- #
# Stage 2/3 へ渡す checkpoint を、このラッパーが明示的に解決する。
#
# stage 1 を回した直後ならその run が最新なのでこれで一致するが、**明示的に固定する
# ことに意味がある**: dual スクリプトに任せると実行時にもう一度ディレクトリを見に行く
# ので、その間に別の run が生えると取り違える。ここで掴んだものをそのまま渡す。
# --------------------------------------------------------------------------- #
resolve_latest_ckpt() {
    local root="$1" run ckpt
    while IFS= read -r run; do
        ckpt=$(find "$run" -maxdepth 1 -name 'model_*.pt' 2>/dev/null | sort -V | tail -n 1)
        if [[ -n "$ckpt" ]]; then
            echo "$ckpt"
            return 0
        fi
    done < <(find "$root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r)
    return 1
}

if want 2 || want 3; then
    _failed_stage="Stage 1 の checkpoint 解決"
    if ! INSIDE_CKPT=$(resolve_latest_ckpt "$INSIDE_LOG_ROOT"); then
        echo "[ERROR] $INSIDE_LOG_ROOT に model_*.pt を持つ run がありません。" >&2
        echo "[ERROR] Stage 1 を先に回すか、INSIDE_CKPT=<path> で明示してください。" >&2
        exit 1
    fi
    echo
    echo "[INFO] Stage 2/3 の引き継ぎ元: $INSIDE_CKPT"
fi

# --------------------------------------------------------------------------- #
# Stage 2: 観測履歴 + fewa 束 (平坦)
# --------------------------------------------------------------------------- #
if want 2; then
    _failed_stage="Stage 2 (dual / fewa 束)"
    banner "Stage 2/3: 観測履歴 + fewa 束 (平坦)  iters=$DUAL_ITER"
    STAGE=2 INSIDE_CKPT="$INSIDE_CKPT" DUAL_ITER="$DUAL_ITER" \
        "$SCRIPT_DIR/train_walk_inside_kick_dual.sh" "$@"
else
    echo "[INFO] Stage 2 は STAGES=$STAGES によりスキップします。"
fi

# --------------------------------------------------------------------------- #
# Stage 3: 凹凸 + ボール物性 DR + σ_direction アニール
#
# 引き継ぎ元 (stage 2 の checkpoint) は dual スクリプト側の find_latest_ckpt に任せる。
# stage 2 を直前にこのラッパーが回している以上、最新 run はそれで一意に決まる。
# --------------------------------------------------------------------------- #
if want 3; then
    _failed_stage="Stage 3 (rough + ボール DR + σ_direction アニール)"
    banner "Stage 3/3: 凹凸 + ボール物性 DR + σ_direction アニール  iters=$ROUGH_ITER"
    if (( ROUGH_ITER < 3000 )); then
        echo "[WARN] ROUGH_ITER=$ROUGH_ITER は 3000 未満です。σ_direction のアニールの窓は"
        echo "[WARN] 500-2500 iteration なので、終値 0.20 まで絞りきる前に終わります。"
    fi
    STAGE=3 ROUGH_ITER="$ROUGH_ITER" \
        "$SCRIPT_DIR/train_walk_inside_kick_dual.sh" "$@"
else
    echo "[INFO] Stage 3 は STAGES=$STAGES によりスキップします。"
fi

trap - ERR
banner "完了 (全 $STAGES 段)"
echo "[INFO] 総経過 $(_elapsed)"
[[ "$LOG_FILE" != "none" ]] && echo "[INFO] ログ: $LOG_FILE"
echo "[INFO] done."
