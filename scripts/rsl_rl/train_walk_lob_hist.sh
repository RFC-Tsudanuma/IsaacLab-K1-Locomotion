#!/usr/bin/env bash
# walk_lob の履歴入力版を 3 段通しで学習する。平坦 / 凹凸を TERRAIN で切り替える。
#
#   Stage 1: walk phase   ボール無しで歩行だけ                  (履歴 + 内界センサ遅延DR)
#   Stage 2: kick         walk_kick の報酬集合で「ボールに当てる」 ← ブートストラップ段
#   Stage 3: lob          高さ特化の報酬集合で「高く上げる」
#
# **stage 2 を飛ばして walk phase から lob へ直行しないこと。** 2026-08-18 に
# 2 段構成で回して失敗している (エピソードが 0.5 秒で base_height 終了し、
# 400 iteration 経っても回復しなかった)。ロブの報酬集合は kick_velocity_scaled を
# 撤去してあるため「まだ一度も当てられない」段階では勾配が出ない。詳細は
# walk_lob_rough_env_cfg.py のモジュール docstring を参照。
#
# **既存 checkpoint は流用できない。** 観測スロット3の意味 (左足裏 → ボール3D位置) と
# actor の形 (ActorCriticHistoryCNN) の 2 点で walk_lob / walk_kick 系と非互換。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_lob_hist.sh                   # 平坦 3 段 (まずこちら)
#   TERRAIN=rough ./scripts/rsl_rl/train_walk_lob_hist.sh     # 凹凸 3 段
#   STAGE=2 ./scripts/rsl_rl/train_walk_lob_hist.sh           # stage 2 だけ
#   STAGE=23 ./scripts/rsl_rl/train_walk_lob_hist.sh          # stage 2 と 3
#   WALK_CKPT=logs/rsl_rl/k1_walk_lob_rough_walk_phase/<run>/model_7999.pt \
#       TERRAIN=rough STAGE=23 ./scripts/rsl_rl/train_walk_lob_hist.sh
#   ITER=25000 ./scripts/rsl_rl/train_walk_lob_hist.sh        # lob を長く回す
#
# 推奨の進め方:
#   1. TERRAIN=flat で 3 段通し、kick_rate と kick_apex_height が出ることを確認
#   2. その stage 3 の checkpoint を LOB_CKPT に渡して TERRAIN=rough STAGE=3 で fine-tune
#      (凹凸 + ボールの組み合わせはこのリポジトリで未検証なので、最後に足す)
#
# 凹凸の stage 1 は 2026-08-17 に 8000 iteration 学習済みで健全なので、
# TERRAIN=rough のときは WALK_CKPT でそれを指せば stage 1 は省ける。
set -euo pipefail

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
                LAB_PY="$_cand"; break
            fi
        done
    fi
fi
[[ -n "$LAB_PY" ]] || { echo "[ERROR] IsaacLab の python が見つかりません。LAB_PYTHON で明示してください。" >&2; exit 1; }
echo "[INFO] python: $LAB_PY"

NUM_ENVS=${NUM_ENVS:-4096}
TERRAIN=${TERRAIN:-flat}
STAGE=${STAGE:-all}

# stage 1: 歩行の獲得。凹凸だと収束が遅いので平坦より長めに取ってある。
WALK_ITER=${WALK_ITER:-8000}
# stage 2: ボールに当てられるようになるまで。both_feet / dual の実績では
#          kick_rate が 0.99 に乗るのに 2000-2500 iteration。
KICK_ITER=${KICK_ITER:-5000}
# stage 3: ロブ。kick_plant_foot のアニールが iteration 4000 まで走るので、
#          最低でも 8000、詰めるなら 20000 以上。
ITER=${ITER:-15000}

case "$TERRAIN" in
    flat)
        WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Lob-Hist-Walk-Phase-v0"
        KICK_TASK="Isaac-Velocity-Flat-K1-Walk-Lob-Hist-Kick-v0"
        LOB_TASK="Isaac-Velocity-Flat-K1-Walk-Lob-Hist-v0"
        WALK_LOG_ROOT="logs/rsl_rl/k1_walk_lob_hist_walk_phase"
        KICK_LOG_ROOT="logs/rsl_rl/k1_walk_lob_hist_kick"
        ;;
    rough)
        WALK_TASK="Isaac-Velocity-Rough-K1-Walk-Lob-Walk-Phase-v0"
        KICK_TASK="Isaac-Velocity-Rough-K1-Walk-Lob-Kick-v0"
        LOB_TASK="Isaac-Velocity-Rough-K1-Walk-Lob-v0"
        WALK_LOG_ROOT="logs/rsl_rl/k1_walk_lob_rough_walk_phase"
        KICK_LOG_ROOT="logs/rsl_rl/k1_walk_lob_rough_kick"
        ;;
    *)
        echo "[ERROR] TERRAIN は flat か rough (指定: $TERRAIN)" >&2; exit 1 ;;
esac
echo "[INFO] terrain: $TERRAIN"

should_run() { [[ "$STAGE" == "all" || "$STAGE" == *"$1"* ]]; }

# 指定 experiment ディレクトリの最新 run から最終 checkpoint を拾う。
find_latest_ckpt() {
    local latest_run ckpt
    latest_run=$(find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
    [[ -n "$latest_run" ]] || { echo "[ERROR] run が見つかりません: $1" >&2; return 1; }
    ckpt=$(find "$latest_run" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)
    [[ -n "$ckpt" ]] || { echo "[ERROR] checkpoint が見つかりません: $latest_run" >&2; return 1; }
    echo "$ckpt"
}

run_stage() {  # $1=見出し $2=task $3=iters $4=pretrained(空なら無し)
    echo "=============================================================="
    echo " $1  (task=$2, iters=$3)"
    [[ -n "$4" ]] && echo " pretrained: $4"
    echo "=============================================================="
    local extra=()
    [[ -n "$4" ]] && extra=(--load_pretrained "$4")
    $LAB_PY scripts/rsl_rl/train.py \
        --task "$2" --headless --num_envs "$NUM_ENVS" --max_iterations "$3" \
        "${extra[@]}" "${TRAIN_ARGS[@]}"
}

TRAIN_ARGS=("$@")

if should_run 1; then
    run_stage "Stage 1/3: walk phase ($TERRAIN)" "$WALK_TASK" "$WALK_ITER" ""
fi

if should_run 2; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT")}"
    run_stage "Stage 2/3: kick ($TERRAIN)" "$KICK_TASK" "$KICK_ITER" "$WALK_CKPT"
fi

if should_run 3; then
    LOB_CKPT="${LOB_CKPT:-$(find_latest_ckpt "$KICK_LOG_ROOT")}"
    run_stage "Stage 3/3: lob ($TERRAIN)" "$LOB_TASK" "$ITER" "$LOB_CKPT"
fi

echo "[INFO] done."
