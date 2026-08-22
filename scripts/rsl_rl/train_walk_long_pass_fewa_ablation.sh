#!/usr/bin/env bash
# walk_long_pass_fewa の Stage 4 ablation をまとめて回す。
#
# どの変種も Stage 4 (Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-v0) を継承して
# **1 箇所だけ** 変えたもので、観測 (100, 55)・行動 12 次元は基底と同一。よって
# **全変種が同じ checkpoint から始められる** ($LP_CKPT を全員に配る)。
#
#   band6      帯の終点 (3.2,5.0) → (3.2,6.0) / σ_v 0.9 → 1.4   … もっと強く蹴る
#   calm       lin_vel_z_l2 -0.8→-2.0, ang_vel_xy_l2 -0.32→-1.0,
#              feet_air_time +0.2→0.0                            … 跳ねを抑える
#   band6calm  band6 + calm                                      … 両立するか
#   grounded   軸足の接地力を報酬に足す (実装があるときだけ)      … 跳ねを抑える
#
# 変更点と仮説の詳細は
#   source/.../walk_long_pass_fewa/walk_long_pass_fewa_ablation_env_cfg.py
# のモジュール docstring。
#
# 出発 checkpoint (LP_CKPT)
# =========================
# **本番の意図は「実機で動いた fewa の Stage 4 checkpoint」から始めること。**
# 別マシンから持ってきたファイルを
#   logs/rsl_rl/k1_walk_long_pass_fewa/<run>/model_<N>.pt
# に置いて、そのパスを LP_CKPT で指す:
#
#   LP_CKPT=logs/rsl_rl/k1_walk_long_pass_fewa/2026-08-22_10-00-00/model_5000.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh
#
# LP_CKPT 未指定のときは logs/rsl_rl/k1_walk_long_pass_fewa_loop_360 (履歴入力版
# Stage 3) の最新 run を使う。これは「このマシンで 4 段を回した場合」の既定であって、
# 実機 checkpoint がある場合はそちらの方が遥かに良い出発点になる (Stage 4 の
# 帯カリキュラムを一度通っているので、帯が最初から目標付近にいる)。
#
# 使い方
# ======
#   # 逐次 (既定)。GPU 1 枚で順に回す。
#   LP_CKPT=... ./scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh
#
#   # 4 枚の GPU に 1 変種ずつ割り当てて同時に回す (一晩でまとめて見るならこちら)
#   PARALLEL=1 GPUS_LIST="0,1,2,3" LP_CKPT=... \
#       ./scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh
#
#   # 一部だけ
#   VARIANTS="band6 calm" LP_CKPT=... ./scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh
#
#   # iteration を伸ばす (既定 3000)
#   ITER=5000 LP_CKPT=... ./scripts/rsl_rl/train_walk_long_pass_fewa_ablation.sh
#
# ログは logs/ablation_<tag>.log。学習ログ (tensorboard) は
# logs/rsl_rl/k1_walk_long_pass_fewa_<tag>/<run>/ に出る。
#
# NOTE: PARALLEL=1 は変種を GPUS_LIST 上でラウンドロビンに割り当てる。GPU 枚数より
#       変種が多いと 1 枚に 2 本乗るので、NUM_ENVS を落とすこと (4096 × 2 は載らない)。
# NOTE: 各変種は 1 プロセス 1 GPU。DDP (torchrun) は使わない。変種間の比較が目的で、
#       1 本あたりを速くするより本数を同時に回す方が効くため。

# このスクリプトは stage を持たないが、_orbit_common.sh が STAGE を参照するので入れておく。
STAGE=${STAGE:-all}

# shellcheck source=scripts/rsl_rl/_orbit_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_orbit_common.sh"

ITER=${ITER:-3000}
PARALLEL=${PARALLEL:-0}
GPUS_LIST=${GPUS_LIST:-0}
LOOP360_LOG_ROOT="logs/rsl_rl/k1_walk_long_pass_fewa_loop_360"

# 既定の変種一覧。grounded は実装がある場合だけ回る (下の task_for で弾く)。
# 既定は「帯 6.0」を全部に入れた 4 本。band6 単独も候補 (跳ね対策が蹴りを
# 弱める可能性があるので、強さだけを取った版を必ず残す)。calm / grounded 単独は
# 切り分け用で既定には入れない。
VARIANTS=${VARIANTS:-"band6 band6calm band6grounded band6calmgrounded"}

# tag -> gym タスク ID。ここに無い tag はエラーで止める (綴り間違いを黙って
# 素通りさせない)。grounded は __init__.py に登録が無ければスキップする。
task_for() {
    case "$1" in
        band6)     echo "Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6-v0" ;;
        calm)      echo "Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Calm-v0" ;;
        band6calm) echo "Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6Calm-v0" ;;
        grounded)  echo "Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Grounded-v0" ;;
        band6grounded)     echo "Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6Grounded-v0" ;;
        band6calmgrounded) echo "Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Band6CalmGrounded-v0" ;;
        *)         return 1 ;;
    esac
}

PKG_INIT="source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based/walk_long_pass_fewa/__init__.py"

is_registered() {
    grep -q "id=\"$1\"" "$PKG_INIT"
}

# -- 出発 checkpoint
LP_CKPT="${LP_CKPT:-$(find_latest_ckpt "$LOOP360_LOG_ROOT")}"
if [[ ! -f "$LP_CKPT" ]]; then
    echo "[ERROR] LP_CKPT が見つかりません: $LP_CKPT" >&2
    exit 1
fi
echo "[INFO] 出発 checkpoint: $LP_CKPT"
if [[ "$LP_CKPT" != logs/rsl_rl/k1_walk_long_pass_fewa/* ]]; then
    echo "[WARN] 実機で動いた fewa の Stage 4 checkpoint は" >&2
    echo "[WARN]   logs/rsl_rl/k1_walk_long_pass_fewa/<run>/model_<N>.pt" >&2
    echo "[WARN] に置いて LP_CKPT で指すのが本来の使い方です。" >&2
    echo "[WARN] 起動ログの 'Skipped N tensors' が 0 本かを必ず確認すること。" >&2
fi

# -- 実際に回す変種を確定する
RUN_TAGS=()
for tag in $VARIANTS; do
    if ! task=$(task_for "$tag"); then
        echo "[ERROR] 知らない変種です: $tag" >&2
        echo "[ERROR] 使えるのは band6 / band6calm / band6grounded / band6calmgrounded (切り分け用: calm / grounded)" >&2
        exit 1
    fi
    if ! is_registered "$task"; then
        echo "[WARN] $tag ($task) は未登録なのでスキップします。" >&2
        continue
    fi
    RUN_TAGS+=("$tag")
done
if [[ ${#RUN_TAGS[@]} -eq 0 ]]; then
    echo "[ERROR] 回せる変種がありません。" >&2
    exit 1
fi
echo "[INFO] 変種: ${RUN_TAGS[*]}  (iters=$ITER, num_envs=$NUM_ENVS, parallel=$PARALLEL)"

mkdir -p logs

# train.py へそのまま渡す追加引数。
PASSTHRU=("$@")

# launch <tag> <gpu>
# 逐次なら前景、PARALLEL=1 なら nohup でバックグラウンドに投げる。
launch() {
    local tag="$1" gpu="$2" task log
    task=$(task_for "$tag")
    log="logs/ablation_${tag}.log"

    local -a cmd=(env "CUDA_VISIBLE_DEVICES=$gpu" "${LAB_PY_CMD[@]}"
        scripts/rsl_rl/train.py
        --task "$task" --headless
        --num_envs "$NUM_ENVS" --max_iterations "$ITER"
        --load_pretrained "$LP_CKPT")
    cmd+=(${PASSTHRU[@]+"${PASSTHRU[@]}"})

    echo "=============================================================="
    echo " ablation: $tag"
    echo " task=$task  gpu=$gpu  iters=$ITER  num_envs=$NUM_ENVS"
    echo " log: $log"
    echo "=============================================================="

    if [[ "$PARALLEL" -eq 1 ]]; then
        nohup "${cmd[@]}" > "$log" 2>&1 &
        echo "[INFO] $tag を pid $! で起動しました。"
    else
        # 逐次でも log は残す (tee で画面にも出す)。
        "${cmd[@]}" 2>&1 | tee "$log"
    fi
}

IFS=',' read -r -a GPU_ARR <<< "$GPUS_LIST"
if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
    GPU_ARR=(0)
fi

i=0
for tag in "${RUN_TAGS[@]}"; do
    if [[ "$PARALLEL" -eq 1 ]]; then
        gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    else
        gpu="${GPU_ARR[0]}"
    fi
    launch "$tag" "$gpu"
    i=$((i + 1))
done

if [[ "$PARALLEL" -eq 1 ]]; then
    echo "[INFO] 全変種を起動しました。終了を待ちます (tail -f logs/ablation_*.log で進捗)。"
    wait
fi

echo "[INFO] done."
echo "[INFO] 比較: logs/rsl_rl/k1_walk_long_pass_fewa_<tag>/ を tensorboard で開く。"
echo "[INFO]   強さ  : Curriculum/kick_speed_range/{speed_max,alpha}, Metrics/.../kick_vel_ratio"
echo "[INFO]   跳ね  : Episode_Reward/lin_vel_z_l2, Episode_Reward/feet_air_time"
echo "[INFO]   壊れ  : Train/mean_episode_length, Episode_Termination/base_height, kick_rate"
