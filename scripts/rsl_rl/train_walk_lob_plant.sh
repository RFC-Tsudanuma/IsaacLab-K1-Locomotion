#!/usr/bin/env bash
# walk_lob_plant (軸足の踏み込みつきロブキック) を 3 段通しで学習する。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Lob-Plant-Walk-Phase-v0   (歩行のみ / 平坦)
#            ボール無しで歩くだけ。**観測は 100 フレームの履歴**で、共用の歩行
#            checkpoint (1 フレーム観測) から --warm_start_from_single_frame で入る。
#            歩容そのものは既に収束しているので、この段の仕事は「同じ歩容を履歴 actor
#            という別のネットワークで再現し直す」ことだけ。だから WALK_ITER は 2000。
#
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Lob-Plant-v0              (ロブ本体 / 平坦)
#            **この系列の本体。** walk_lob のロブ報酬 (kick_velocity_scaled 撤去 /
#            vz_sat 5.0 / phi_sat 60° / σ_direction 0.6) を土台に 6 点を足す:
#              1. ガウス版の軸足 kick_plant_foot を項ごと撤去 (5 run 反証済み)
#              2. 線形テントの kick_plant_lon + kick_plant_yaw を追加 (inside の流儀)
#              3. kick_velocity_strong を折れ線で復活 = **発見の呼び水**
#                 [(0,0), (500,W), (1200,0)] — 立ち上げてから退場させる
#              4. kick_loft / kick_elevation を 5.0 → 10.0、kick_foot_lift を 2.0 → 6.0
#              5. 接触幾何のメトリクスを出す (plant_yaw_dot が見えるようになる)
#              6. 全カリキュラムの steps_per_iteration を 48 に統一
#
#   Stage 3: Isaac-Velocity-Rough-K1-Walk-Lob-Plant-v0             (凹凸 + ボール DR)
#            凹凸地形 (±1-4 cm) と、ボール DR の 4 点セット (足の反発 / ボール物性 /
#            初期回転 / 転がり減速)。カリキュラムは env cfg 側で終値に固定済み。
#
# 設計と各段の根拠は
# source/isaaclab_k1_locomotion/.../walk_lob_plant/walk_lob_plant_env_cfg.py の
# モジュール docstring。
#
# なぜ「キック段」を挟まないのか (walk_lob_rough との違い)
# --------------------------------------------------------
# walk_lob_rough は walk phase → lob が直行しない問題 (2026-08-18: eplen 25 ステップの
# まま 400 iteration 改善せず) に対して、間に walk_kick の報酬集合で回す **キック段**
# を挟んだ。こちらは段を足さず、stage 2 の中で kick_velocity_strong を
# 「0 → 500 で立ち上げ、500 → 1200 で退場」させることで同じ役割を持たせる
# (walk_inside_kick が実証済みの device)。
#
# 立ち上がらなかったとき (kick_rate が 500 iteration で 0 のまま) は、この設計が
# 失敗したということなので walk_lob_rough 方式 (キック段を挟む) に戻す判断になる。
#
# ITER が長いのはロブだから
# -------------------------
# stage 2 の既定は 8000。カリキュラムの終点が lon_span の第 3 段 (4000 iteration) に
# あることに加えて、**ロブは apex がなかなか飽和しない** — loop_shoot 系では
# 10000 iteration を超えても apex が上がり続けていた。途中で止めた値を「頭打ち」と
# 読まないこと。
#
# 起動直後に必ず見ること
# ----------------------
# * ログの "Loaded N tensors" / "Skipped N tensors"。actor.* が Skipped 側に並んで
#   いたら checkpoint が繋がっていないので止めて引数を直す。
# * Train/mean_episode_length と Episode_Termination/base_height。段の切り替え直後は
#   必ず崩れるが、100-200 iteration で eplen が戻らなければその段は前段から
#   ブートストラップできていない (2026-08-18 の失敗と同じ形)。
#
# TensorBoard で追うもの (stage 2)
# --------------------------------
#   Metrics/kick_direction/kick_rate         まず これ。inside は 250 iteration で
#                                            0.85 を超えた。500 iteration までに
#                                            立ち上がらなければ呼び水 (strong) の失敗。
#   Metrics/kick_direction/kick_apex_height  本命。旧 flat lob の頭打ち 0.425 を
#                                            超えて 0.9 へ向かうか。
#   Metrics/kick_direction/plant_lon         -0.42 から 0 側へ動くか。1500 iteration
#                                            までに -0.30 側へ動いていなければ、
#                                            span の折れ線 (第 2 段が 1500 から) を
#                                            後ろへずらして回し直すこと。
#   Metrics/kick_direction/plant_yaw_dot     **1 iteration 目の値を必ず記録する**
#                                            (ロブ系での実測がまだ無い。1 = 蹴り方向 /
#                                            0 = 真横。素で 0.9 級ならこの項は効かない
#                                            ので yaw_span を絞る側で考え直す)。
#   Metrics/kick_direction/kick_vel_ratio    副作用の監視。apex ∝ (v·sinφ)² なので
#   Metrics/kick_direction/foot_vz           apex は実質ボール速度が支配している。
#                                            軸足 2 項が威力を食っていないかはここでしか
#                                            見えない。落ちながら plant_lon だけ動いて
#                                            いるなら span を緩める。
#
# 使ってはいけないフラグ
# ----------------------
# * --resume: experiment_name が段ごとに違うので前段の run を検出できないうえ、
#   common_step_counter を同期してしまい、キック報酬のフェードイン (0→500) と
#   strong の折れ線が「もう終わった」と判定される。段の引き継ぎは常に --load_pretrained。
# * --reset_noise_std: 歩行 checkpoint のスイングを壊す。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_lob_plant.sh                    # 既定 (STAGE=123)
#   STAGE=2 ./scripts/rsl_rl/train_walk_lob_plant.sh            # ロブ本体だけ
#   STAGE=23 ./scripts/rsl_rl/train_walk_lob_plant.sh           # stage 1 を飛ばす
#   ITER=12000 ./scripts/rsl_rl/train_walk_lob_plant.sh         # stage 2/3 を長く
#   WALK_ITER=8000 ROUGH_ITER=5000 ./scripts/rsl_rl/train_walk_lob_plant.sh
#   WALK_CKPT="" WALK_ITER=20000 ./scripts/rsl_rl/train_walk_lob_plant.sh
#       stage 1 をゼロから回す (歩行 checkpoint を使わない)。**通常は不要** —
#       共用の歩行 checkpoint と物理も観測レイアウトも同一なので、warm start した
#       方が速いし壊れない。
#   LOB_CKPT=logs/.../model_7999.pt STAGE=3 ./scripts/rsl_rl/train_walk_lob_plant.sh
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_walk_lob_plant.sh
#   GPUS=4 ./scripts/rsl_rl/train_walk_lob_plant.sh             # 4 GPU で DDP
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/rsl_rl/train_walk_lob_plant.sh
#   GPUS=2 MASTER_PORT=29600 ./scripts/rsl_rl/train_walk_lob_plant.sh  # 2本同時
#
# NUM_ENVS は GPU 1 枚あたりの数 (合計は NUM_ENVS × GPUS)。詳細は _orbit_common.sh の
# マルチ GPU のコメント。
#
# NOTE: 履歴長 H = 100 を変えると checkpoint 同士が繋がらない (環境側の
#       walk_kick_dual_env_cfg._OBS_HISTORY_LENGTH が全 dual 系の一元管理点)。
#       ONNX の入力形状も (1, H, 55) になるので、実機側のリングバッファ長も
#       合わせること。

# _orbit_common.sh の既定は STAGE=all。3 段あることを明示するため 123 を既定にする。
# **source より前に置くこと** — あちらが STAGE=${STAGE:-all} を実行してしまうと、
# 後から ${STAGE:-123} を書いても効かない。
STAGE=${STAGE:-123}

source "$(dirname "${BASH_SOURCE[0]}")/_orbit_common.sh"

# 履歴観測 (N, 100, 55) は 1 フレーム観測の 100 倍のメモリを食う。断片化で OOM
# しないようアロケータを可変セグメントにする (既に設定済みならそちらを尊重)。
# RunnerCfg 側でも num_mini_batches を 4 → 8 に割ってある (総バッチ量は不変)。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ITER は stage 2/3 の既定。stage 1 は性格が違うので別枠 (WALK_ITER)。
ITER=${ITER:-8000}
WALK_ITER=${WALK_ITER:-2000}
LOB_ITER=${LOB_ITER:-$ITER}
ROUGH_ITER=${ROUGH_ITER:-3000}

WALK_TASK=${WALK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Lob-Plant-Walk-Phase-v0"}
LOB_TASK=${LOB_TASK:-"Isaac-Velocity-Flat-K1-Walk-Lob-Plant-v0"}
ROUGH_TASK=${ROUGH_TASK:-"Isaac-Velocity-Rough-K1-Walk-Lob-Plant-v0"}

# 引き継ぎ元の log ルート。experiment_name は agents/rsl_rl_ppo_cfg.py が決めている。
WALK_LOG_ROOT=${WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_lob_plant_walk_phase"}
LOB_LOG_ROOT=${LOB_LOG_ROOT:-"logs/rsl_rl/k1_walk_lob_plant"}

# Stage 1 の引き継ぎ元 = 共用の歩行 checkpoint。観測 55 次元・並びとも同一なので
# そのまま載る (1 フレーム → 履歴の橋渡しは --warm_start_from_single_frame)。
#
# NOTE: ここは「最新 run から自動で拾う」を **してはいけない**。共用の
#       k1_walk_kick_walk_phase には中断した run (model_4.pt / model_0.pt /
#       model_400.pt) が混ざっていて、run 名の新しい順に拾うと 400 iteration の
#       中断 run を掴む。他の通しスクリプトと同じ既知の完走 checkpoint を直に指定する。
#
# WALK_CKPT="" にすると --load_pretrained を付けずにゼロから回す。そのときは
# WALK_ITER も 20000 級へ上げること (歩行の獲得そのものからやり直すため)。
WALK_CKPT=${WALK_CKPT-"logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt"}

# 利用者が自分で --warm_start_from_single_frame を渡していれば二重に付けないための控え。
SCRIPT_ARGS=("$@")

# checkpoint が 1 フレーム観測 (素の ActorCritic) で学習されたものかを判定する。
# 履歴入力版は actor.mlp.0.weight を、1 フレーム版は actor.0.weight を持つ。
# 移植元: train_walk_inside_kick_dual.sh の同名関数。
is_single_frame_ckpt() {
    "${LAB_PY_CMD[@]}" - "$1" <<'PY' >/dev/null 2>&1
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = sd.get("model_state_dict", sd)
sys.exit(0 if "actor.0.weight" in sd else 1)
PY
}

# 1 フレーム観測の checkpoint なら --warm_start_from_single_frame を自動で付ける。
#
# 付け忘れると actor が 1 本も引き継がれず、乱数のまま歩行からやり直しになる。
# train.py 側でも止まるようにしてあるが、1 フレーム checkpoint (= 共用の歩行段) から
# stage 1 を始めるのは正当な使い方なので、ここで正しい引数を組み立てる。
# 呼び出し側で配列 EXTRA_ARGS を用意しておくこと。
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

if should_run 1; then
    EXTRA_ARGS=()
    if [[ -n "$WALK_CKPT" ]]; then
        if [[ ! -f "$WALK_CKPT" ]]; then
            echo "[ERROR] 歩行 checkpoint がありません: $WALK_CKPT" >&2
            echo "[ERROR] WALK_CKPT=<path> で明示するか、WALK_CKPT=\"\" でゼロから回してください。" >&2
            exit 1
        fi
        maybe_warm_start "$WALK_CKPT"
    else
        echo "[INFO] WALK_CKPT が空なので stage 1 をゼロから回します"
        echo "[INFO] (WALK_ITER=$WALK_ITER。歩行の獲得からやり直すなら 20000 級を推奨)"
    fi
    run_stage "Stage 1/3: 歩行のみ (平坦・観測履歴)" \
        "$WALK_TASK" "$WALK_ITER" "$WALK_CKPT" "${EXTRA_ARGS[@]}" "$@"
fi

if should_run 2; then
    # 履歴 → 履歴なので warm start は不要。念のため判定は通す (1 フレームの
    # checkpoint を WALK_STAGE_CKPT で明示的に渡した場合に備えて)。
    WALK_STAGE_CKPT="${WALK_STAGE_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT")}"
    EXTRA_ARGS=()
    maybe_warm_start "$WALK_STAGE_CKPT"
    run_stage "Stage 2/3: ロブ本体 (平坦・軸足の踏み込み + 呼び水 + 高さ重視)" \
        "$LOB_TASK" "$LOB_ITER" "$WALK_STAGE_CKPT" "${EXTRA_ARGS[@]}" "$@"
fi

if should_run 3; then
    LOB_CKPT="${LOB_CKPT:-$(find_latest_ckpt "$LOB_LOG_ROOT")}"
    EXTRA_ARGS=()
    maybe_warm_start "$LOB_CKPT"
    run_stage "Stage 3/3: ロブ + 凹凸地形 + ボール物性 DR" \
        "$ROUGH_TASK" "$ROUGH_ITER" "$LOB_CKPT" "${EXTRA_ARGS[@]}" "$@"
fi

echo "[INFO] done."
