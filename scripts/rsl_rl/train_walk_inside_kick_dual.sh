#!/usr/bin/env bash
# walk_inside_kick (右足インサイドキック) の stage 2 / stage 3 を通しで学習する。
#
#   Stage 1: (このスクリプトでは回さない) Isaac-Velocity-Flat-K1-Walk-Inside-Kick-v0
#            = ./scripts/rsl_rl/train_walk_inside_kick.sh。インサイドの「型」を
#            ここで発見・収束させる。基準 run: 2026-08-22_11-56-42 (3600 iteration、
#            alpha 1.0 / kick_rate 0.998 / plant_lon -0.107 / foot_kick_dot -0.030 /
#            vel_ratio 0.887)。
#
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Dual-v0
#            actor の入力を 100 フレームの観測履歴にする (dual encoder:
#            「直近 5 フレームそのまま」+「100 フレームの 1D-CNN 潜在」)。
#            **変更点はそれだけ。** 報酬・地形・ボール DR は stage 1 と同一なので、
#            指標が動いたら原因は履歴以外にあり得ない。
#
#   Stage 3: Isaac-Velocity-Rough-K1-Walk-Inside-Kick-Dual-v0
#            stage 2 の上に凹凸地形 (±1-4 cm のランダムノイズ、段差・坂なし) と、
#            ボール物性 DR の帯の拡大 (静摩擦 0.3-1.0 / 動摩擦 0.2-0.8 /
#            反発 0.0-0.7、walk_loop_shoot 相当) を載せる。
#
# 設計と各段の根拠は
# source/isaaclab_k1_locomotion/.../walk_inside_kick/walk_inside_kick_env_cfg.py の
# モジュール docstring 「stage 2 / stage 3 (dual history / rough + DR)」節。
#
# カリキュラムは env cfg 側で終値に固定してある
# ---------------------------------------------
# stage 2/3 は **収束済み** checkpoint からの fine-tune。--load_pretrained は
# common_step_counter を 0 に戻すので、カリキュラムを生かしたままだと全ランプが
# 巻き戻る (キック報酬は 0 から / 拡大ゲートは限定レンジから / σ_velocity は 1.0 から /
# strong は満額で復活)。特にキック報酬のフェードインは
# 「最初の 500 iteration は蹴らない方が得」を明示的に作るので致命的。
# そこで K1WalkInsideKickDualEnvCfg が _pin_curricula_at_end() で全項の終値を
# 対象へ直接書き込み、curriculum 項そのものを None にしている。
#
# **その帰結として、flat 段のような「ITER は 3000 以上」の下限は無い。**
# flat 段が 3000 を要求していたのは middle 由来のカリキュラム (strong のフェード
# アウト / σ_velocity のアニール / overshoot 罰) と拡大ゲートが 3000 iteration で
# 終点に着くからで、こちらは最初から終点に居る。既定 3000 は「fine-tune として
# これくらい」という量であって、短く切っても報酬定義は途中で変わらない。
#
# 1 フレーム checkpoint からの橋渡し (--warm_start_from_single_frame)
# ------------------------------------------------------------------
# stage 1 の checkpoint は 1 フレーム観測 (素の ActorCritic) なので、履歴 actor
# (ActorCriticHistoryCNN) とは actor MLP 1 層目の形も重みの名前も違う
# (actor.0.* vs actor.mlp.0.*)。そのまま渡すと actor が 1 本も引き継がれない。
# --warm_start_from_single_frame を付けると旧 actor を履歴 actor の「最新フレームの
# 列」へ移植するので、学習開始時点の出力が stage 1 のポリシーと一致する。
# このスクリプトは checkpoint に actor.0.weight があるかを見て自動で付ける。
# critic は 1 フレームのまま (61 次元) なので無加工でそのまま載る。
# stage 2 → stage 3 は履歴 → 履歴なので付かない (付ける必要が無い)。
#
# 起動直後に必ず見ること
# ----------------------
# * ログの "Loaded N tensors" / "Skipped N tensors"。actor.* が Skipped 側に並んで
#   いたら checkpoint が繋がっていないので止めて引数を直す。
# * TensorBoard の 1 iteration 目。出発点が収束済みなので、下の 4 つはほぼ基準値の
#   はずで、そうなっていなければ引き継ぎに失敗している:
#     Metrics/kick_direction/sole_height_at_kick ≈ 0.087 が基準。2026-08-24 の
#                                                     変更 (接触点を 0.05 側へ) で
#                                                     下がるのが正常。段を越えて
#                                                     0.087 へ戻るなら低い当て方を
#                                                     壊している = 実機の巻き込み事故
#                                                     の直接原因が復活している
#     Metrics/kick_direction/foot_kick_dot  ≈  0.00  (1 へ上がったらトーキック回帰)
#     Metrics/kick_direction/kick_vel_ratio ≈  0.89  (水平成分。use_3d_speed=True に
#                                                     したので仰角ぶん下がって見える
#                                                     のは正常)
#     Metrics/kick_direction/plant_lon      ≈ -0.11  (**観察用**。2026-08-24 に報酬
#                                                     から外した = 採否の判断材料に
#                                                     しない)
#     Metrics/kick_direction/kick_rate      ≈  1.00  (stage 3 は序盤に落ちてよい。
#                                                     数百 iteration で戻らなければ
#                                                     地形が厳しすぎる)
#
# 使ってはいけないフラグ
# ----------------------
# * --resume: experiment_name が段ごとに違うので前段の run を検出できないうえ、
#   common_step_counter を同期してしまう。段の引き継ぎは常に --load_pretrained。
# * --reset_noise_std: 収束済みの std (0.06-0.1 級) を戻すと当たり所の精度が壊れる。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_inside_kick_dual.sh            # 既定 (STAGE=23)
#   STAGE=2 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh    # 平坦の履歴段だけ
#   STAGE=3 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh    # 凹凸段だけ
#   ITER=6000 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh  # 両段を長く回す
#   DUAL_ITER=6000 ROUGH_ITER=3000 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh
#   INSIDE_CKPT=logs/rsl_rl/k1_walk_inside_kick/2026-08-22_11-56-42/model_3600.pt \
#       ./scripts/rsl_rl/train_walk_inside_kick_dual.sh        # 引き継ぎ元を明示
#   DUAL_CKPT=logs/.../model_2999.pt STAGE=3 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh
#   GPUS=4 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh     # 4 GPU で DDP
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh
#   GPUS=2 MASTER_PORT=29600 ./scripts/rsl_rl/train_walk_inside_kick_dual.sh  # 2本同時
#
# NUM_ENVS は GPU 1 枚あたりの数 (合計は NUM_ENVS × GPUS)。IsaacLab は rank ごとに
# num_envs 個の env を作るので、合計を据え置きたいなら NUM_ENVS を GPUS で割ること。
# 使う GPU は CUDA_VISIBLE_DEVICES で絞り、同じマシンで 2 本回すときは MASTER_PORT を
# ずらす。詳細は _orbit_common.sh のマルチ GPU のコメント。
#
# NOTE: 履歴長 H = 100 を変えると checkpoint 同士が繋がらない (環境側の
#       walk_kick_dual_env_cfg._OBS_HISTORY_LENGTH が全 dual 系の一元管理点)。
#       ONNX の入力形状も (1, H, 55) になるので、実機側のリングバッファ長も
#       合わせること。

# _orbit_common.sh の既定は STAGE=all。このスクリプトは stage 1 (型の発見) を
# 持たないので「2 と 3」を既定にする。**source より前に置くこと** — あちらが
# STAGE=${STAGE:-all} を実行してしまうと、後から ${STAGE:-23} を書いても効かない。
# (should_run は "all" でも 2/3 の両方で真になるので実害は無いが、STAGE の値が
#  そのままログや意図の表明になるので明示しておく)
STAGE=${STAGE:-23}

source "$(dirname "${BASH_SOURCE[0]}")/_orbit_common.sh"

# 履歴観測 (N, 100, 55) は 1 フレーム観測の 100 倍のメモリを食う。断片化で OOM
# しないようアロケータを可変セグメントにする (既に設定済みならそちらを尊重)。
# RunnerCfg 側でも num_mini_batches を 4 → 8 に割ってある (総バッチ量は不変)。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ITER=${ITER:-3000}
DUAL_ITER=${DUAL_ITER:-$ITER}
ROUGH_ITER=${ROUGH_ITER:-$ITER}

DUAL_TASK=${DUAL_TASK:-"Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Dual-v0"}
ROUGH_TASK=${ROUGH_TASK:-"Isaac-Velocity-Rough-K1-Walk-Inside-Kick-Dual-v0"}

# 引き継ぎ元の log ルート。experiment_name は agents/rsl_rl_ppo_cfg.py が決めている。
INSIDE_LOG_ROOT=${INSIDE_LOG_ROOT:-"logs/rsl_rl/k1_walk_inside_kick"}
DUAL_LOG_ROOT=${DUAL_LOG_ROOT:-"logs/rsl_rl/k1_walk_inside_kick_dual"}

# 利用者が自分で --warm_start_from_single_frame を渡していれば二重に付けないための控え。
SCRIPT_ARGS=("$@")

# checkpoint が 1 フレーム観測 (素の ActorCritic) で学習されたものかを判定する。
# 履歴入力版は actor.mlp.0.weight を、1 フレーム版は actor.0.weight を持つ。
# 移植元: train_walk_kick_dual.sh の同名関数。
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
# train.py 側でも止まるようにしてあるが、1 フレーム checkpoint (= stage 1) から
# この段を始めるのは正当な使い方なので、ここで正しい引数を組み立てる。
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

if should_run 2; then
    # 既定は k1_walk_inside_kick の最新 run。基準 run を固定したいなら
    # INSIDE_CKPT=logs/rsl_rl/k1_walk_inside_kick/2026-08-22_11-56-42/model_3600.pt
    INSIDE_CKPT="${INSIDE_CKPT:-$(find_latest_ckpt "$INSIDE_LOG_ROOT")}"
    EXTRA_ARGS=()
    maybe_warm_start "$INSIDE_CKPT"
    run_stage "Stage 2/3: inside kick + 観測履歴 (平坦)" \
        "$DUAL_TASK" "$DUAL_ITER" "$INSIDE_CKPT" "${EXTRA_ARGS[@]}" "$@"
fi

if should_run 3; then
    # 履歴 → 履歴なので warm start は不要。念のため判定は通す (1 フレームの
    # checkpoint を DUAL_CKPT で明示的に渡した場合に備えて)。
    DUAL_CKPT="${DUAL_CKPT:-$(find_latest_ckpt "$DUAL_LOG_ROOT")}"
    EXTRA_ARGS=()
    maybe_warm_start "$DUAL_CKPT"
    run_stage "Stage 3/3: inside kick + 観測履歴 + 凹凸地形 + ボール DR 拡大" \
        "$ROUGH_TASK" "$ROUGH_ITER" "$DUAL_CKPT" "${EXTRA_ARGS[@]}" "$@"
fi

echo "[INFO] done."
