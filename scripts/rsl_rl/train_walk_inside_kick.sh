#!/usr/bin/env bash
# walk_inside_kick (右足インサイドキック) を 1 段で学習する。
#
#   Stage 1: (学習しない) 共用の歩行 checkpoint k1_walk_kick_walk_phase を使う。
#            観測 55 次元・並びとも inside_kick と同じなのでそのまま引き継げる。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Inside-Kick-v0
#            限定レンジ (ボール±60°/蹴り±45°) から始め、キック成立率のゲートで
#            全方位 (ボール±180°/蹴り±180°/距離 1.5m) まで拡大する。
#            ボール観測のノイズ+遅延は最初から入っている。
#
# なぜ 1 段なのか
# ---------------
# weak / middle 系は「限定レンジ → 全方位 → 観測ノイズ」を段で分けているが、
# こちらは分けない。
#   * 範囲の拡大は壁時計ではなく kick_rate のゲート (kick_rate_gated_expansion) が
#     進めるので、崩れたら自分で止まり、崩れ続ければ蹴れていた範囲まで戻る。
#     段分けの本来の目的 (難しくしすぎたら前段からやり直す) が要らない。
#   * 観測ノイズは「進む軸」ではない。ノイズが乗っていても latch は発火してキック
#     報酬は払われ続けるので、収支が逆転して「蹴らずに歩く」に落ちる心配が無い。
# 詳細は walk_inside_kick_env_cfg.py のモジュール docstring。
#
# ITER は 3000 以上にすること
# ------------------------------
# 土台にしている middle のカリキュラム (kick_velocity_strong のフェードアウト、
# σ_velocity の 1.0→0.5、overshoot 罰のフェードイン) が 3000 iteration でようやく
# 終点に着く。拡大ゲートの公称終点も 3000 なので、既定は 5000 にしてある。
#
# --resume は使わないこと
# -----------------------
# common_step_counter が同期され、キック報酬のフェードイン (0→500) と拡大ゲートの
# start_step (500) が「もう終わった」と判定されてランプしなくなる。
# --reset_noise_std も付けないこと (0.3 は収束 std の 3-5 倍で歩行が壊れる実測あり)。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_inside_kick.sh                 # 既定 (ノイズ込み本命)
#   ITER=20000 ./scripts/rsl_rl/train_walk_inside_kick.sh      # 仕上げ
#   CLEAN=1 ./scripts/rsl_rl/train_walk_inside_kick.sh         # フォールバック
#       ボール観測ノイズ無しの Clean タスクへ切り替える。**通常は使わない** —
#       インサイドの発見期が立ち上がらなかったときに、原因が観測ノイズなのか
#       報酬設計なのかを切り分けるためだけ。
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_walk_inside_kick.sh
#   STAGE1_CKPT=logs/.../model_4999.pt ./scripts/rsl_rl/train_walk_inside_kick.sh
#   GPUS=4 ./scripts/rsl_rl/train_walk_inside_kick.sh          # 4 GPU で DDP
#   CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/rsl_rl/train_walk_inside_kick.sh
#   GPUS=2 MASTER_PORT=29600 ./scripts/rsl_rl/train_walk_inside_kick.sh  # 2本同時
#
# NUM_ENVS は GPU 1 枚あたりの数 (合計は NUM_ENVS × GPUS)。詳細は
# _orbit_common.sh のマルチ GPU のコメント参照。
#
# 起動直後に必ず見ること: ログの "Loaded N tensors" / "Skipped N tensors"。
# actor.* が Skipped 側に並んでいたら checkpoint が繋がっていないので止めて引数を直す。

source "$(dirname "${BASH_SOURCE[0]}")/_orbit_common.sh"

ITER=${ITER:-5000}
CLEAN=${CLEAN:-0}

# Stage 1 の歩行 checkpoint。inside_kick 専用の walk phase は作っていない — 観測が
# 同一なので共用タスクのものをそのまま使う。
#
# NOTE: ここは「最新 run から自動で拾う」を **してはいけない**。共用の
#       k1_walk_kick_walk_phase には中断した run (model_4.pt / model_0.pt /
#       model_400.pt) が混ざっていて、run 名の新しい順に拾うと 400 iteration の
#       中断 run を掴む。他の通しスクリプトと同じ既知の完走 checkpoint を直に指定する。
STAGE1_CKPT=${STAGE1_CKPT:-"logs/rsl_rl/k1_walk_kick_walk_phase/2026-08-03_11-22-52/model_4999.pt"}

if [[ "$CLEAN" != "0" ]]; then
    TASK=${TASK:-"Isaac-Velocity-Flat-K1-Walk-Inside-Kick-Clean-v0"}
    TITLE="Inside Kick (Clean: ボール観測ノイズ無し / 切り分け用)"
else
    TASK=${TASK:-"Isaac-Velocity-Flat-K1-Walk-Inside-Kick-v0"}
    TITLE="Inside Kick (本命: ボール観測ノイズ+遅延あり)"
fi

if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "[ERROR] 歩行 checkpoint がありません: $STAGE1_CKPT" >&2
    echo "[ERROR] STAGE1_CKPT=<path> で明示してください。" >&2
    exit 1
fi

run_stage "$TITLE" "$TASK" "$ITER" "$STAGE1_CKPT" "$@"

echo "[INFO] done."
