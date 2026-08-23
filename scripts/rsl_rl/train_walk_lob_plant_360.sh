#!/usr/bin/env bash
# walk_lob_plant の 360° 系統 (stage 2b / 3b) を通しで学習する。
#
#   Stage 2b: Isaac-Velocity-Flat-K1-Walk-Lob-Plant-360-v0    (平坦 + 全方位)
#             stage 2 の収束済み checkpoint から入り、限定レンジ
#             (heading ±45° / half_angle 60° / dist 0.5-0.8) を **apex 込みゲート**で
#             全方位 (heading ±180° / half_angle 180° / dist 0.5-1.5) へ広げる。
#             エピソード長は 10 → 15 秒 (回り込みの移動が 2.5-3 m になるため)。
#
#   Stage 3b: Isaac-Velocity-Rough-K1-Walk-Lob-Plant-360-v0   (凹凸 + DR + ノイズ)
#             凹凸地形 (±1-4 cm)、ボール DR の 4 点セット、そして **fewa (Stage 4)
#             方式の観測ノイズ**を載せる。IMU / エンコーダの遅延はこの系列に
#             1 つも入っていなかったので、ここが実質的な純増。
#
# stage 1 (歩行のみ) と stage 2 (平坦・限定レンジ) は既存の
# ./scripts/rsl_rl/train_walk_lob_plant.sh と共通なので、このスクリプトには無い。
# 先にあちらで STAGE=12 まで通してから、こちらを回すこと。
#
# なぜ既存の stage 2 / 3 を書き換えず新しい系統にするのか
# ------------------------------------------------------
# 旧 stage 2 / stage 3 の run (k1_walk_lob_plant/2026-08-23_02-19-00,
# k1_walk_lob_plant_rough/2026-08-23_08-05-22) は checkpoint が git 追跡下にある。
# cfg を書き換えると「その model_*.pt がどの設定で出たのか」が読めなくなる。
#
# 何を見て判断したか (2026-08-23 のログ分析)
# -----------------------------------------
# * stage 2 は **4300 iteration で頭打ち** (apex 0.615 → 7700 まで平ら)。残りを
#   範囲の拡大に使うのが素直。
# * 旧 stage 3 は転移で壊れた: iteration 3 の時点で apex 0.60 → 0.24 /
#   elevation 30° → 14° / 方向誤差 7.7° → 22°。1700 iteration かけて apex 0.39 まで
#   しか戻らず、回復ペースは直近 500 iteration で +0.02。**ROUGH_ITER 3000 では
#   stage 2 の水準に戻らない。**
# * fewa の Stage 4 は「凹凸 + 360° + フルノイズ」で方向誤差 7.1-7.9° / 追従 0.86 を
#   3 run 一致で出している (band6 / band6calm / band6grounded)。この組み合わせ自体は
#   成立する。
#
# 起動直後に必ず見ること
# ----------------------
# * ログの "Loaded N tensors" / "Skipped N tensors"。actor.* が Skipped 側に並んで
#   いたら checkpoint が繋がっていない。
# * ``Curriculum/kick_expansion/alpha`` が 0 から始まっていること (1 から始まって
#   いたら pin が二重に掛かっている)。
#
# TensorBoard で追うもの (stage 2b)
# ---------------------------------
#   Curriculum/kick_expansion/alpha     0 → 1。**止まっているのが正常な状態**もある
#                                       (ゲートが閉じている = 今の実力の上限)。
#                                       3000 iteration 使って 0.3 程度で止まるなら
#                                       全方位はこのポリシーには早い。
#   Curriculum/kick_expansion/apex_ema  0.40 を割ると拡大が止まり、0.25 を割ると
#                                       戻り始める。EMA なので実測より遅れる。
#   Metrics/kick_direction/kick_apex_height   基準は stage 2 の 0.60。
#   Metrics/kick_direction/kick_dir_error_deg 基準は stage 2 の 7.7°。全方位化で
#                                       悪化するが、15° を超えたまま戻らないなら
#                                       回り込みが成立していない。
#
# TensorBoard で追うもの (stage 3b)
# ---------------------------------
# 出発点が収束済みなので「伸びしろより壊れていないこと」を先に見る。旧 stage 3 の
# 失敗 (iteration 3 で apex が 4 割落ちる) が再現していないかが最初のチェック。
#
# 使ってはいけないフラグ
# ----------------------
# * --resume: experiment_name が段ごとに違ううえ common_step_counter を同期する。
#   拡大ゲートの start_step (200) が「もう過ぎた」と判定される。段の引き継ぎは
#   常に --load_pretrained。
# * --reset_noise_std: 収束済みの std を戻すと当たり所の精度が壊れる。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_lob_plant_360.sh                  # 既定 (STAGE=23)
#   STAGE=2 ./scripts/rsl_rl/train_walk_lob_plant_360.sh          # 全方位化だけ
#   STAGE=3 ./scripts/rsl_rl/train_walk_lob_plant_360.sh          # 凹凸 + ノイズだけ
#   ITER=6000 ./scripts/rsl_rl/train_walk_lob_plant_360.sh        # stage 2b を長く
#   ROUGH_ITER=5000 ./scripts/rsl_rl/train_walk_lob_plant_360.sh
#   LOB_CKPT=logs/.../model_7600.pt ./scripts/rsl_rl/train_walk_lob_plant_360.sh
#   NUM_ENVS=2048 GPUS=4 ./scripts/rsl_rl/train_walk_lob_plant_360.sh
#
# NUM_ENVS は GPU 1 枚あたりの数 (合計は NUM_ENVS × GPUS)。

# 2 段構成なので既定は 23 (stage 1 / 2 は train_walk_lob_plant.sh 側)。
# **source より前に置くこと** — _orbit_common.sh が STAGE=${STAGE:-all} を実行する。
STAGE=${STAGE:-23}

source "$(dirname "${BASH_SOURCE[0]}")/_orbit_common.sh"

# 履歴観測 (N, 100, 55) は 1 フレーム観測の 100 倍のメモリを食う。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# stage 2b は拡大ゲートの窓 (200 → 3000) を走り切る余裕を見て 6000 を既定にする。
# ゲートが閉じている間は α が進まないので、窓の長さ = 所要時間ではない。
ITER=${ITER:-6000}
LOB360_ITER=${LOB360_ITER:-$ITER}
ROUGH_ITER=${ROUGH_ITER:-3000}

LOB360_TASK=${LOB360_TASK:-"Isaac-Velocity-Flat-K1-Walk-Lob-Plant-360-v0"}
ROUGH_TASK=${ROUGH_TASK:-"Isaac-Velocity-Rough-K1-Walk-Lob-Plant-360-v0"}

# 引き継ぎ元の log ルート。experiment_name は agents/rsl_rl_ppo_cfg.py が決めている。
LOB_LOG_ROOT=${LOB_LOG_ROOT:-"logs/rsl_rl/k1_walk_lob_plant"}
LOB360_LOG_ROOT=${LOB360_LOG_ROOT:-"logs/rsl_rl/k1_walk_lob_plant_360"}

if should_run 2; then
    # stage 2 (平坦・限定レンジ) の最終 checkpoint。履歴 → 履歴なので warm start 不要。
    LOB_CKPT="${LOB_CKPT:-$(find_latest_ckpt "$LOB_LOG_ROOT")}"
    run_stage "Stage 2b: 平坦 + 全方位 (apex 込みゲートで漸進)" \
        "$LOB360_TASK" "$LOB360_ITER" "$LOB_CKPT" "$@"
fi

if should_run 3; then
    LOB360_CKPT="${LOB360_CKPT:-$(find_latest_ckpt "$LOB360_LOG_ROOT")}"
    echo "[NOTE] stage 3b は拡大ゲートを α = 1 (全方位) で固定して入ります。"
    echo "[NOTE] stage 2b の Curriculum/kick_expansion/alpha が 1.0 に届いているか"
    echo "[NOTE] TensorBoard で確認してください。届いていないなら env cfg の"
    echo "[NOTE] pin_curricula_at_end(self, expansion_alpha=...) をその値に合わせること。"
    run_stage "Stage 3b: 凹凸 + ボール DR + fewa 方式の観測ノイズ" \
        "$ROUGH_TASK" "$ROUGH_ITER" "$LOB360_CKPT" "$@"
fi

echo "[INFO] done."
