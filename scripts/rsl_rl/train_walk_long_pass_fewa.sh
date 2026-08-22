#!/usr/bin/env bash
# walk_long_pass_fewa (5-10 m の強い転がしパス) を **歩行から通しで** 学習する。
#
# 移植元: コミット 47b8863 / ブランチ fewa/walk_kick_dual_encoder_tune の
#         scripts/rsl_rl/train_walk_long_pass.sh。**この 4 段で学習した Stage 4 の
#         checkpoint が実機で動いている。** タスク ID と log ルートを fewa 系に
#         振り直し、共通部を _orbit_common.sh に寄せ、LP_CKPT / LP_ITER を足しただけ。
#
#   Stage 1: Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Walk-Phase-v0
#            ボール無し・通常の歩行コマンドで歩行だけを学習する。
#   Stage 2: Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-Pass-v0
#            限定レンジ (ボール±60°/蹴り±45°, 0.5-0.8m) で浮かせる蹴りを獲得する。
#   Stage 3: Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-360-v0
#            全方位 (360°/360°, 0.5-1.5m) + 回り込み。
#   Stage 4: Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-v0
#            速度帯を (2.0,3.0) → (3.2,5.0) へ引き上げる。
#
# 全 stage とも観測 (100, 55)・行動 12 次元・同じ並びなので、--load_pretrained で
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
# --reset_noise_std は **既定で付けない**。継承元の収束 std は 0.06-0.095 で、0.3 は
# その 3-5 倍。実測ではこれだけで歩行が壊れ、base_height 終了が 12 iteration で 0.82 まで
# 上がって kick_rate は 0.19 止まりだった。std リセットを外すと 14 iteration で
# kick_rate 0.99 / vel_ratio 0.85 に戻る。「速度域を探索させたい」意図は帯のゲート付き
# カリキュラムが代替している。足すとしても継承元の 1.5-2 倍 (RESET_NOISE_STD=0.12) から。
#
# ITER は 5000 未満にしないこと。Stage 4 の速度帯カリキュラムは公称 500 → 3000
# iteration で (2.0,3.0) → (3.2,5.0) を動かし、その後の収束に 2000 iteration を
# 見込んでいる。ただし帯は **kick_rate で開閉するゲート付き** なので、蹴れていない
# 間は進まない。終了時に Curriculum/kick_speed_range/alpha が 1.0 に達しているかを
# 必ず確認し、達していなければ LONG_ITER を伸ばすこと。
#
# 使い方:
#   ./scripts/rsl_rl/train_walk_long_pass_fewa.sh                 # 4 段を通しで実行
#   STAGE=34 ./scripts/rsl_rl/train_walk_long_pass_fewa.sh        # 3,4 だけ
#   STAGE=4  ./scripts/rsl_rl/train_walk_long_pass_fewa.sh        # Stage 4 だけ
#   STAGE=4 LP_CKPT=logs/rsl_rl/k1_walk_long_pass_fewa/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_fewa.sh             # 出発 ckpt を明示
#   LP_ITER=10000 ./scripts/rsl_rl/train_walk_long_pass_fewa.sh   # Stage 4 だけ延長
#   WALK_ITER=8000 ./scripts/rsl_rl/train_walk_long_pass_fewa.sh  # Stage 1 だけ延長
#   RESET_NOISE_STD=0.12 ./scripts/rsl_rl/train_walk_long_pass_fewa.sh
#   GPUS=2 ./scripts/rsl_rl/train_walk_long_pass_fewa.sh          # DDP (_orbit_common.sh)
#
#   # 1 フレーム観測の checkpoint から Stage 4 だけ始める場合。旧 actor を履歴 actor の
#   # 「最新フレームの列」へ移植するので、学習開始時点の挙動が旧ポリシーと一致する。
#   # 下の maybe_warm_start が checkpoint を見て自動で付けるので、通常は指定不要。
#   STAGE=4 LP_CKPT=logs/rsl_rl/k1_walk_loop_pass_360/<run>/model_<N>.pt \
#       ./scripts/rsl_rl/train_walk_long_pass_fewa.sh --warm_start_from_single_frame
#
# NOTE: STAGE で途中から始められるのは **前段をこのスクリプトで回してある場合だけ**。
#       各段は直前の段の log ルート (k1_walk_long_pass_fewa_*) から checkpoint を拾う。
#       共用タスクの既存 run (k1_walk_kick_walk_phase / k1_walk_loop_pass_360 など) は
#       1 フレーム観測で学習されていて actor の形が違うので引き継ぎ元にならない
#       (どうしても使うなら --warm_start_from_single_frame。上の例を参照)。

# 既定は 4 段通し。_orbit_common.sh は STAGE 未設定なら "all" にするので、
# その前に既定を入れておく (should_run は部分文字列一致なので "1234" で 4 段全部)。
STAGE=${STAGE:-1234}

# shellcheck source=scripts/rsl_rl/_orbit_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_orbit_common.sh"

# kick 系 (Stage 2/3/4) の既定 iteration 数。
ITER=${ITER:-5000}
WALK_ITER=${WALK_ITER:-5000}
LOOP_ITER=${LOOP_ITER:-$ITER}
LOOP360_ITER=${LOOP360_ITER:-$ITER}
# LP_ITER は LONG_ITER の別名 (ablation スクリプトと綴りを揃えるため)。
LONG_ITER=${LP_ITER:-${LONG_ITER:-$ITER}}
# 空文字なら --reset_noise_std を付けない (既定はこちら)。
RESET_NOISE_STD=${RESET_NOISE_STD-}

WALK_TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Walk-Phase-v0"
LOOP_TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-Pass-v0"
LOOP360_TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-Loop-360-v0"
LONG_TASK="Isaac-Velocity-Flat-K1-Walk-Long-Pass-Fewa-v0"

WALK_LOG_ROOT="logs/rsl_rl/k1_walk_long_pass_fewa_walk_phase"
LOOP_LOG_ROOT="logs/rsl_rl/k1_walk_long_pass_fewa_loop_pass"
LOOP360_LOG_ROOT="logs/rsl_rl/k1_walk_long_pass_fewa_loop_360"

# checkpoint が 1 フレーム観測 (素の ActorCritic) で学習されたものかを判定する。
# 履歴入力版は actor.mlp.0.weight を、1 フレーム版は actor.0.weight を持つ。
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
# 付け忘れると actor が 1 本も引き継がれず、乱数のまま歩行からやり直しになる
# (train.py 側でも止まるようにしてあるが、共用タスクの checkpoint から始めるのは
#  正当な使い方なので、ここで正しい引数を組み立てる)。
# 標準出力に足すべき引数を吐く (呼び出し側は配列に読む)。
maybe_warm_start() {
    local ckpt="$1"; shift
    if [[ "$*" == *--warm_start_from_single_frame* ]]; then
        return 0
    fi
    if is_single_frame_ckpt "$ckpt"; then
        echo "[INFO] 引き継ぎ元が 1 フレーム観測の checkpoint なので" >&2
        echo "[INFO] --warm_start_from_single_frame を自動で付けます。" >&2
        echo "--warm_start_from_single_frame"
    fi
}

# 引き継ぎ元が fewa 系列 (履歴入力) の run かどうかを見て警告する。
warn_if_foreign_ckpt() {
    local ckpt="$1" expected_root="$2"
    if [[ "$ckpt" != "$expected_root"/* ]]; then
        echo "[WARN] 引き継ぎ元が $expected_root の run ではありません:" >&2
        echo "[WARN]   $ckpt" >&2
        echo "[WARN] 起動ログの 'Skipped N tensors' を必ず確認すること (0 本なら問題なし)。" >&2
    fi
}

if should_run 1; then
    run_stage "Stage 1/4: walk phase" "$WALK_TASK" "$WALK_ITER" "" "$@"
fi

if should_run 2; then
    WALK_CKPT="${WALK_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT")}"
    warn_if_foreign_ckpt "$WALK_CKPT" "$WALK_LOG_ROOT"
    run_stage "Stage 2/4: loop_pass" "$LOOP_TASK" "$LOOP_ITER" "$WALK_CKPT" "$@"
fi

if should_run 3; then
    LOOP_CKPT="${LOOP_CKPT:-$(find_latest_ckpt "$LOOP_LOG_ROOT")}"
    warn_if_foreign_ckpt "$LOOP_CKPT" "$LOOP_LOG_ROOT"
    run_stage "Stage 3/4: loop_pass_360" "$LOOP360_TASK" "$LOOP360_ITER" "$LOOP_CKPT" "$@"
fi

if should_run 4; then
    # 出発 checkpoint の優先順位:
    #   LP_CKPT (明示。実機で動いた fewa の Stage 4 checkpoint を別マシンから
    #            持ってきたときはこれを使う)
    #   > LOOP360_CKPT (旧名)
    #   > 履歴入力版 Stage 3 の最新 run
    LONG_CKPT="${LP_CKPT:-${LOOP360_CKPT:-$(find_latest_ckpt "$LOOP360_LOG_ROOT")}}"
    warn_if_foreign_ckpt "$LONG_CKPT" "$LOOP360_LOG_ROOT"

    EXTRA_ARGS=()
    if [[ -n "$RESET_NOISE_STD" ]]; then
        EXTRA_ARGS+=(--reset_noise_std "$RESET_NOISE_STD")
    fi
    while IFS= read -r _arg; do
        [[ -n "$_arg" ]] && EXTRA_ARGS+=("$_arg")
    done < <(maybe_warm_start "$LONG_CKPT" "$@")

    echo "[INFO] reset_noise_std: ${RESET_NOISE_STD:-"(off)"}"
    run_stage "Stage 4/4: long_pass (fewa)" "$LONG_TASK" "$LONG_ITER" "$LONG_CKPT" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "$@"
fi

echo "[INFO] done."
