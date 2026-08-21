#!/usr/bin/env bash
# 横移動特化の下位ポリシー (階層型ゴールキーパーの下位) の学習。
# 目標: **全方向で頑健に歩けて、横が速い**。現行の下位 07-28
# (k1_gk_direct_stage1/2026-07-28_17-13-15、実機デプロイ済み) の後継。
#
# ★★ 2026-08-20: **既定を from scratch に変更した** (07-28 からの --resume をやめた)。
#    2026-08-20 の実機デプロイで、07-28 系譜の延長では後退・ドリフト・上半身の揺れが
#    直らないと判断したため。報酬設計も作り替えている
#    (詳細は goalkeeper_lateral_env_cfg.py の docstring)。
#    from scratch でも 07-28 は iter 500 で最終カリキュラム段に到達しており、
#    5000 iter で最終性能の 99% に届く (07-28 の学習曲線の実測)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_gk_lateral.sh --max_iterations 5000
#   ./scripts/rsl_rl/train_gk_lateral.sh --num_envs 16 --max_iterations 5   # スモークテスト
#
# 既定は **歩行 ckpt (0524_walk.pt) からの actor ウォームスタート**。07-28 も同じ条件。
#
# 07-28 から続きをやりたい場合 (--resume では max_iterations は「**追加**の反復数」):
#   RESUME_CKPT=logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/model_14999.pt \
#       ./scripts/rsl_rl/train_gk_lateral.sh --max_iterations 3000
#
# 完全にランダム初期化 (歩行の重みも使わない。5000 iter ではまず立たない):
#   WARMSTART= ./scripts/rsl_rl/train_gk_lateral.sh --max_iterations 15000
#
# ★ 学習開始 12 分後に Train/mean_episode_length を必ず確認すること。
#   下降し続けていたら即停止 (過去に COM の DR で 4 時間潰した)。
#
# ★ 止めどきの判断は **iter 数ではなく** 次の 2 つ:
#     Curriculum/lin_vel_command/stage      … 最終段 (4) に届いたか
#     Curriculum/lin_vel_command/error_ema  … 寝たか
#   届いていなければ --resume で継ぎ足す。2026-08-19 の run は 5000 iter 回して
#   stage 0 のまま終わり、昇格まであと約 500 iter のところで止まっていた。
#
#   本タスクで特に見る項目:
#     Episode_Reward/track_lin_vel_y  … 横追従。**下がったら速度を売っている = 失敗**
#     Episode_Reward/lateral_speed_bonus … 横の全開性能。同上
#     Episode_Reward/track_lin_vel_x  … 前後追従 (★今回追加。後退対策の主指標)
#     Episode_Reward/heading_hold     … 0 に近づくか (★今回追加。ヨードリフト)
#     Episode_Reward/ground_exposure  … 0 に近づくか (★今回追加。足上げの主指標)
#     Episode_Reward/flight_phase     … 跳躍に逃げていないか
#     Episode_Reward/foot_clearance   … 足上げ (山を作る側)
#     Metrics/base_velocity/error_vel_yaw … 旋回追従。悪化させないこと
#   ☠ 露出率は **跳べば下がる** ので ground_exposure と flight_phase は必ず同時に見る。
#
# ★ 学習後は **ckpt 掃引**すること (最終 ckpt が最良とは限らない実績あり):
#     for m in 3000 4000 4999; do
#       /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/eval_gk_direct_lateral.py \
#         --checkpoint logs/rsl_rl/k1_gk_lateral/<run>/model_${m}.pt \
#         --num_envs 64 --headless
#     done

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Isaac Lab の起動スクリプト。貸し GPU (vast.ai 等) でパスが違う場合は
# ISAACLAB_SH=/path/to/isaaclab.sh を指定する。
# pip/uv で isaacsim ごと入れた環境には isaaclab.sh が無いので、その場合は
# 本スクリプトを使わず train.py を直接叩くこと:
#   uv run python scripts/rsl_rl/train.py --task Isaac-GKLateral-K1-v0 \
#       --warmstart_actor logs/rsl_rl/k1_flat/main_walk/0524_walk.pt \
#       --headless --num_envs 4096 --max_iterations 5000
ISAACLAB_SH=${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}
if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab の起動スクリプトが見つかりません: ${ISAACLAB_SH}" >&2
    echo "        ISAACLAB_SH=/path/to/isaaclab.sh を指定してください。" >&2
    exit 1
fi

# 既定は空 = from scratch。07-28 から続けたいときだけ環境変数で渡す。
RESUME_CKPT=${RESUME_CKPT-}
WARMSTART=${WARMSTART-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

EXTRA=()
if [[ -n "${RESUME_CKPT}" ]]; then
    EXTRA+=(--resume --checkpoint "${RESUME_CKPT}")
elif [[ -n "${WARMSTART}" ]]; then
    EXTRA+=(--warmstart_actor "${WARMSTART}")
fi

"${ISAACLAB_SH}" -p scripts/rsl_rl/train.py \
    --task Isaac-GKLateral-K1-v0 \
    "${EXTRA[@]}" \
    --headless --num_envs 4096 "$@"
