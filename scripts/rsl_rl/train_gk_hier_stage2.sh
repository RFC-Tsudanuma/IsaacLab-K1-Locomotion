#!/usr/bin/env bash
# ゴールキーパー 階層版 v2 / Stage 2 の学習。
# ゴール + ボール。セーブ成功率 (EMA) に応じて「狙い先の広さ → ボール初速」の順に
# 難易度を上げる適応カリキュラム付き。Stage 1 の ckpt から --resume で継続する。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   STAGE1_CKPT=logs/rsl_rl/k1_gk_hier_stage1/<run>/model_XXXX.pt \
#       ./scripts/rsl_rl/train_gk_hier_stage2.sh
#
# ★ --override_json は **JSON 文字列ではなくファイルパス** を取る。既定で
#   gk_hier_stage2_overrides.json を読む (OVERRIDE_JSON で差し替え可)。
#   このファイルには perc_vel_bias_range 0.05〜0.15 が入っており、**外すと学習が進まない**。
#   cfg 既定の 0.5〜1.0 はボール横速度より大きいノイズで、到達点予測が 43% の確率で
#   逆側のポストへ飛ぶ (詳細は JSON 内のコメント)。実際にこれを入れ忘れて 12500 iter
#   回し、カリキュラムが最初の段から一度も動かなかった。
#
# 比較対象 (直接制御版のベースライン、k1_gk_direct_stage2/2026-08-12_12-06-02 @ 76k iter):
#   success_ema 0.796 / ball_speed_hi 3.10 m/s / goal_conceded 0.451
#   ただし foot_clearance が劣化し続けていた (0.311、直近3000iterで -0.039)。
#   階層版は下位が凍結なのでこの劣化は構造的に起きない。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 凍結する下位ポリシー。**Stage 1 と必ず同じものを使うこと** (下位が変わると上位が
# 学んだタイミングの前提が丸ごと崩れる)。TorchScript を使う理由は stage1 の sh を参照。
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt}
OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/gk_hier_stage2_overrides.json}

if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "STAGE1_CKPT に Stage 1 のチェックポイントを指定してください。" >&2
    echo "例: STAGE1_CKPT=logs/rsl_rl/k1_gk_hier_stage1/<run>/model_2999.pt $0" >&2
    exit 1
fi

# ★ --high_action_clip / --high_action_deadband / --cmd_scale_range / --cmd_delay_range は
#   Stage 1 と必ず同じ値にすること。特に deadband と clip が変わると、Stage 1 で学習した
#   指令の出し方がそのまま意味を失う。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-GoalkeeperHier-Stage2-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 1.0 1.3 1.0 \
    --high_action_deadband 0.1 \
    --cmd_scale_range 0.8 1.0 \
    --cmd_delay_range 1 3 \
    --override_json "${OVERRIDE_JSON}" \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --headless --num_envs 4096 "$@"
