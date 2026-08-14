#!/usr/bin/env bash
# ゴールキーパー デュアルヒストリー版 (arXiv:2401.16889 の試験実装) / Stage 1 の学習。
# 中身は train_gk_hier_stage1.sh と同一で --task だけ違う。
# ボールなし。ランダム目標 y への到達と停止 + 姿勢/前後位置の維持。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_gk_hier_dh_stage1.sh
#   ./scripts/rsl_rl/train_gk_hier_dh_stage1.sh --num_envs 16 --max_iterations 5   # スモークテスト
#
# ★ 既存階層版 (k1_gk_hier_stage1) の ckpt からは --resume できない。actor の構造が違う
#   (観測 59 → 444、CNN エンコーダ追加)。ここから学習し直すこと。
# ★ このステージのボールは検出範囲外 (park_pos = 9m) にいるので、履歴のボール系
#   チャンネルは常にゼロになる。CNN が実際に働き出すのは Stage 2 から。
#   Stage 1 でも自機 pose の履歴は動くので、停止判断には効く。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 凍結する下位ポリシー。既存階層版と同じ 07-28 (実機デプロイ実績あり、横 1.28 m/s)。
# TorchScript を使う理由は train_gk_hier_stage1.sh のコメント参照。
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt}

# --high_action_clip / --high_action_deadband / --cmd_scale_range / --cmd_delay_range は
# 既存階層版とまったく同じ値にしてある。比較対象と条件を揃えるため変えないこと。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-GoalkeeperHierDH-Stage1-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 1.0 1.3 1.0 \
    --high_action_deadband 0.1 \
    --cmd_scale_range 0.8 1.0 \
    --cmd_delay_range 1 3 \
    --headless --num_envs 4096 "$@"
