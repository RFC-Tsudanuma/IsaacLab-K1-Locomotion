#!/usr/bin/env bash
# ボール回り込み (around_ball) の上位ポリシー学習。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
# 学習スクリプト本体は階層学習で共通の train_dribble.py を task 引数で使い回す。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_around_ball.sh
#   FROZEN_CKPT=logs/rsl_rl/k1_flat/<run>/model_XXX.pt ./scripts/rsl_rl/train_around_ball.sh
#   ./scripts/rsl_rl/train_around_ball.sh --num_envs 16 --max_iterations 5   # スモークテスト
set -e

# ログ (logs/rsl_rl/...) の相対パスをリポジトリ直下に揃えるため、必ずリポジトリ直下へ移動
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# frozen 歩行ポリシーのチェックポイント (FROZEN_CKPT で上書き可)
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_dribble.py \
    --task Isaac-AroundBall-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --headless --num_envs 4096 "$@"
