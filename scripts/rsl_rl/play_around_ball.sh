#!/usr/bin/env bash
# ボール回り込み (around_ball) ポリシーの再生 (GUI)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
# 再生スクリプト本体は階層再生で共通の play_dribble.py を task 引数で使い回す。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/play_around_ball.sh --checkpoint logs/rsl_rl/k1_around_ball/<run>/model_XXX.pt
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# frozen 歩行ポリシーのチェックポイント (FROZEN_CKPT で上書き可)
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play_dribble.py \
    --task Isaac-AroundBall-K1-Play-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --num_envs 16 "$@"
