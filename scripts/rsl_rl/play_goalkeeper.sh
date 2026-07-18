#!/usr/bin/env bash
# ゴールキーパーポリシーの再生 (GUI)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
# 再生スクリプト本体は goalkeeper 専用の play_goalkeeper.py (階層再生エンジン)。
#
# 使い方 (コンテナ内):
#   ./scripts/rsl_rl/play_goalkeeper.sh --checkpoint logs/rsl_rl/k1_goalkeeper/<run>/model_XXXX.pt
#   TASK=Isaac-Goalkeeper-Stage1-K1-Play-v0 ./scripts/rsl_rl/play_goalkeeper.sh \
#       --checkpoint logs/rsl_rl/k1_goalkeeper_stage1/<run>/model_XXXX.pt   # Stage1 の確認
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}
TASK=${TASK:-Isaac-Goalkeeper-K1-Play-v0}

# --high_action_clip は学習時と同じ値にすること (train_goalkeeper_*.sh と揃えている)。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play_goalkeeper.py \
    --task "${TASK}" \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 0.6 0.8 1.0 \
    --num_envs 1 "$@"
