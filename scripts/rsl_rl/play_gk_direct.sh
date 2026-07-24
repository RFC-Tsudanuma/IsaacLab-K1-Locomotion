#!/usr/bin/env bash
# ゴールキーパー (直接制御版) の再生 (GUI)。単一ポリシーなので通常の play.py を使う。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   ./scripts/rsl_rl/play_gk_direct.sh --checkpoint logs/rsl_rl/k1_gk_direct_stage2/<run>/model_XXXX.pt
#   TASK=Isaac-GoalkeeperDirect-Stage1-K1-Play-v0 ./scripts/rsl_rl/play_gk_direct.sh \
#       --checkpoint logs/rsl_rl/k1_gk_direct_stage1/<run>/model_XXXX.pt   # Stage1 の確認
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TASK=${TASK:-Isaac-GoalkeeperDirect-K1-Play-v0}

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task "${TASK}" \
    --num_envs 1 "$@"
