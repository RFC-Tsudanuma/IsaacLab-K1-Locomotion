#!/usr/bin/env bash
# 横移動特化の下位ポリシー (Isaac-GKLateral-K1-v0) の再生 (GUI)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   ./scripts/rsl_rl/play_gk_lateral.sh --checkpoint logs/rsl_rl/k1_gk_lateral/<run>/model_XXXX.pt
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TASK=${TASK:-Isaac-GKLateral-K1-Play-v0}

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task "${TASK}" \
    --num_envs 1 "$@"
