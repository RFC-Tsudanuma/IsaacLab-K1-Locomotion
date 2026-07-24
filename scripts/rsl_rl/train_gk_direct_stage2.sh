#!/usr/bin/env bash
# ゴールキーパー (直接制御版) Stage 2: ゴール + ボールを置いてセーブを学習する。
# Stage 1 の ckpt から --resume で歩容を引き継ぐ (観測レイアウトは全ステージ共通 59 次元)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   STAGE1_CKPT=logs/rsl_rl/k1_gk_direct_stage1/<run>/model_XXXX.pt \
#       ./scripts/rsl_rl/train_gk_direct_stage2.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "STAGE1_CKPT に Stage 1 のチェックポイントを指定してください。" >&2
    echo "例: STAGE1_CKPT=logs/rsl_rl/k1_gk_direct_stage1/<run>/model_7999.pt $0" >&2
    exit 1
fi

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-GoalkeeperDirect-K1-v0 \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --headless --num_envs 4096 "$@"
