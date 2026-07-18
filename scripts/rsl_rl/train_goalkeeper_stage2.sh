#!/usr/bin/env bash
# ゴールキーパー Stage 2 の学習: 遅いボール (初速 0.5〜1.0 m/s、スポーン距離・角度・
# 狙い先ランダム)。Stage 1 のチェックポイントから --resume で継続する。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   STAGE1_CKPT=logs/rsl_rl/k1_goalkeeper_stage1/<run>/model_XXXX.pt \
#       ./scripts/rsl_rl/train_goalkeeper_stage2.sh
#
# 初速レンジ等を変えたいときは goalkeeper_stage3_overrides.json の形式で
#   --override_json <json> を追加する (例: {"env": {"goalkeeper.ball_speed_max": 0.8}})。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "STAGE1_CKPT に Stage 1 のチェックポイントを指定してください。" >&2
    echo "例: STAGE1_CKPT=logs/rsl_rl/k1_goalkeeper_stage1/<run>/model_3999.pt $0" >&2
    exit 1
fi

# --high_action_clip は Stage 1 と必ず同じ値にすること。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-Goalkeeper-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 0.6 0.8 1.0 \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --headless --num_envs 4096 "$@"
