#!/usr/bin/env bash
# ゴールキーパー Stage 3 の学習: セーブ成功率 (EMA, 閾値 0.85) に応じてボール初速上限を
# 連続的に引き上げる適応カリキュラム。Stage 2 のチェックポイントから --resume で継続。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   1. eval_goalkeeper_speed.py で Stage 1 の実効横移動速度を計測し、提案された
#      ball_speed_cap を goalkeeper_stage3_overrides.json に書く
#   2. STAGE2_CKPT=logs/rsl_rl/k1_goalkeeper_stage2/<run>/model_XXXX.pt \
#          ./scripts/rsl_rl/train_goalkeeper_stage3.sh
#
# 遷移条件・初速レンジは goalkeeper_stage3_overrides.json で制御する
# (env cfg の goalkeeper.* フィールドへのドットパス上書き)。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}
OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/goalkeeper_stage3_overrides.json}

if [[ -z "${STAGE2_CKPT}" ]]; then
    echo "STAGE2_CKPT に Stage 2 のチェックポイントを指定してください。" >&2
    echo "例: STAGE2_CKPT=logs/rsl_rl/k1_goalkeeper_stage2/<run>/model_9999.pt $0" >&2
    exit 1
fi

# --high_action_clip は Stage 1/2 と必ず同じ値にすること。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-Goalkeeper-Stage3-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 0.6 0.8 1.0 \
    --resume --checkpoint "${STAGE2_CKPT}" \
    --override_json "${OVERRIDE_JSON}" \
    --headless --num_envs 4096 "$@"
