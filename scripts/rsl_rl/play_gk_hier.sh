#!/usr/bin/env bash
# ゴールキーパー 階層版 v2 の再生 (GUI)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   ./scripts/rsl_rl/play_gk_hier.sh --checkpoint logs/rsl_rl/k1_gk_hier_stage2/<run>/model_XXXX.pt
#
#   # Stage1 (ボールなし・目標 y への到達と停止) の確認
#   TASK=Isaac-GoalkeeperHier-Stage1-K1-Play-v0 ./scripts/rsl_rl/play_gk_hier.sh \
#       --checkpoint logs/rsl_rl/k1_gk_hier_stage1/<run>/model_XXXX.pt
#
#   # 学習時と同じ下位 DR を掛けて頑健性を見る (既定は公称挙動を見るため DR なし)
#   ./scripts/rsl_rl/play_gk_hier.sh --checkpoint <ckpt> --cmd_scale_range 0.8 1.0 --cmd_delay_range 1 3
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt}
TASK=${TASK:-Isaac-GoalkeeperHier-Stage2-K1-Play-v0}

# ★ --high_action_clip と --high_action_deadband は学習時と同じ値にすること。
#   deadband を外すと、学習中に一度も評価されていない指令域 (ノルム < 0.1) で
#   ポリシーを走らせることになる。実際の学習値は
#   logs/rsl_rl/<experiment>/<run>/params/goalkeeper_meta.txt に記録されている。
#   下位エンベロープの DR (--cmd_scale_range / --cmd_delay_range) は既定で切ってある。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play_goalkeeper.py \
    --task "${TASK}" \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 1.0 1.3 1.0 \
    --high_action_deadband 0.1 \
    --num_envs 1 "$@"
