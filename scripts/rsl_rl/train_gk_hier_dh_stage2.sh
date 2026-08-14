#!/usr/bin/env bash
# ゴールキーパー デュアルヒストリー版 (arXiv:2401.16889 の試験実装) / Stage 2 の学習。
# 中身は train_gk_hier_stage2.sh と同一で --task だけ違う。
# ゴール + ボール + 適応カリキュラム。Stage 1 の ckpt から --resume で継続する。
#
# 使い方 (コンテナ内):
#   STAGE1_CKPT=logs/rsl_rl/k1_gk_hier_dh_stage1/<run>/model_XXXX.pt \
#       ./scripts/rsl_rl/train_gk_hier_dh_stage2.sh
#
# ★ STAGE1_CKPT は **デュアルヒストリー版 Stage1** (k1_gk_hier_dh_stage1) のものを使うこと。
#   既存階層版 (k1_gk_hier_stage1) の ckpt は actor の構造が違うので読めない。
#
# ★ --override_json (既定 gk_hier_stage2_overrides.json) を外すと学習が進まない。
#   perc_vel_bias_range 0.05〜0.15 が入っており、cfg 既定の 0.5〜1.0 では到達点予測が
#   43% の確率で逆側のポストへ飛ぶ (詳細は JSON 内のコメント)。
#   ※ この試験の狙いは「ボール速度を手組み α-β ではなく履歴から学ばせる」ことなので、
#     学習が回ることを確認したら **この override を外した条件**でも比較する価値がある
#     (速度バイアスに強くなっているなら、そこで差が出るはず)。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt}
OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/gk_hier_stage2_overrides.json}

if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "STAGE1_CKPT にデュアルヒストリー版 Stage 1 のチェックポイントを指定してください。" >&2
    echo "例: STAGE1_CKPT=logs/rsl_rl/k1_gk_hier_dh_stage1/<run>/model_4999.pt $0" >&2
    exit 1
fi

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-GoalkeeperHierDH-Stage2-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 1.0 1.3 1.0 \
    --high_action_deadband 0.1 \
    --cmd_scale_range 0.8 1.0 \
    --cmd_delay_range 1 3 \
    --override_json "${OVERRIDE_JSON}" \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --headless --num_envs 4096 "$@"
