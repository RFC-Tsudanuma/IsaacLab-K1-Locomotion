#!/usr/bin/env bash
# ゴールキーパー (直接制御版) Stage 3: セーブ成功率 (EMA) に応じてボール初速上限を上げる。
#
# ★★ 2026-07-24: 通常はこのスクリプトを使う必要はない ★★
#   train_gk_direct_stage2.sh が同じ適応カリキュラムタスクを Stage 1 ckpt から
#   直接回すようになった (旧 Stage2 の固定レンジ段を廃止して 1 段に統合)。
#   通常の学習は Stage1 → train_gk_direct_stage2.sh の 2 段で完結する。
#
#   本スクリプトが要るのは「統合版の学習を途中から追加で継続したい」ときだけ:
#     STAGE2_CKPT=logs/rsl_rl/k1_gk_direct_stage3/<run>/model_XXXXX.pt \
#         ./scripts/rsl_rl/train_gk_direct_stage3.sh --max_iterations 5000
#   (--max_iterations は resume 時「追加ぶん」であって総数ではない)
#
# 遷移条件・初速レンジは JSON で制御。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/goalkeeper_stage3_overrides.json}

if [[ -z "${STAGE2_CKPT}" ]]; then
    echo "STAGE2_CKPT に Stage 2 のチェックポイントを指定してください。" >&2
    echo "例: STAGE2_CKPT=logs/rsl_rl/k1_gk_direct_stage2/<run>/model_11999.pt $0" >&2
    exit 1
fi

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-GoalkeeperDirect-Stage3-K1-v0 \
    --resume --checkpoint "${STAGE2_CKPT}" \
    --override_json "${OVERRIDE_JSON}" \
    --headless --num_envs 4096 "$@"
