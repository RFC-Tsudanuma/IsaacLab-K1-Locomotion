#!/usr/bin/env bash
# ゴールキーパー ボール履歴版: 「どこへどれだけ速く動くか」の判断を方策に渡した構成。
#
# 直接版との差は 2 点だけ (詳細は goalkeeper/ballhist/__init__.py):
#   * 方策の velocity_commands スロットをゼロ埋め (手書きの指令を方策から隠す)
#   * ボール相対位置の履歴 (0.4s / base yaw frame) を観測の末尾に追加
#
# **Stage1 のやり直しは不要**。直接版 Stage2 の ckpt を expand_ckpt_for_ballhist.py で
# 拡張すると、初期状態が元の方策と数学的に同一になる。そこから追加学習する。
#
# 使い方 (コンテナ内・リポジトリ直下):
#   # 1. 直接版の ckpt を ボール履歴版の観測次元へ拡張する (初回だけ)
#   /isaac-sim/python.sh scripts/rsl_rl/expand_ckpt_for_ballhist.py \
#       --src logs/rsl_rl/k1_gk_direct_stage2/<run>/model_XXXXX.pt \
#       --dst logs/rsl_rl/k1_gk_direct_stage2/<run>/ballhist_seed.pt
#
#   # 2. そこから追加学習
#   STAGE1_CKPT=logs/rsl_rl/k1_gk_direct_stage2/<run>/ballhist_seed.pt \
#       ./scripts/rsl_rl/train_gk_ballhist.sh --max_iterations 20000
#
# ★ カリキュラム進行度は resume 元のディレクトリの curriculum_state.json から読まれる。
#   拡張 ckpt を元の run ディレクトリに置いておけば難易度を引き継げる。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Isaac Lab の起動スクリプト。貸し GPU (vast.ai 等) でパスが違う場合は
# ISAACLAB_SH=/path/to/isaaclab.sh を指定する。
ISAACLAB_SH=${ISAACLAB_SH:-/workspace/isaaclab/isaaclab.sh}
if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab の起動スクリプトが見つかりません: ${ISAACLAB_SH}" >&2
    echo "        ISAACLAB_SH=/path/to/isaaclab.sh を指定してください。" >&2
    exit 1
fi

OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/goalkeeper_stage3_overrides.json}

# 基準版 (critic/報酬に外挿式を特権情報として残す) が既定。
# 手書きゼロ版は TASK=Isaac-GoalkeeperBallHistPure-Stage2-K1-v0 を指定する。
TASK=${TASK:-Isaac-GoalkeeperBallHist-Stage2-K1-v0}

if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "STAGE1_CKPT に拡張済み ckpt を指定してください。" >&2
    echo "例: STAGE1_CKPT=logs/rsl_rl/k1_gk_direct_stage2/<run>/ballhist_seed.pt $0" >&2
    exit 1
fi

"${ISAACLAB_SH}" -p scripts/rsl_rl/train.py \
    --task "${TASK}" \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --override_json "${OVERRIDE_JSON}" \
    --headless --num_envs 4096 "$@"
