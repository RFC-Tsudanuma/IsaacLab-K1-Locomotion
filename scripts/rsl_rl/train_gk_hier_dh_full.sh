#!/usr/bin/env bash
# ゴールキーパー デュアルヒストリー版 (arXiv:2401.16889 の試験実装):
# Stage 1 → Stage 2 を自動で連続実行する。Stage 1 の最終 ckpt は自動で解決する。
#
# 想定は貸し GPU (vast.ai 等) での放置実行。ローカルで段階的に確認するなら
# train_gk_hier_dh_stage1.sh / train_gk_hier_dh_stage2.sh を個別に使うこと。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_gk_hier_dh_full.sh
#   NUM_ENVS=2048 ./scripts/rsl_rl/train_gk_hier_dh_full.sh
#   NUM_ENVS=16 STAGE1_ITERS=5 STAGE2_ITERS=5 ./scripts/rsl_rl/train_gk_hier_dh_full.sh  # スモークテスト
#
# ★ 事前に必要なもの (どちらも git 管理外なので、貸し GPU では自分で持ち込むこと):
#     1. 凍結下位ポリシー  logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt
#        (274KB。logs/ は .gitignore されているので clone しても付いてこない)
#     2. scripts/rsl_rl/gk_hier_stage2_overrides.json
#   起動前に両方の存在をチェックし、無ければ Stage 1 を回す前に止める。
#
# ★ 途中で落ちたときは Stage 1 からやり直す必要はない。
#   STAGE1_CKPT=<出力された ckpt> ./scripts/rsl_rl/train_gk_hier_dh_stage2.sh で続きから。
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


NUM_ENVS=${NUM_ENVS:-4096}
STAGE1_ITERS=${STAGE1_ITERS:-5000}    # k1_gk_hier_dh_stage1 の既定と同じ
STAGE2_ITERS=${STAGE2_ITERS:-60000}   # k1_gk_hier_dh_stage2 の既定と同じ
STAGE1_TAG=${STAGE1_TAG:-dhfull}

FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt}
OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/gk_hier_dh_stage2_overrides.json}

# --- 前提ファイルの確認 (Stage 1 を数時間回してから Stage 2 で落ちるのを防ぐ) ---
for f in "${FROZEN_CKPT}" "${OVERRIDE_JSON}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] 必要なファイルがありません: ${f}" >&2
        echo "        logs/ は .gitignore 対象です。凍結下位 ckpt は手動で転送してください。" >&2
        exit 1
    fi
done

# Stage 1 / Stage 2 で必ず同じ値にしなければならない引数。ここに集約して取り違えを防ぐ。
#   clip / deadband が変わると Stage 1 で学習した指令の出し方がそのまま意味を失う。
#   ★ play / eval / 実機デプロイでも同じ値にすること。
COMMON_ARGS=(
    --frozen_checkpoint "${FROZEN_CKPT}"
    --low_level_obs_group low_level
    --high_action_clip 1.0 1.3 1.0
    --high_action_deadband 0.1
    --cmd_scale_range 0.8 1.0
    --cmd_delay_range 1 3
    --headless
    --num_envs "${NUM_ENVS}"
)

echo "=============================================================="
echo "[Stage 1/2] ボールなし。ランダム目標 y への到達と停止 (${STAGE1_ITERS} iter, ${NUM_ENVS} envs)"
echo "=============================================================="
"${ISAACLAB_SH}" -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-GoalkeeperHierDH-Stage1-K1-v0 \
    --run_name "${STAGE1_TAG}" \
    --max_iterations "${STAGE1_ITERS}" \
    "${COMMON_ARGS[@]}"

# Stage 1 の run ディレクトリ (タイムスタンプ_タグ) から最終チェックポイントを解決
STAGE1_RUN=$(ls -td logs/rsl_rl/k1_gk_hier_dh_stage1/*_"${STAGE1_TAG}" 2>/dev/null | head -1)
if [[ -z "${STAGE1_RUN}" ]]; then
    echo "[ERROR] Stage 1 の run ディレクトリ (logs/rsl_rl/k1_gk_hier_dh_stage1/*_${STAGE1_TAG}) が見つかりません。" >&2
    exit 1
fi
# model_*.pt は自然順ソート (model_9999 < model_10000) が要るので ls -v を使う
STAGE1_CKPT=$(ls -v "${STAGE1_RUN}"/model_*.pt 2>/dev/null | tail -1)
if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "[ERROR] Stage 1 のチェックポイント (${STAGE1_RUN}/model_*.pt) が見つかりません。" >&2
    exit 1
fi

echo "=============================================================="
echo "[Stage 2/2] ゴール + ボール + 適応カリキュラム (追加 ${STAGE2_ITERS} iter)"
echo "            resume: ${STAGE1_CKPT}"
echo "=============================================================="
# ★ --max_iterations は --resume 時「追加分」であって総数ではない (train_goalkeeper.py 参照)。
"${ISAACLAB_SH}" -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-GoalkeeperHierDH-Stage2-K1-v0 \
    --override_json "${OVERRIDE_JSON}" \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --max_iterations "${STAGE2_ITERS}" \
    "${COMMON_ARGS[@]}"

STAGE2_RUN=$(ls -td logs/rsl_rl/k1_gk_hier_dh_stage2/* 2>/dev/null | head -1)
echo "=============================================================="
echo "パイプライン完了。"
echo "  Stage 1 ckpt : ${STAGE1_CKPT}"
echo "  Stage 2 出力 : ${STAGE2_RUN}"
echo "=============================================================="
