#!/usr/bin/env bash
# ゴールキーパー (直接制御版) Stage 2: ゴール + ボールを置いてセーブを学習する。
# **適応カリキュラム版** (旧 Stage 3 を統合した 1 段構成)。
# Stage 1 の ckpt から --resume で歩容を引き継ぐ (観測レイアウトは全ステージ共通 59 次元)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内):
#   STAGE1_CKPT=logs/rsl_rl/k1_gk_direct_stage1/<run>/model_XXXX.pt \
#       ./scripts/rsl_rl/train_gk_direct_stage2.sh
#
# ★ 2026-07-24: 旧「Stage2 (初速 0.5〜1.0 固定で 12000 iter) → Stage3 (適応)」の
#   2 段構成を廃止し、最初から適応カリキュラムで回す 1 段構成にした。
#   理由: 適応カリキュラム (adaptive_ball_speed) は初期値が ball_speed_max = 1.0 で、
#   成功率 EMA が閾値を超えたときだけ初速上限を上げ、下限は ball_speed_max で
#   クランプされる。つまり **旧 Stage2 の固定レンジから始まって段々速くなる** 挙動を
#   単体で内包しており、固定レンジ専用の段は不要だった。
#   むしろ「急がなくても取れる遅い球」で 12000 iter 過ごすことが、Stage 1 で
#   獲得した横移動 (実測 1.49 m/s) を忘却させる主因になっていた。
#   ウォームアップぶんは adaptive_warmup_episodes (既定 2000) が担う。
#
#   旧 2 段構成に戻したい場合は --task を Isaac-GoalkeeperDirect-K1-v0 (固定レンジ) に
#   変え、その ckpt から train_gk_direct_stage3.sh を回す。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

OVERRIDE_JSON=${OVERRIDE_JSON:-scripts/rsl_rl/goalkeeper_stage3_overrides.json}

if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "STAGE1_CKPT に Stage 1 のチェックポイントを指定してください。" >&2
    echo "例: STAGE1_CKPT=logs/rsl_rl/k1_gk_direct_stage1/<run>/model_3999.pt $0" >&2
    exit 1
fi

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-GoalkeeperDirect-Stage3-K1-v0 \
    --resume --checkpoint "${STAGE1_CKPT}" \
    --override_json "${OVERRIDE_JSON}" \
    --headless --num_envs 4096 "$@"
