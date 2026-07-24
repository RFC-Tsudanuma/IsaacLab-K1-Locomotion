#!/usr/bin/env bash
# ゴールキーパー (直接制御版) Stage 1: 12 関節を直接制御し、横移動に特化した歩容を学習する。
# 凍結歩行の横移動 (0.66 m/s) がセーブ率の頭打ちだったため、横移動そのものを学習対象にした。
# locomotion の速度コマンド追従タスクがベースで、コマンド範囲を横重視 (vx ±1.0 / vy ±1.5)、
# 横方向の追従報酬と実速度ボーナスを上乗せしてある。ボール系の観測はゼロのダミー (次元は確保)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_gk_direct_stage1.sh
#   ./scripts/rsl_rl/train_gk_direct_stage1.sh --num_envs 16 --max_iterations 5   # スモークテスト
#
# 既定で歩行 ckpt から actor をウォームスタートする (先頭 49 スロットが歩行と同一構造)。
# ゼロから学習したい場合は WARMSTART= (空) を指定する。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

WARMSTART=${WARMSTART-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

EXTRA=()
if [[ -n "${WARMSTART}" ]]; then
    EXTRA+=(--warmstart_actor "${WARMSTART}")
fi

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-GoalkeeperDirect-Stage1-K1-v0 \
    "${EXTRA[@]}" \
    --headless --num_envs 4096 "$@"
