#!/usr/bin/env bash
# ゴールキーパー Stage 1 の学習: ボールなし。ゴール幅内 (±1.25m) のランダム目標 y への
# 速い到達と停止を学習する。目標到達で次の目標が再サンプルされる (エピソードは切らない)。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
# 学習スクリプト本体は goalkeeper 専用の train_goalkeeper.py (階層学習エンジン)。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_goalkeeper_stage1.sh
#   ./scripts/rsl_rl/train_goalkeeper_stage1.sh --num_envs 16 --max_iterations 5   # スモークテスト
#
# 学習後は eval_goalkeeper_speed.py で実効横移動速度を計測し、Stage 3 の
# ball_speed_cap を逆算すること (goalkeeper_stage3_overrides.json)。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# frozen 歩行ポリシーのチェックポイント (FROZEN_CKPT で上書き可)
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

# --high_action_clip (vx, vy, wz): キーパーの主役は vy (横ステップ)。
# 0524_walk.pt の学習カリキュラム上端 (vx±1.8, vy±0.9) の内側で、
# vy は play の実績値 0.8 まで使い、vx は前後の微調整用に控えめ。
# play / eval 側と必ず同じ値にすること。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-Goalkeeper-Stage1-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 0.6 0.8 1.0 \
    --headless --num_envs 4096 "$@"
