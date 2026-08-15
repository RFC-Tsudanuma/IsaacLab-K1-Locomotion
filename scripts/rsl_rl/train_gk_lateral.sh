#!/usr/bin/env bash
# 横移動特化の下位ポリシー (階層型ゴールキーパーの下位) の学習。
# 現行の下位 07-28 (k1_gk_direct_stage1/2026-07-28_17-13-15、実機デプロイ済み) の後継候補。
# 07-28 との違いは報酬だけ: 立ち上がり (最優先) / heading 保持 / 支持脚基準の足上げ /
# 後退ドリフト。観測 59 次元・アクション・ネットワーク形状・速度指令レンジは同一。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_gk_lateral.sh
#   ./scripts/rsl_rl/train_gk_lateral.sh --num_envs 16 --max_iterations 5   # スモークテスト
#
# 既定は **07-28 の ckpt から --resume**。定常速度の追従 (誤差 2%) はもう改善余地が
# 無く、ゼロから学習し直すとそこを作り直すことになるため。報酬だけ差し替えて過渡と
# 姿勢を上書きする方が速いし、退行リスクも小さい。
#
# ゼロから学習したい場合 (歩行 ckpt からのウォームスタートに切り替わる):
#   RESUME_CKPT= ./scripts/rsl_rl/train_gk_lateral.sh
#
# ★ --resume では max_iterations は「**追加**の反復数」として効く (rsl_rl の仕様。
#   ckpt の 15000 から連番で続くので、ログの iter は 15000 始まり)。cfg 既定は 15000 だが、
#   報酬の差し替えだけなので 3000〜5000 で効果が見えるはず。ckpt は 100 iter ごと。

#
# ★ 学習開始 12 分後に Train/mean_episode_length を必ず確認すること。
#   下降し続けていたら即停止 (過去に COM の DR で 4 時間潰した)。
#   本タスクで特に見る項目:
#     Episode_Reward/onset_speed   … 立ち上がりが伸びているか (主目的)
#     Episode_Reward/heading_hold  … 0 に近づくか (yaw ドリフトが減っているか)
#     Episode_Reward/foot_clearance… 足上げ
#     Episode_Reward/lin_vel_z_l2  … 跳躍に逃げていないか (絶対値が増え続けたら黄信号)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RESUME_CKPT=${RESUME_CKPT-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/model_14999.pt}
WARMSTART=${WARMSTART-logs/rsl_rl/k1_flat/main_walk/0524_walk.pt}

EXTRA=()
if [[ -n "${RESUME_CKPT}" ]]; then
    EXTRA+=(--resume --checkpoint "${RESUME_CKPT}")
elif [[ -n "${WARMSTART}" ]]; then
    EXTRA+=(--warmstart_actor "${WARMSTART}")
fi

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-GKLateral-K1-v0 \
    "${EXTRA[@]}" \
    --headless --num_envs 4096 "$@"
