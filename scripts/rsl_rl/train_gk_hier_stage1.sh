#!/usr/bin/env bash
# ゴールキーパー 階層版 v2 / Stage 1 の学習。
# ボールなし。ゴール幅内 (±1.3m) のランダム目標 y への到達と停止 + 姿勢/前後位置の維持。
# このマシン用: docker コンテナ isaac-lab-base 内で実行する (isaaclab.sh -p 経由)。
#
# 使い方 (コンテナ内・どこから実行してもOK):
#   ./scripts/rsl_rl/train_gk_hier_stage1.sh
#   ./scripts/rsl_rl/train_gk_hier_stage1.sh --num_envs 16 --max_iterations 5   # スモークテスト
#
# ★ このステージは実機検証のマイルストーンを兼ねる。学習後、実機で「指定 y へ行って止まる」
#   を確認し、そこで次の 3 つを測ること。Stage 2 の ball_speed_cap をその実測から決める。
#     1. 指令 vy = 0.6 / 0.9 / 1.2 / 1.3 それぞれの定常横速度
#     2. 静止 → 定常までの立ち上がり時間 (sim では約 0.6s。ここが遅いと全部後手に回る)
#     3. 指令 0 にしてから止まるまでの距離
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 凍結する下位ポリシー (FROZEN_CKPT で上書き可)。
#
# ★ TorchScript (exported/policy.pt) を使うこと。07-28 は actor_obs_normalization=True で
#   学習されており、生の model_*.pt を渡すと _build_frozen_policy が agent cfg の
#   low_level_policy 設定でネットワークを組んでから strict=False で読むため、
#   agent cfg 側の指定を間違えると actor_obs_normalizer.* が黙って捨てられ、
#   正規化なしで走る (= 完全に壊れるが例外は出ない)。TorchScript は正規化器ごと
#   焼き込まれているのでこの罠が原理的に無い。
#   0524_walk.pt でこの問題が顕在化しなかったのは、あちらが正規化なし世代だから。
#
# ★ exported/policy.pt がどの iteration から出力されたか記録が無い場合は、使いたい
#   model_XXXXX.pt を指定して play.py で export し直してから使うこと。
FROZEN_CKPT=${FROZEN_CKPT:-logs/rsl_rl/k1_gk_direct_stage1/2026-07-28_17-13-15/exported/policy.pt}

# --high_action_clip: 上位コマンドの上限 (vx, vy, wz)。
#   07-28 の学習コマンドレンジ (vx±1.0 / vy±1.3 / wz±1.0) と同一 = 分布内。
#   実測では vy=1.5 でも 1.474 m/s と綺麗に追従するが、学習分布の外なので採らない。
#   上げたくなったら Stage2 ckpt から --resume して clip だけ変えればよい
#   (clip は wrapper 側の引数でネットワーク構造には焼き込まれていない)。
#   ★ play / eval / 実機と必ず同じ値にすること。
#
# --high_action_deadband: 指令ノルムがこれ未満なら 3 成分とも 0 に落とす。
#   ガウス方策は厳密な 0 を出せないので、これが無いと下位の停止規約 (||cmd|| < 0.05 で
#   歩行位相ゼロ) に一生入れず、キーパーがその場足踏みし続ける。
#   ★ 軸別ではなくノルム基準。07-28 は横移動中に yaw 約 10°/s・後退約 0.10 m/s の
#     定常ドリフトを持ち、上位はそれを打ち消す小さな定常オフセット (wz≈-0.175,
#     vx≈+0.10) を出し続ける必要がある。軸別だとこの補正が潰れる。
#   ★ 実機の推論ループにも同じ判定を実装すること。
#
# --cmd_scale_range / --cmd_delay_range: 下位エンベロープの DR (エピソード固定)。
#   上位は「sim の下位」の上でタイミング (何秒前に動き出すか) を学ぶので、実機の下位が
#   遅いとその前提ごと崩れる。1.3 × U(0.8,1.0) = 実効 1.04〜1.30 m/s の帯域になり、
#   実機で確認済みの 1.2 m/s が帯域の内側に入る。
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train_goalkeeper.py \
    --task Isaac-GoalkeeperHier-Stage1-K1-v0 \
    --frozen_checkpoint "${FROZEN_CKPT}" \
    --low_level_obs_group low_level \
    --high_action_clip 1.0 1.3 1.0 \
    --high_action_deadband 0.1 \
    --cmd_scale_range 0.8 1.0 \
    --cmd_delay_range 1 3 \
    --headless --num_envs 4096 "$@"
