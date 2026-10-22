#!/usr/bin/env bash
# ドリブルのポリシーを、カリキュラムのレベルを指定して画面で再生する (ウィンドウを閉じると終了)。
#
#   ./scripts/rsl_rl/play_dribble_target.sh 5                        # Level5 を再生 (最新ランの最終 checkpoint)
#   ./scripts/rsl_rl/play_dribble_target.sh 8 --num_envs 4           # 体数を変える
#   ./scripts/rsl_rl/play_dribble_target.sh 3 --with_noise           # 学習時と同じ観測ノイズを入れる
#   ./scripts/rsl_rl/play_dribble_target.sh 6 --checkpoint logs/rsl_rl/k1_dribble_target/<run>/model_N.pt
#   ./scripts/rsl_rl/play_dribble_target.sh 5 --video --video_length 1000   # 画面の代わりに動画を撮る
#
# レベル: 1 近距離・直線 (2.0〜2.5m) / 2 中距離 (2.5〜4.0m) / 3 カーブ (4〜6m) / 4 L 字 (6〜8m) /
#         5 Z 字 (8〜10m) / 6 障害物 1 本 / 7 Z 字のコーナーに障害物 / 8 1m 幅のコリドー
# Isaac Sim を同時に 2 つ起動しないこと (学習中は使わない)。

set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^[1-8]$ ]]; then
    sed -n '2,13p' "$0"; exit 1
fi
LEVEL=$1; shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
LAB_PY=${LAB_PYTHON:-/isaac-sim/python.sh}
if pgrep -f "kit/python/bin/python3 scripts/rsl_rl/(train|play)" >/dev/null; then
    echo "[ERROR] 別の Isaac Sim (学習 / 再生) が動いています。終わってから実行してください。" >&2
    exit 1
fi
exec $LAB_PY scripts/rsl_rl/play_dribble_target.py --level "$LEVEL" "$@"
