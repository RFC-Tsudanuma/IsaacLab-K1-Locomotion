#!/usr/bin/env bash
# Export an RSL-RL checkpoint to <run>/exported/.
#
# 成果物の名前は export_policy.py 側が
#   <experiment>_<checkpoint>_<エクスポート時刻>.{pt,onnx}
#   (例 k1_walk_inside_kick_model_3600_20260822-215713.onnx)
# として書き出す (scripts/rsl_rl/export_naming.py)。
#
# 以前はここで policy.onnx を policy_<HHMMSS>.onnx へ mv して run の時刻を後付けして
# いたが、(1) run の時刻はエクスポート時刻ではない、(2) step が入らないので
# どの checkpoint から出たか分からない、という 2 点で目的を果たしていなかった。
# 命名はエクスポート時に確定するようになったので、この mv は廃止した。

set -e
source /home/satoshi/.bash_functions

_labpython2 export_policy.py \
    --task Isaac-Velocity-Flat-K1-v0 \
    --headless \
    --num_envs 1 \
    "$@"

# 書き出したものを表示しておく (名前に時刻が入るので、毎回別ファイルになる)。
ckpt_path=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--checkpoint" ]]; then
        ckpt_path="$arg"
        break
    fi
    prev="$arg"
done

if [[ -n "$ckpt_path" ]]; then
    run_dir="$(dirname "$ckpt_path")"
    if [[ -d "${run_dir}/exported" ]]; then
        echo "[export_policy.sh] exported/:"
        ls -1t "${run_dir}/exported" | sed -e 's/^/    /'
    fi
fi
