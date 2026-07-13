#!/usr/bin/env bash
# Export an RSL-RL checkpoint and rename the resulting policy.onnx to
# policy_<HHMMSS>.onnx using the trailing time of the run directory name
# (e.g. 2026-05-04_23-01-14 -> 230114). Falls back to the full run dir name
# if the time pattern is absent (e.g. "0504_best").

set -e
source /home/satoshi/.bash_functions

_labpython2 export_policy.py \
    --task Isaac-Velocity-Flat-K1-v0 \
    --headless \
    --num_envs 1 \
    "$@"

ckpt_path=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--checkpoint" ]]; then
        ckpt_path="$arg"
        break
    fi
    prev="$arg"
done

if [[ -z "$ckpt_path" ]]; then
    echo "[export_policy.sh] no --checkpoint arg; skipping rename" >&2
    exit 0
fi

run_dir="$(dirname "$ckpt_path")"
run_name="$(basename "$run_dir")"

if [[ "$run_name" =~ _([0-9]{2})-([0-9]{2})-([0-9]{2})$ ]]; then
    tag="${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
else
    tag="$run_name"
fi

src="${run_dir}/exported/policy.onnx"
dst="${run_dir}/exported/policy_${tag}.onnx"
if [[ -f "$src" ]]; then
    mv -f "$src" "$dst"
    echo "[export_policy.sh] renamed: $src -> $dst"
else
    echo "[export_policy.sh] WARN: $src not found, skipping rename" >&2
fi
