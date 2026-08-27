#!/usr/bin/env bash
# Train the inside-kick + CVKF policy in two phases:
#
#   1. Learn the history/inside stationary-ball task with the appended CVKF.
#   2. Continue from Phase 1 with incoming balls.  Speed and the robot-frame
#      closest-point radius progress together through 0--2.0 m/s.
#
# Both phases use the same 306D actor schema (inside 223D + CVKF 83D).  The
# transition uses --load_pretrained so Phase 2 starts its moving-ball curriculum
# at stage zero without carrying Phase 1's iteration counter.
#
# Examples:
#   ./scripts/rsl_rl/train_walk_kick_likelihood.sh
#   STAGE=1 ./scripts/rsl_rl/train_walk_kick_likelihood.sh
#   STAGE=2 STATIONARY_CKPT=logs/.../model_4999.pt \
#       ./scripts/rsl_rl/train_walk_kick_likelihood.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Resolve an Isaac Lab Python entrypoint. LAB_PYTHON can override discovery.
for bash_functions_file in \
    "${BASH_FUNCTIONS:-}" \
    "$HOME/.bash_functions" \
    /home/satoshi/.bash_functions; do
    if [[ -n "$bash_functions_file" && -f "$bash_functions_file" ]]; then
        # shellcheck disable=SC1090
        source "$bash_functions_file"
        break
    fi
done

LAB_PY=""
if [[ -n "${LAB_PYTHON:-}" ]]; then
    LAB_PY="$LAB_PYTHON"
elif type _labpython2 >/dev/null 2>&1; then
    LAB_PY="_labpython2"
else
    for candidate in \
        "$REPO_ROOT/isaaclab.sh" \
        /workspace/isaaclab/isaaclab.sh \
        /isaac-sim/python.sh; do
        if [[ -x "$candidate" ]]; then
            case "$candidate" in
                *isaaclab.sh) LAB_PY="$candidate -p" ;;
                *) LAB_PY="$candidate" ;;
            esac
            break
        fi
    done
    if [[ -z "$LAB_PY" ]]; then
        for candidate in python python3; do
            if command -v "$candidate" >/dev/null 2>&1 \
                && "$candidate" -c "import isaaclab" >/dev/null 2>&1; then
                LAB_PY="$candidate"
                break
            fi
        done
    fi
fi

if [[ -z "$LAB_PY" ]]; then
    echo "[ERROR] Isaac Lab Python was not found. Set LAB_PYTHON explicitly." >&2
    exit 1
fi

echo "[INFO] python: $LAB_PY"

# The 4,096-env PPO mini-batch feeds 49,152 forecast sequences to each LSTM.
# Let the CUDA allocator grow its segments instead of fragmenting a 3+ GiB workspace.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_ENVS=${NUM_ENVS:-4096}
ITER=${ITER:-5000}
STATIONARY_ITER=${STATIONARY_ITER:-$ITER}
MOVING_ITER=${MOVING_ITER:-$ITER}
STAGE=${STAGE:-all}

STATIONARY_TASK=${STATIONARY_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Inside-CVKF-Stationary-v0"}
MOVING_TASK=${MOVING_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Inside-CVKF-Moving-v0"}

STATIONARY_LOG_ROOT=${STATIONARY_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_inside_cvkf_stationary"}

should_run() {
    local stage_number=$1
    [[ "$STAGE" == "all" || "$STAGE" == *"$stage_number"* ]]
}

find_latest_ckpt() {
    local log_root=$1
    local latest_run
    local checkpoint

    latest_run=$(find "$log_root" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    if [[ -z "$latest_run" ]]; then
        echo "[ERROR] No training run found under: $log_root" >&2
        return 1
    fi

    checkpoint=$(find "$latest_run" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -n 1)
    if [[ -z "$checkpoint" ]]; then
        echo "[ERROR] No checkpoint found under: $latest_run" >&2
        return 1
    fi
    echo "$checkpoint"
}

run_stage() {
    local label=$1
    local task=$2
    local iterations=$3
    local checkpoint=${4:-}
    local -a command
    shift 4

    echo "=============================================================="
    echo " $label  (task=$task, iters=$iterations)"
    if [[ -n "$checkpoint" ]]; then
        echo " pretrained: $checkpoint"
    fi
    echo "=============================================================="

    # LAB_PY may intentionally contain two words (for example isaaclab.sh -p).
    # shellcheck disable=SC2206
    command=($LAB_PY)
    command+=(
        scripts/rsl_rl/train.py
        --task "$task"
        --headless
        --num_envs "$NUM_ENVS"
        --max_iterations "$iterations"
    )
    if [[ -n "$checkpoint" ]]; then
        command+=(--load_pretrained "$checkpoint")
    fi
    command+=("$@")
    "${command[@]}"
}

if should_run 1; then
    run_stage \
        "Phase 1/2: stationary inside kick + CVKF" \
        "$STATIONARY_TASK" \
        "$STATIONARY_ITER" \
        "" \
        "$@"
fi

if should_run 2; then
    STATIONARY_CKPT=${STATIONARY_CKPT:-$(find_latest_ckpt "$STATIONARY_LOG_ROOT")}
    run_stage \
        "Phase 2/2: incoming-ball speed/near-point curriculum" \
        "$MOVING_TASK" \
        "$MOVING_ITER" \
        "$STATIONARY_CKPT" \
        "$@"
fi

echo "[INFO] done."
