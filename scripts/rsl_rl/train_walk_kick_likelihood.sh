#!/usr/bin/env bash
# Train the DirectKicking-compatible policy in three stages:
#
#   1. Walk only, while preserving the 132D observation/model contract.
#   2. Kick a stationary ball toward a randomized global target.
#   3. Continue from Stage 2 with the moving-ball speed curriculum.
#
# Each transition uses --load_pretrained rather than --resume.  This transfers
# the identical model weights without carrying the preceding environment's
# iteration counter into the next stage's curriculum.
#
# Examples:
#   ./scripts/rsl_rl/train_walk_kick_likelihood.sh
#   STAGE=2 WALK_CKPT=logs/.../model_19999.pt \
#       ./scripts/rsl_rl/train_walk_kick_likelihood.sh
#   STAGE=3 STATIONARY_CKPT=logs/.../model_19999.pt \
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

NUM_ENVS=${NUM_ENVS:-4096}
ITER=${ITER:-20000}
WALK_ITER=${WALK_ITER:-$ITER}
STATIONARY_ITER=${STATIONARY_ITER:-$ITER}
MOVING_ITER=${MOVING_ITER:-$ITER}
STAGE=${STAGE:-all}

WALK_TASK=${WALK_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Walk-Phase-v0"}
STATIONARY_TASK=${STATIONARY_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-Stationary-v0"}
MOVING_TASK=${MOVING_TASK:-"Isaac-Velocity-Flat-K1-Walk-Kick-Likelihood-Global-Target-v0"}

WALK_LOG_ROOT=${WALK_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_likelihood_global_target_walk_phase"}
STATIONARY_LOG_ROOT=${STATIONARY_LOG_ROOT:-"logs/rsl_rl/k1_walk_kick_likelihood_global_target_stationary"}

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
    run_stage "Stage 1/3: walk phase" "$WALK_TASK" "$WALK_ITER" "" "$@"
fi

if should_run 2; then
    WALK_CKPT=${WALK_CKPT:-$(find_latest_ckpt "$WALK_LOG_ROOT")}
    run_stage \
        "Stage 2/3: stationary-ball kick" \
        "$STATIONARY_TASK" \
        "$STATIONARY_ITER" \
        "$WALK_CKPT" \
        "$@"
fi

if should_run 3; then
    STATIONARY_CKPT=${STATIONARY_CKPT:-$(find_latest_ckpt "$STATIONARY_LOG_ROOT")}
    run_stage \
        "Stage 3/3: moving-ball curriculum" \
        "$MOVING_TASK" \
        "$MOVING_ITER" \
        "$STATIONARY_CKPT" \
        "$@"
fi

echo "[INFO] done."
