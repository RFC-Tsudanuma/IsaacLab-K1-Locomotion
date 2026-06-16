#!/usr/bin/env bash
# Convenience launcher for the Optuna tuning driver.
#
#   ./scripts/rsl_rl/optuna_tune/tune.sh                                  (uses search_space.example.yaml)
#   ./scripts/rsl_rl/optuna_tune/tune.sh --config path/to/your.yaml
#   ./scripts/rsl_rl/optuna_tune/tune.sh --config ... --n_trials 50

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"

# IsaacLab launcher: honor $ISAACLAB_SH, else use the path matched by the
# project's other scripts (train.sh / play.sh source ~/.bash_functions).
ISAACLAB_SH_DEFAULT="/home/satoshi/workspace/IsaacLab-2.3.2/isaaclab.sh"
ISAACLAB_SH="${ISAACLAB_SH:-${ISAACLAB_SH_DEFAULT}}"
if [[ -x "${ISAACLAB_SH}" ]]; then
  PY=("${ISAACLAB_SH}" -p)
elif [[ -f "${ISAACLAB_SH}" ]]; then
  PY=(bash "${ISAACLAB_SH}" -p)
else
  PY=(python)
fi

# Pre-flight: optuna + pyyaml + tensorboard must be importable.
if ! "${PY[@]}" -c "import optuna, yaml, tensorboard" >/dev/null 2>&1; then
  echo "[tune.sh] missing deps. Install with:" >&2
  echo "    ${PY[*]} -m pip install optuna pyyaml tensorboard" >&2
  exit 1
fi

CONFIG_DEFAULT="${HERE}/search_space.example.yaml"

has_config=0
for a in "$@"; do
  if [[ "${a}" == "--config" || "${a}" == --config=* ]]; then
    has_config=1
    break
  fi
done

cd "${REPO_ROOT}"
if [[ "${has_config}" -eq 0 ]]; then
  exec "${PY[@]}" "${HERE}/tune.py" --config "${CONFIG_DEFAULT}" "$@"
else
  exec "${PY[@]}" "${HERE}/tune.py" "$@"
fi
