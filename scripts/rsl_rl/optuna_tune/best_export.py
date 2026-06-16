"""Export the best trial of an Optuna study as an override JSON.

The output JSON can be passed straight back to train.py:

    ./isaaclab.sh -p scripts/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-K1-v0 \
        --max_iterations 1500 --headless \
        --override_json best.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna


def _split_overrides(params: dict) -> dict:
    out: dict = {"env": {}, "agent": {}}
    for path, value in params.items():
        if path.startswith("agent."):
            out["agent"][path[len("agent."):]] = value
        elif path.startswith("env."):
            out["env"][path[len("env."):]] = value
        else:
            print(f"[best_export] skipping non-scoped param: {path}", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export best Optuna trial as override JSON.")
    parser.add_argument("--study", required=True, help="Optuna study_name")
    parser.add_argument("--storage", required=True, help="Optuna storage URL (e.g. sqlite:///path.db)")
    parser.add_argument("--out", required=True, help="Output override JSON path")
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Specific trial number to export (default: best_trial of the study)",
    )
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study, storage=args.storage)
    if args.trial is not None:
        trial = next((t for t in study.trials if t.number == args.trial), None)
        if trial is None:
            raise SystemExit(f"trial {args.trial} not found in study '{args.study}'")
    else:
        trial = study.best_trial

    overrides = _split_overrides(trial.params)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(overrides, f, indent=2)

    print(f"[best_export] wrote {out_path}")
    print(f"  study : {args.study}")
    print(f"  trial : {trial.number}")
    print(f"  value : {trial.value}")
    print(f"  log_dir: {trial.user_attrs.get('log_dir')}")


if __name__ == "__main__":
    main()
