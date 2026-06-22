"""Optuna driver for K1 locomotion hyperparameter / reward-weight tuning.

Each trial:
  1. Sample values from the user-defined search space (YAML).
  2. Write a JSON override file (env / agent dot-paths).
  3. Spawn `train.py --override_json <path> --run_name optuna_t{N} ...` as a subprocess.
  4. Locate the new logs/rsl_rl/{experiment}/{ts}_optuna_t{N}/ directory.
  5. Read the user-specified TensorBoard tag and return it as the objective.

Runs serially. Resume-friendly: pass the same study_name + storage and Optuna
will continue adding trials.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
import yaml

# allow `from optuna_tune.tb_metric import ...` regardless of CWD
HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent  # scripts/rsl_rl
REPO_ROOT = SCRIPTS_DIR.parent.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from optuna_tune.tb_metric import read_metric  # noqa: E402


# --------------------------------------------------------------------- config


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    for key in ("study_name", "storage", "direction", "n_trials", "train", "metric", "space"):
        if key not in cfg:
            raise ValueError(f"search-space yaml missing required key: '{key}'")
    if "tag" not in cfg["metric"]:
        raise ValueError("search-space yaml: metric.tag is required (no default)")
    for key in ("task", "experiment_name"):
        if key not in cfg["train"]:
            raise ValueError(f"search-space yaml: train.{key} is required")
    if not cfg["space"]:
        raise ValueError("search-space yaml: 'space' is empty — nothing to tune")


# ------------------------------------------------------------------- sampling


def _sample(trial: optuna.Trial, space: dict) -> dict:
    """Return a flat dict {dot.path: sampled_value}."""
    out: dict = {}
    for path, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"space[{path}]: must be a mapping with a 'type' key")
        t = spec["type"]
        if t == "float":
            out[path] = trial.suggest_float(path, float(spec["low"]), float(spec["high"]))
        elif t == "loguniform":
            out[path] = trial.suggest_float(path, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "int":
            step = int(spec.get("step", 1))
            out[path] = trial.suggest_int(path, int(spec["low"]), int(spec["high"]), step=step)
        elif t == "categorical":
            out[path] = trial.suggest_categorical(path, list(spec["choices"]))
        else:
            raise ValueError(f"space[{path}]: unknown type '{t}'")
    return out


def _split_overrides(flat: dict) -> dict:
    """Convert {'agent.x.y': v, 'env.a.b': v} -> {'agent': {'x.y': v}, 'env': {'a.b': v}}."""
    out: dict = {"env": {}, "agent": {}}
    for path, value in flat.items():
        if path.startswith("agent."):
            out["agent"][path[len("agent."):]] = value
        elif path.startswith("env."):
            out["env"][path[len("env."):]] = value
        else:
            raise ValueError(f"override path must start with 'agent.' or 'env.': {path}")
    return out


# ------------------------------------------------------------ subprocess plumbing


def _python_exec() -> list[str]:
    """Command prefix used to launch train.py inside the IsaacLab env.

    Resolution order:
      1. $ISAACLAB_PYTHON (split on whitespace) — full custom command
      2. $ISAACLAB_SH (or its default path) -p — matches train.sh / play.sh
      3. plain `python`
    """
    override = os.environ.get("ISAACLAB_PYTHON")
    if override:
        return override.split()
    isaaclab_sh = os.environ.get(
        "ISAACLAB_SH", "/home/satoshi/workspace/IsaacLab-2.3.2/isaaclab.sh"
    )
    if os.path.exists(isaaclab_sh):
        if os.access(isaaclab_sh, os.X_OK):
            return [isaaclab_sh, "-p"]
        return ["bash", isaaclab_sh, "-p"]
    return [sys.executable]


def _run_training(
    *,
    task: str,
    experiment_name: str,
    num_envs: int | None,
    max_iterations: int,
    seed: int,
    run_name: str,
    override_json_path: Path,
    extra_args: list[str],
    log_path: Path,
) -> int:
    cmd = _python_exec() + [
        str(SCRIPTS_DIR / "train.py"),
        "--task", task,
        "--max_iterations", str(max_iterations),
        "--seed", str(seed),
        "--run_name", run_name,
        "--logger", "tensorboard",
        "--experiment_name", experiment_name,
        "--override_json", str(override_json_path),
    ]
    if num_envs is not None:
        cmd += ["--num_envs", str(num_envs)]
    # Force all tuning trials onto GPU 1 unless the config already overrides --device.
    if not any(a == "--device" or a.startswith("--device=") for a in extra_args):
        cmd += ["--device", "cuda:1"]
    cmd += list(extra_args)

    print(f"[tune] launching: {' '.join(cmd)}")
    print(f"[tune] stdout/err -> {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(SCRIPTS_DIR))
    return proc.returncode


def _find_trial_log_dir(experiment_name: str, run_name: str, started_after: dt.datetime) -> Path | None:
    """Locate the log dir created by train.py for this trial.

    train.py writes to: <CWD>/logs/rsl_rl/<experiment>/<YYYY-MM-DD_HH-MM-SS>_<run_name>/
    We invoke train.py with cwd=SCRIPTS_DIR so logs land under SCRIPTS_DIR/logs/...
    """
    root = SCRIPTS_DIR / "logs" / "rsl_rl" / experiment_name
    if not root.exists():
        return None
    pattern = str(root / f"*_{run_name}")
    candidates = sorted(glob.glob(pattern))
    # filter to dirs created during/after this trial
    out = []
    for c in candidates:
        try:
            st = dt.datetime.fromtimestamp(os.path.getmtime(c))
        except OSError:
            continue
        if st >= started_after - dt.timedelta(seconds=5):
            out.append(c)
    return Path(out[-1]) if out else (Path(candidates[-1]) if candidates else None)


# ------------------------------------------------------------------- objective


def make_objective(cfg: dict, *, runs_dir: Path):
    train_cfg = cfg["train"]
    metric_cfg = cfg["metric"]
    fail_value = float(metric_cfg.get("fail_value", float("nan")))
    base_seed = int(cfg.get("seed", 42))
    seed_offset = int(train_cfg.get("trial_seed_offset", 1000))

    def objective(trial: optuna.Trial) -> float:
        sampled = _sample(trial, cfg["space"])
        overrides = _split_overrides(sampled)

        run_name = f"optuna_t{trial.number}"
        trial_dir = runs_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        override_json_path = trial_dir / "override.json"
        with open(override_json_path, "w") as f:
            json.dump(overrides, f, indent=2)
        trial.set_user_attr("override_json", str(override_json_path))
        trial.set_user_attr("sampled", sampled)

        started = dt.datetime.now()
        rc = _run_training(
            task=train_cfg["task"],
            experiment_name=train_cfg["experiment_name"],
            num_envs=train_cfg.get("num_envs"),
            max_iterations=int(train_cfg.get("max_iterations", 300)),
            seed=base_seed + seed_offset + trial.number,
            run_name=run_name,
            override_json_path=override_json_path,
            extra_args=list(train_cfg.get("extra_args", [])),
            log_path=trial_dir / "train.log",
        )

        if rc != 0:
            print(f"[tune] trial {trial.number} train.py exited rc={rc} — using fail_value")
            return fail_value

        log_dir = _find_trial_log_dir(train_cfg["experiment_name"], run_name, started)
        if log_dir is None:
            print(f"[tune] trial {trial.number} could not locate TB log dir — using fail_value")
            return fail_value
        trial.set_user_attr("log_dir", str(log_dir))

        value = read_metric(
            str(log_dir),
            metric_cfg["tag"],
            reduce=metric_cfg.get("reduce", "mean_last_n"),
            last_n=int(metric_cfg.get("last_n", 30)),
        )
        if value != value:  # NaN
            print(f"[tune] trial {trial.number} metric NaN — using fail_value")
            return fail_value
        print(f"[tune] trial {trial.number} -> {metric_cfg['tag']}={value:.4f}")
        return value

    return objective


# ------------------------------------------------------------------------- main


def _make_sampler(cfg: dict) -> optuna.samplers.BaseSampler:
    name = cfg.get("sampler", "tpe").lower()
    seed = int(cfg.get("seed", 42))
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"unknown sampler '{name}' (expected tpe / random / cmaes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning driver for K1 locomotion PPO + reward weights.")
    parser.add_argument("--config", required=True, help="Path to search-space YAML.")
    parser.add_argument("--n_trials", type=int, default=None, help="Override n_trials from yaml.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    n_trials = args.n_trials if args.n_trials is not None else int(cfg.get("n_trials", 20))

    storage = cfg["storage"]
    if storage.startswith("sqlite:///") and not storage.startswith("sqlite:////"):
        sqlite_path = storage[len("sqlite:///"):]
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    runs_dir = HERE / "runs" / cfg["study_name"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        direction=cfg["direction"],
        sampler=_make_sampler(cfg),
        load_if_exists=True,
    )

    print(
        f"[tune] study='{cfg['study_name']}' storage='{storage}' direction='{cfg['direction']}' "
        f"n_trials={n_trials} sampler='{cfg.get('sampler', 'tpe')}'"
    )
    print(f"[tune] resuming with {len(study.trials)} existing trial(s)")

    study.optimize(make_objective(cfg, runs_dir=runs_dir), n_trials=n_trials, gc_after_trial=True)

    print("\n=== best trial ===")
    bt = study.best_trial
    print(f"  number: {bt.number}")
    print(f"  value : {bt.value}")
    print("  params:")
    for k, v in bt.params.items():
        print(f"    {k}: {v}")
    print(f"  log_dir: {bt.user_attrs.get('log_dir')}")


if __name__ == "__main__":
    main()
