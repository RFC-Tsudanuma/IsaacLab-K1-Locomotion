# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import ast
import datetime as _datetime
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import yaml


@dataclass(frozen=True)
class SweepAxis:
    values: List[float]


def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_cmd_path(path: Path) -> str:
    return path.as_posix() if _is_wsl() else str(path)


def _current_python_has_isaacsim() -> bool:
    return importlib.util.find_spec("isaacsim") is not None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_limit_dicts(joint_keys: List[str]) -> Tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    import re

    cfg_path = (
        _repo_root()
        / "source"
        / "isaaclab_k1_locomotion"
        / "isaaclab_k1_locomotion"
        / "tasks"
        / "manager_based"
        / "locomotion"
        / "rough_env_cfg.py"
    )
    if not cfg_path.exists():
        raise RuntimeError(f"Config file not found for defaults: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")

    def _extract_dict(limit_name: str) -> dict[str, float]:
        pattern = rf"{limit_name}\s*=\s*(\{{.*?\}})"
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Missing {limit_name} in {cfg_path}.")
        parsed = ast.literal_eval(match.group(1))
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{limit_name} is not a dict in {cfg_path}.")
        return {str(k): float(v) for k, v in parsed.items()}

    effort_dict = _extract_dict("effort_limit")
    velocity_dict = _extract_dict("velocity_limit")

    missing = [key for key in joint_keys if key not in effort_dict or key not in velocity_dict]
    if missing:
        raise RuntimeError(f"Missing defaults for keys {missing} in {cfg_path}.")

    default_effort = {key: effort_dict[key] for key in joint_keys}
    default_velocity = {key: velocity_dict[key] for key in joint_keys}
    return default_effort, default_velocity, effort_dict, velocity_dict


def _build_axis(axis_cfg: dict[str, Any], default_end: float) -> SweepAxis:
    if "values" in axis_cfg:
        values = [float(v) for v in axis_cfg["values"]]
        if not values:
            raise ValueError("Sweep axis 'values' must not be empty.")
        return SweepAxis(values=values)

    start = float(axis_cfg["start"])
    step = float(axis_cfg["step"])
    end_raw = axis_cfg["end"]
    end = default_end if isinstance(end_raw, str) and end_raw.lower() == "default" else float(end_raw)

    if step <= 0:
        raise ValueError("Sweep axis 'step' must be > 0.")
    if start > end:
        raise ValueError("Sweep axis 'start' must be <= 'end'.")

    values: List[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += step
    if abs(values[-1] - end) > 1e-9:
        values.append(round(end, 6))
    return SweepAxis(values=values)


def _format_value(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _build_run_name(base: dict[str, Any], effort: float, velocity: float) -> str:
    run_name_format = base.get("run_name_format", "knee_eff{effort}_vel{velocity}")
    return run_name_format.format(effort=_format_value(effort), velocity=_format_value(velocity))


def _train_log_root(base: dict[str, Any]) -> Path:
    experiment_name = base.get("experiment_name")
    if not experiment_name:
        raise RuntimeError("base.experiment_name is required for resume support.")
    return _repo_root() / "logs" / "rsl_rl" / str(experiment_name)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    number = int(match.group(1)) if match else -1
    return number, path.name


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_sort_key)
    return checkpoints[-1] if checkpoints else None


def _find_latest_train_run_dir(base: dict[str, Any], run_name: str) -> Path | None:
    log_root = _train_log_root(base)
    if not log_root.exists():
        return None
    candidates = [path for path in log_root.iterdir() if path.is_dir() and path.name.endswith(f"_{run_name}")]
    if not candidates:
        candidates = [path for path in log_root.iterdir() if path.is_dir() and path.name == run_name]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _infer_resume_spec_from_checkpoint(base: dict[str, Any], checkpoint_path: Path) -> dict[str, str]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise RuntimeError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.suffix != ".pt":
        raise RuntimeError(f"Resume checkpoint must be a .pt file: {checkpoint_path}")

    run_dir = checkpoint_path.parent
    log_root = _train_log_root(base).resolve()
    try:
        run_dir.relative_to(log_root)
    except ValueError as exc:
        raise RuntimeError(
            "Resume checkpoint must be inside the experiment log root used by train.py: "
            f"{log_root}. Got: {checkpoint_path}"
        ) from exc

    return {
        "checkpoint_path": str(checkpoint_path),
        "load_run": re.escape(run_dir.name) + "$",
        "checkpoint": re.escape(checkpoint_path.name) + "$",
        "source_run_dir": str(run_dir),
    }


def _infer_index_from_run_name(
    run_dir_name: str,
    base: dict[str, Any],
    sweep_entries: list[tuple[int, float, float, str]],
) -> int | None:
    for index, effort, velocity, run_name in sweep_entries:
        if run_dir_name == run_name or run_dir_name.endswith(f"_{run_name}"):
            return index
    return None


def _state_template(
    cfg_path: Path,
    base: dict[str, Any],
    sweep_entries: list[tuple[int, float, float, str]],
    runner_log_root: Path,
) -> dict[str, Any]:
    now = _datetime.datetime.now().isoformat(timespec="seconds")
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "config_path": str(cfg_path.resolve()),
        "experiment_name": base.get("experiment_name"),
        "train_log_root": str(_train_log_root(base).resolve()),
        "task_runner_log_root": str(runner_log_root.resolve()),
        "runs": [
            {
                "index": index,
                "effort_limit": effort,
                "velocity_limit": velocity,
                "run_name": run_name,
                "status": "pending",
                "attempts": 0,
                "train_log_dir": None,
                "latest_checkpoint": None,
                "last_log": None,
                "resume_from": None,
                "last_return_code": None,
            }
            for index, effort, velocity, run_name in sweep_entries
        ],
    }


def _load_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict[str, Any], state_path: Path) -> None:
    state["updated_at"] = _datetime.datetime.now().isoformat(timespec="seconds")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def _resolve_resume_run_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path.resolve()


def _state_path_from_resume_run(path: Path) -> Path | None:
    if path.is_file() and path.name.endswith(".json"):
        return path
    if path.is_dir():
        candidate = path / "task_runner_state.json"
        if candidate.exists():
            return candidate
    return None


def _resume_checkpoint_from_run(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".pt":
        return path
    if path.is_dir():
        direct = _latest_checkpoint(path)
        if direct is not None:
            return direct
        child_checkpoints = sorted(path.glob("*/model_*.pt"), key=lambda p: (p.parent.stat().st_mtime, _checkpoint_sort_key(p)))
        if child_checkpoints:
            return child_checkpoints[-1]
    return None


def _state_run(state: dict[str, Any], index: int) -> dict[str, Any]:
    for run in state["runs"]:
        if int(run["index"]) == index:
            return run
    raise RuntimeError(f"Missing index {index} in task_runner state.")


def _sync_state_with_existing_logs(
    state: dict[str, Any],
    base: dict[str, Any],
    sweep_entries: list[tuple[int, float, float, str]],
    skip_completed: bool,
    keep_pending_index: int | None = None,
) -> None:
    for index, _, _, run_name in sweep_entries:
        run = _state_run(state, index)
        latest_train_run = _find_latest_train_run_dir(base, run_name)
        if latest_train_run is None:
            continue
        latest_checkpoint = _latest_checkpoint(latest_train_run)
        run["train_log_dir"] = str(latest_train_run)
        run["latest_checkpoint"] = str(latest_checkpoint) if latest_checkpoint else None
        if (
            skip_completed
            and latest_checkpoint is not None
            and index != keep_pending_index
            and run.get("status") not in {"running"}
        ):
            run["status"] = "completed"


def _resume_spec_for_index(
    state: dict[str, Any],
    index: int,
    base: dict[str, Any],
    explicit_resume: dict[str, str] | None,
    explicit_resume_index: int | None,
) -> dict[str, str] | None:
    if explicit_resume is not None and (explicit_resume_index is None or explicit_resume_index == index):
        return explicit_resume
    run = _state_run(state, index)
    checkpoint = run.get("latest_checkpoint")
    if checkpoint and Path(checkpoint).exists() and run.get("status") != "completed":
        return _infer_resume_spec_from_checkpoint(base, Path(checkpoint))
    return None


def _build_patch_code(
    effort_overrides: dict[str, float],
    velocity_overrides: dict[str, float],
    train_script: Path,
    repo_root: Path,
    patch_sim_limits: bool,
) -> str:
    import json

    effort_json = json.dumps(effort_overrides)
    velocity_json = json.dumps(velocity_overrides)
    train_path = str(train_script)
    repo_path = str(repo_root)
    isaaclab_path = str(repo_root.parent / "IsaacLab")
    patch_sim_limits_json = json.dumps(bool(patch_sim_limits))
    return "\n".join(
        [
            "import importlib",
            "import json",
            "import os",
            "import runpy",
            "import sys",
            "import types",
            f"repo_root = r\"{repo_path}\"",
            f"isaaclab_root = r\"{isaaclab_path}\"",
            f"train_path = r\"{train_path}\"",
            "train_dir = os.path.dirname(train_path)",
            "candidate_paths = [",
            "    repo_root,",
            "    os.path.join(repo_root, 'source'),",
            "    train_dir,",
            "    os.path.join(isaaclab_root, 'source', 'isaaclab'),",
            "    os.path.join(isaaclab_root, 'source', 'isaaclab_tasks'),",
            "    os.path.join(isaaclab_root, 'source', 'isaaclab_rl'),",
            "]",
            "for path in reversed(candidate_paths):",
            "    if os.path.isdir(path) and path not in sys.path:",
            "        sys.path.insert(0, path)",
            f"effort_over = json.loads({effort_json!r})",
            f"velocity_over = json.loads({velocity_json!r})",
            f"patch_sim_limits = json.loads({patch_sim_limits_json!r})",
            "print('[TASK_RUNNER] runpy.run_path setup complete')",
            "print(f'[TASK_RUNNER] train_path={train_path}')",
            "print(f'[TASK_RUNNER] effort_overrides={effort_over}')",
            "print(f'[TASK_RUNNER] velocity_overrides={velocity_over}')",
            "print(f'[TASK_RUNNER] patch_sim_limits={patch_sim_limits}')",
            "def _to_builtin(value):",
            "    try:",
            "        import torch",
            "        if isinstance(value, torch.Tensor):",
            "            data = value.detach().cpu()",
            "            if data.ndim >= 2:",
            "                data = data[0]",
            "            return data.tolist()",
            "    except Exception:",
            "        pass",
            "    return value",
            "def _fmt(value):",
            "    value = _to_builtin(value)",
            "    return repr(value)",
            "def _dump_cfg(label, env_cfg):",
            "    print(f'[TASK_RUNNER] {label}')",
            "    try:",
            "        legs = env_cfg.scene.robot.actuators['legs']",
            "        print(f'[TASK_RUNNER] env_cfg.scene.robot.actuators[\"legs\"]={legs!r}')",
            "        print(f'[TASK_RUNNER] legs.class_type={getattr(legs.class_type, \"__name__\", legs.class_type)}')",
            "        print(f'[TASK_RUNNER] legs.joint_names_expr={_fmt(legs.joint_names_expr)}')",
            "        print(f'[TASK_RUNNER] legs.effort_limit={_fmt(legs.effort_limit)}')",
            "        print(f'[TASK_RUNNER] legs.velocity_limit={_fmt(legs.velocity_limit)}')",
            "        print(f'[TASK_RUNNER] legs.effort_limit_sim={_fmt(legs.effort_limit_sim)}')",
            "        print(f'[TASK_RUNNER] legs.velocity_limit_sim={_fmt(legs.velocity_limit_sim)}')",
            "    except Exception as exc:",
            "        print(f'[TASK_RUNNER][WARN] cfg dump failed at {label}: {exc}')",
            "def _apply_overrides(env_cfg, source):",
            "    _dump_cfg(f'BEFORE PATCH ({source})', env_cfg)",
            "    legs = env_cfg.scene.robot.actuators['legs']",
            "    if legs.effort_limit is None or not isinstance(legs.effort_limit, dict):",
            "        legs.effort_limit = {}",
            "    if legs.velocity_limit is None or not isinstance(legs.velocity_limit, dict):",
            "        legs.velocity_limit = {}",
            "    for key, value in effort_over.items():",
            "        legs.effort_limit[key] = float(value)",
            "    for key, value in velocity_over.items():",
            "        legs.velocity_limit[key] = float(value)",
            "    if patch_sim_limits:",
            "        sim_effort = dict(legs.effort_limit) if isinstance(legs.effort_limit, dict) else {}",
            "        sim_velocity = dict(legs.velocity_limit) if isinstance(legs.velocity_limit, dict) else {}",
            "        if isinstance(legs.effort_limit_sim, dict):",
            "            sim_effort.update(legs.effort_limit_sim)",
            "        if isinstance(legs.velocity_limit_sim, dict):",
            "            sim_velocity.update(legs.velocity_limit_sim)",
            "        for key, value in effort_over.items():",
            "            sim_effort[key] = float(value)",
            "        for key, value in velocity_over.items():",
            "            sim_velocity[key] = float(value)",
            "        legs.effort_limit_sim = sim_effort",
            "        legs.velocity_limit_sim = sim_velocity",
            "    _dump_cfg(f'AFTER PATCH ({source})', env_cfg)",
            "def _tensor_row(value):",
            "    try:",
            "        data = value.detach().cpu()",
            "        if data.ndim >= 2:",
            "            data = data[0]",
            "        return data.tolist()",
            "    except Exception:",
            "        return value",
            "def _select(values, ids):",
            "    try:",
            "        row = _tensor_row(values)",
            "        return [row[int(i)] for i in ids]",
            "    except Exception as exc:",
            "        return f'<select failed: {exc}>'",
            "def _dump_runtime_env(env):",
            "    print('[TASK_RUNNER] RUNTIME ENV DUMP')",
            "    try:",
            "        unwrapped = getattr(env, 'unwrapped', env)",
            "        _dump_cfg('POST gym.make env.cfg', unwrapped.cfg)",
            "        robot = unwrapped.scene['robot']",
            "        legs = robot.actuators['legs']",
            "        print(f'[TASK_RUNNER] runtime resolved joint names (all)={robot.joint_names}')",
            "        matched_ids, matched_names = robot.find_joints(list(effort_over.keys()))",
            "        print(f'[TASK_RUNNER] runtime resolved joint names (target)={matched_names}')",
            "        print(f'[TASK_RUNNER] runtime target joint ids={matched_ids}')",
            "        print(f'[TASK_RUNNER] runtime legs.joint_names={legs.joint_names}')",
            "        print(f'[TASK_RUNNER] runtime legs.joint_indices={_fmt(legs.joint_indices)}')",
            "        print(f'[TASK_RUNNER] runtime legs.cfg.effort_limit={_fmt(legs.cfg.effort_limit)}')",
            "        print(f'[TASK_RUNNER] runtime legs.cfg.velocity_limit={_fmt(legs.cfg.velocity_limit)}')",
            "        print(f'[TASK_RUNNER] runtime legs.cfg.effort_limit_sim={_fmt(legs.cfg.effort_limit_sim)}')",
            "        print(f'[TASK_RUNNER] runtime legs.cfg.velocity_limit_sim={_fmt(legs.cfg.velocity_limit_sim)}')",
            "        print(f'[TASK_RUNNER] runtime legs.effort_limit row0={_fmt(legs.effort_limit)}')",
            "        print(f'[TASK_RUNNER] runtime legs.velocity_limit row0={_fmt(legs.velocity_limit)}')",
            "        print(f'[TASK_RUNNER] runtime legs.effort_limit_sim row0={_fmt(legs.effort_limit_sim)}')",
            "        print(f'[TASK_RUNNER] runtime legs.velocity_limit_sim row0={_fmt(legs.velocity_limit_sim)}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch actuator effort_limit={_select(legs.effort_limit, [legs.joint_names.index(n) for n in matched_names if n in legs.joint_names])}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch actuator velocity_limit={_select(legs.velocity_limit, [legs.joint_names.index(n) for n in matched_names if n in legs.joint_names])}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch actuator effort_limit_sim={_select(legs.effort_limit_sim, [legs.joint_names.index(n) for n in matched_names if n in legs.joint_names])}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch actuator velocity_limit_sim={_select(legs.velocity_limit_sim, [legs.joint_names.index(n) for n in matched_names if n in legs.joint_names])}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch data.joint_effort_limits={_select(robot.data.joint_effort_limits, matched_ids)}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch data.joint_vel_limits={_select(robot.data.joint_vel_limits, matched_ids)}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch PhysX max_forces={_select(robot.root_physx_view.get_dof_max_forces(), matched_ids)}')",
            "        print(f'[TASK_RUNNER] Knee_Pitch PhysX max_velocities={_select(robot.root_physx_view.get_dof_max_velocities(), matched_ids)}')",
            "    except Exception as exc:",
            "        print(f'[TASK_RUNNER][WARN] runtime dump failed: {exc}')",
            "def _install_gym_make_probe():",
            "    try:",
            "        import gymnasium as gym",
            "    except Exception as exc:",
            "        print(f'[TASK_RUNNER][WARN] gym probe not installed: {exc}')",
            "        return",
            "    original_make = gym.make",
            "    def patched_make(*args, **kwargs):",
            "        print('[TASK_RUNNER] gym.make called')",
            "        env = original_make(*args, **kwargs)",
            "        _dump_runtime_env(env)",
            "        return env",
            "    gym.make = patched_make",
            "def _lazy_hydra_task_config(task_name, agent_cfg_entry_point):",
            "    sys.modules.pop('isaaclab_tasks.utils.hydra', None)",
            "    real = importlib.import_module('isaaclab_tasks.utils.hydra')",
            "    original_load_cfg = real.load_cfg_from_registry",
            "    def patched_load_cfg_from_registry(task_name, entry_point):",
            "        cfg = original_load_cfg(task_name, entry_point)",
            "        if entry_point == 'env_cfg_entry_point':",
            "            _apply_overrides(cfg, 'load_cfg_from_registry before ConfigStore')",
            "        return cfg",
            "    real.load_cfg_from_registry = patched_load_cfg_from_registry",
            "    _install_gym_make_probe()",
            "    print('[TASK_RUNNER] hydra_task_config patched before decorator returns')",
            "    return real.hydra_task_config(task_name, agent_cfg_entry_point)",
            "shim = types.ModuleType('isaaclab_tasks.utils.hydra')",
            "shim.hydra_task_config = _lazy_hydra_task_config",
            "sys.modules['isaaclab_tasks.utils.hydra'] = shim",
            "sys.argv[0] = train_path",
            "print('[TASK_RUNNER] runpy.run_path starting')",
            "runpy.run_path(train_path, run_name='__main__')",
        ]
    )


def _write_patch_script(code: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="k1_runner_patch_"))
    script_path = temp_dir / "run_train_patched.py"
    script_path.write_text(code, encoding="utf-8")
    return script_path


def _build_command(
    base: dict[str, Any],
    train_script: Path,
    effort: float,
    velocity: float,
    joint_keys: List[str],
    effort_dict: dict[str, float],
    velocity_dict: dict[str, float],
    resume_spec: dict[str, str] | None = None,
) -> List[str]:
    python_cmd = base.get("python_command", "python")
    if python_cmd == "auto":
        env_python = os.environ.get("ISAACSIM_PYTHON")
        if env_python:
            python_cmd = env_python
        elif _current_python_has_isaacsim():
            python_cmd = sys.executable
        else:
            isaac_path = os.environ.get("ISAAC_PATH")
            if isaac_path:
                candidate = Path(isaac_path) / "python.bat"
                if candidate.exists():
                    python_cmd = str(candidate)
                else:
                    candidate = Path(isaac_path) / "python.sh"
                    if candidate.exists():
                        python_cmd = str(candidate)
            isaaclab_root = _repo_root().parent / "IsaacLab"
            if _is_wsl():
                candidate = isaaclab_root / "_isaac_sim" / "python.sh"
            else:
                candidate = isaaclab_root / "_isaac_sim" / "python.bat"
            if candidate.exists():
                python_cmd = str(candidate)
    if python_cmd == "auto":
        python_cmd = "python"

    is_windows = os.name == "nt"
    cmd = shlex.split(python_cmd, posix=not is_windows)

    script_path = _to_cmd_path(train_script)
    cmd.append(script_path)
    cmd += ["--task", base["task"]]
    if base.get("num_envs") is not None:
        cmd += ["--num_envs", str(base["num_envs"])]
    if base.get("headless", True):
        cmd.append("--headless")

    experiment_name = base.get("experiment_name")
    if experiment_name:
        cmd += ["--experiment_name", experiment_name]

    run_name = _build_run_name(base, effort, velocity)
    cmd += ["--run_name", run_name]

    if resume_spec is not None:
        cmd.append("--resume")
        cmd += ["--load_run", str(resume_spec["load_run"])]
        cmd += ["--checkpoint", str(resume_spec["checkpoint"])]
    elif base.get("resume"):
        cmd.append("--resume")
        if base.get("load_run"):
            cmd += ["--load_run", str(base["load_run"])]
        if base.get("checkpoint"):
            cmd += ["--checkpoint", str(base["checkpoint"])]

    extra_args = base.get("extra_args") or []
    if not isinstance(extra_args, list):
        raise ValueError("base.extra_args must be a list.")
    cmd += [str(arg) for arg in extra_args]

    override_mode = base.get("override_mode", "patched_register")
    if override_mode == "patched_register":
        effort_over = {key: float(effort) for key in joint_keys}
        velocity_over = {key: float(velocity) for key in joint_keys}
        patch_sim_limits = bool(base.get("patch_sim_limits", True))
        patch_code = _build_patch_code(effort_over, velocity_over, train_script, _repo_root(), patch_sim_limits)
        patch_script = _write_patch_script(patch_code)
        script_index = cmd.index(script_path)
        cmd_prefix = cmd[:script_index]
        cmd = cmd_prefix + [_to_cmd_path(patch_script)] + cmd[script_index + 1 :]
    else:
        raise ValueError("base.override_mode must be 'patched_register'.")
    return cmd


def _resolved_python_for_env(cmd: List[str]) -> str | None:
    if not cmd:
        return None
    first = Path(cmd[0])
    if first.name.lower() in {"python.bat", "python.sh"}:
        return str(first)
    return None


def _run_command_with_log(cmd: List[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("Command: " + " ".join(cmd) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=_repo_root(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def _iter_sweep(effort_axis: SweepAxis, velocity_axis: SweepAxis) -> Iterable[Tuple[int, float, float]]:
    index = 0
    for effort in effort_axis.values:
        for velocity in velocity_axis.values:
            yield index, effort, velocity
            index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep K1 Knee_Pitch limits for IsaacLab-K1-Locomotion.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("task_runner_config.yaml")),
        help="Path to task_runner_config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--start-index", type=int, default=0, help="Start sweep index (0-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of runs to execute.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override num_envs for this run.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max_iterations for this run.")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path to resume the matching sweep run from.",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        default=None,
        help="Task-runner state dir/json or train run dir containing model_*.pt to resume from.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Existing task_runner_state.json to continue, or output state path for a new sweep.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip runs marked completed in task_runner state.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    base = cfg.get("base") or {}
    if args.num_envs is not None:
        base["num_envs"] = args.num_envs
    if args.max_iterations is not None:
        extra_args = list(base.get("extra_args") or [])
        extra_args += ["--max_iterations", str(args.max_iterations)]
        base["extra_args"] = extra_args
    sweep = cfg.get("sweep") or {}

    if "task" not in base:
        raise RuntimeError("base.task is required in the config.")

    joint_keys = sweep.get("joint_keys")
    if joint_keys is None:
        joint_keys = [".*_Knee_Pitch"]
    if not isinstance(joint_keys, list) or not joint_keys:
        raise RuntimeError("sweep.joint_keys must be a non-empty list.")

    default_effort, default_velocity, effort_dict, velocity_dict = _load_limit_dicts(joint_keys)

    effort_end_default = max(default_effort.values())
    velocity_end_default = max(default_velocity.values())
    effort_axis = _build_axis(sweep["effort_limit"], effort_end_default)
    velocity_axis = _build_axis(sweep["velocity_limit"], velocity_end_default)

    train_script = _repo_root() / Path(base.get("train_script", "scripts/rsl_rl/train.py"))
    if not train_script.exists():
        raise RuntimeError(f"Train script not found: {train_script}")

    sweep_entries = [
        (index, effort, velocity, _build_run_name(base, effort, velocity))
        for index, effort, velocity in _iter_sweep(effort_axis, velocity_axis)
    ]
    skip_completed = bool(base.get("skip_completed", True)) if args.skip_completed is None else bool(args.skip_completed)

    explicit_resume: dict[str, str] | None = None
    explicit_resume_index: int | None = None
    resume_state_path: Path | None = None
    if args.resume_checkpoint and args.resume_run:
        raise RuntimeError("Use only one of --resume-checkpoint or --resume-run.")
    if args.resume_checkpoint:
        checkpoint_path = _resolve_resume_run_path(args.resume_checkpoint)
        explicit_resume = _infer_resume_spec_from_checkpoint(base, checkpoint_path)
        explicit_resume_index = _infer_index_from_run_name(
            Path(explicit_resume["source_run_dir"]).name, base, sweep_entries
        )
    elif args.resume_run:
        resume_path = _resolve_resume_run_path(args.resume_run)
        resume_state_path = _state_path_from_resume_run(resume_path)
        if resume_state_path is None:
            checkpoint_path = _resume_checkpoint_from_run(resume_path)
            if checkpoint_path is None:
                raise RuntimeError(f"No task_runner_state.json or model_*.pt found in: {resume_path}")
            explicit_resume = _infer_resume_spec_from_checkpoint(base, checkpoint_path)
            explicit_resume_index = _infer_index_from_run_name(
                Path(explicit_resume["source_run_dir"]).name, base, sweep_entries
            )

    if args.state:
        state_path_arg = _resolve_resume_run_path(args.state)
        if state_path_arg.exists() and state_path_arg.is_dir():
            resume_state_path = state_path_arg / "task_runner_state.json"
        elif state_path_arg.suffix.lower() != ".json":
            resume_state_path = state_path_arg / "task_runner_state.json"
        else:
            resume_state_path = state_path_arg

    print(f"Config: {cfg_path}")
    print(f"Default joint limits: effort_limit={default_effort}, velocity_limit={default_velocity}")
    print(f"Sweep counts: effort={len(effort_axis.values)}, velocity={len(velocity_axis.values)}")
    if resume_state_path is not None and resume_state_path.exists():
        state = _load_state(resume_state_path)
        runner_log_root = Path(state["task_runner_log_root"])
        state_path = resume_state_path
        print(f"Loaded task runner state: {state_path}")
    else:
        timestamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        runner_log_root = _repo_root() / str(base.get("task_runner_log_dir", "logs/task_runner")) / timestamp
        state_path = (
            resume_state_path
            if resume_state_path is not None and resume_state_path.name.endswith(".json")
            else runner_log_root / "task_runner_state.json"
        )
        state = _state_template(cfg_path, base, sweep_entries, runner_log_root)

    _sync_state_with_existing_logs(
        state,
        base,
        sweep_entries,
        skip_completed=skip_completed,
        keep_pending_index=explicit_resume_index,
    )

    effective_start_index = args.start_index
    if explicit_resume_index is not None:
        effective_start_index = max(effective_start_index, explicit_resume_index)
    print(f"Task runner logs: {runner_log_root}")
    print(f"Task runner state: {state_path}")
    print(f"Effective start index: {effective_start_index}")
    print(f"Patch mode: ConfigStore-before patch, patch_sim_limits={bool(base.get('patch_sim_limits', True))}")
    print(f"Skip completed: {skip_completed}")
    if explicit_resume is not None:
        if explicit_resume_index is None:
            for candidate_index, _, _, candidate_run_name in sweep_entries:
                if candidate_index < args.start_index:
                    continue
                candidate_state = _state_run(state, candidate_index)
                if skip_completed and candidate_state.get("status") == "completed":
                    continue
                explicit_resume_index = candidate_index
                break
        print(
            "Resume checkpoint: "
            f"{explicit_resume['checkpoint_path']} "
            f"(load_run={explicit_resume['load_run']}, checkpoint={explicit_resume['checkpoint']})"
        )
        if explicit_resume_index is None:
            print("[WARN] Could not infer or select a sweep index for the checkpoint.")
        else:
            print(f"Resume checkpoint matched sweep index: {explicit_resume_index}")

    executed = 0
    if not args.dry_run:
        _save_state(state, state_path)

    for index, effort, velocity, run_name in sweep_entries:
        if index < effective_start_index:
            continue
        if args.limit is not None and executed >= args.limit:
            break

        state_run = _state_run(state, index)
        if skip_completed and state_run.get("status") == "completed":
            print(f"\n[{index}] skipping completed run_name={run_name}")
            continue

        resume_spec = _resume_spec_for_index(state, index, base, explicit_resume, explicit_resume_index)
        cmd = _build_command(
            base,
            train_script,
            effort,
            velocity,
            joint_keys,
            effort_dict,
            velocity_dict,
            resume_spec=resume_spec,
        )
        attempts = int(state_run.get("attempts") or 0) + 1
        suffix = "" if attempts == 1 else f"_attempt{attempts}"
        log_path = runner_log_root / f"{index:04d}_eff{_format_value(effort)}_vel{_format_value(velocity)}{suffix}.log"
        print(f"\n[{index}] effort_limit={effort}, velocity_limit={velocity}")
        print(f"Run name: {run_name}")
        if resume_spec is not None:
            print(
                "Resume: "
                f"load_run={resume_spec['load_run']}, "
                f"checkpoint={resume_spec['checkpoint']}, "
                f"path={resume_spec['checkpoint_path']}"
            )
        print("Command:", " ".join(cmd))
        print(f"Log: {log_path}")

        executed += 1
        if args.dry_run:
            continue

        state_run["status"] = "running"
        state_run["attempts"] = attempts
        state_run["last_log"] = str(log_path)
        state_run["resume_from"] = resume_spec["checkpoint_path"] if resume_spec is not None else None
        state_run["last_command"] = cmd
        _save_state(state, state_path)

        env = os.environ.copy()
        env.pop("CONDA_PREFIX", None)
        if not env.get("ISAACSIM_PYTHON"):
            resolved_python = _resolved_python_for_env(cmd)
            if resolved_python:
                env["ISAACSIM_PYTHON"] = resolved_python
                print(f"ISAACSIM_PYTHON auto-set for child: {resolved_python}")
        return_code = _run_command_with_log(cmd, env, log_path)
        state_run["last_return_code"] = return_code
        latest_train_run = _find_latest_train_run_dir(base, run_name)
        if latest_train_run is not None:
            state_run["train_log_dir"] = str(latest_train_run)
            latest_checkpoint = _latest_checkpoint(latest_train_run)
            state_run["latest_checkpoint"] = str(latest_checkpoint) if latest_checkpoint else None
        if return_code != 0:
            state_run["status"] = "failed"
            _save_state(state, state_path)
            print(f"Run failed (index={index}, code={return_code}). Continuing.")
            continue
        state_run["status"] = "completed"
        _save_state(state, state_path)

    print("\nSweep finished.")
    if args.dry_run:
        print(f"Task runner state planned (dry-run, not written): {state_path}")
    else:
        print(f"Task runner state saved: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
