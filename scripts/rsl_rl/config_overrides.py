"""Apply dot-path overrides to env_cfg / agent_cfg from a JSON file.

JSON layout:
    {
        "env":   {"rewards.feet_phase.weight": 2.5, "scene.num_envs": 2048, ...},
        "agent": {"algorithm.learning_rate": 5e-4, "policy.init_noise_std": 0.7, ...}
    }

Used by train.py (--override_json) and the Optuna tuning pipeline
(scripts/rsl_rl/optuna_tune/tune.py) to inject sampled hyperparameters
without modifying the source config classes.
"""

from __future__ import annotations

import json
from typing import Any


def apply_overrides_from_file(json_path: str, *, env_cfg: Any, agent_cfg: Any) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)
    apply_overrides(data, env_cfg=env_cfg, agent_cfg=agent_cfg)


def apply_overrides(data: dict, *, env_cfg: Any, agent_cfg: Any) -> None:
    env_overrides = data.get("env", {})
    agent_overrides = data.get("agent", {})

    for path, value in env_overrides.items():
        _set_dotted(env_cfg, path, value, scope="env")
    for path, value in agent_overrides.items():
        _set_dotted(agent_cfg, path, value, scope="agent")

    if env_overrides or agent_overrides:
        print(
            f"[override] applied {len(env_overrides)} env + {len(agent_overrides)} agent overrides"
        )


def _set_dotted(root: Any, path: str, value: Any, *, scope: str) -> None:
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise AttributeError(
                f"[override] {scope}.{path}: intermediate attribute '{p}' missing on {type(obj).__name__}"
            )
        obj = getattr(obj, p)
    last = parts[-1]
    if not hasattr(obj, last):
        raise AttributeError(
            f"[override] {scope}.{path}: leaf attribute '{last}' missing on {type(obj).__name__}"
        )
    current = getattr(obj, last)
    coerced = _coerce(current, value)
    setattr(obj, last, coerced)
    print(f"[override] {scope}.{path}: {current!r} -> {coerced!r}")


def _coerce(current: Any, new_value: Any) -> Any:
    """Cast new_value so it matches the type of the current attribute.

    Handles int<->float promotion (YAML/JSON often emit ints where floats are
    expected) and list-of-numbers preserving element type.
    """
    if current is None:
        return new_value
    if isinstance(current, bool):
        return bool(new_value)
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(new_value, float) and not float(new_value).is_integer():
            return new_value
        return int(new_value)
    if isinstance(current, float):
        return float(new_value)
    if isinstance(current, str):
        return str(new_value)
    if isinstance(current, list):
        if not current or not isinstance(new_value, list):
            return new_value
        elem_type = type(current[0])
        if elem_type in (int, float):
            return [elem_type(v) for v in new_value]
        return new_value
    return new_value
