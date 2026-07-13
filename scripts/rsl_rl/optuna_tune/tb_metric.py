"""Read a scalar metric from a TensorBoard event-file directory.

Used by the Optuna driver to score each trial after train.py finishes.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REDUCE_MODES = ("last", "final", "mean_last_n", "max")


def read_metric(
    log_dir: str,
    tag: str,
    *,
    reduce: str = "mean_last_n",
    last_n: int = 30,
) -> float:
    """Return a single scalar from the latest event file under log_dir.

    Returns NaN if the dir is empty, the tag is missing, or no events were
    written (caller is expected to substitute fail_value).
    """
    if reduce not in REDUCE_MODES:
        raise ValueError(f"unknown reduce mode '{reduce}', expected one of {REDUCE_MODES}")

    event_files = _list_event_files(log_dir)
    if not event_files:
        print(f"[tb_metric] no event files under {log_dir}")
        return math.nan

    # use the EventAccumulator on the directory, which auto-merges files
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        print(
            f"[tb_metric] tag '{tag}' not found. available scalar tags: "
            f"{sorted(acc.Tags().get('scalars', []))}"
        )
        return math.nan

    values = [ev.value for ev in acc.Scalars(tag)]
    if not values:
        return math.nan

    if reduce in ("last", "final"):
        return float(values[-1])
    if reduce == "max":
        return float(max(values))
    # mean_last_n
    window = values[-last_n:] if last_n > 0 else values
    return float(sum(window) / len(window))


def _list_event_files(log_dir: str) -> Iterable[str]:
    if not os.path.isdir(log_dir):
        return []
    out = []
    for name in os.listdir(log_dir):
        if name.startswith("events.out.tfevents."):
            out.append(os.path.join(log_dir, name))
    return sorted(out)
