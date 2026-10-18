# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ドリブル Stage2 の立ち上がり (転倒率の一時的な上昇と回復) を、累計サンプル数で前回のランと比べる。

env 数が違うと 1 iteration のサンプル数が違うので、iteration ではなく
「Stage2 開始からの累計サンプル数 = (iteration - 開始 iteration) × num_envs × num_steps_per_env」で揃え、
基準のラン (2026-10-10、4096 env × 48 steps/iter) の何 iteration 目に相当するかと、そのときの基準値を並べる。

基準値 (2026-10-10 のラン、Stage2 開始からの iteration → 転倒率 Episode_Termination/base_height):
    +0 → 0.31 (開始直後)、+500 → 0.70、+1000 → 0.31、+1500 → 0.04、+2000 → 0.035

TensorBoard は EventAccumulator (size_guidance={'scalars': 0}) で読む
(生の protobuf から simple_value を読むと常に 0.0 になる)。Isaac Sim は起動しない。

    python scripts/rsl_rl/stage2_progress.py logs/rsl_rl/k1_dribble_target/<run>
"""

from __future__ import annotations

import argparse
import os

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 基準のラン (2026-10-10) の Stage2 開始からの iteration → 転倒率
REFERENCE = {0: 0.31, 500: 0.70, 1000: 0.31, 1500: 0.04, 2000: 0.035}
REF_SAMPLES_PER_ITER = 4096 * 48


def _reference_at(ref_iter: float) -> float:
    keys = sorted(REFERENCE)
    if ref_iter >= keys[-1]:
        return REFERENCE[keys[-1]]
    for a, b in zip(keys, keys[1:]):
        if a <= ref_iter <= b:
            t = (ref_iter - a) / (b - a)
            return REFERENCE[a] + t * (REFERENCE[b] - REFERENCE[a])
    return REFERENCE[keys[0]]


def _load_run_params(run_dir: str) -> tuple[int, int]:
    with open(os.path.join(run_dir, "params", "env.yaml")) as f:
        env = yaml.unsafe_load(f)
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        agent = yaml.unsafe_load(f)
    return int(env["scene"]["num_envs"]), int(agent["num_steps_per_env"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--tag", default="Episode_Termination/base_height")
    parser.add_argument("--every", type=int, default=250, help="基準ランの iteration 換算で何 iteration ごとに出すか")
    parser.add_argument("--window", type=int, default=10, help="各点の前後この iteration 数で平均する")
    args = parser.parse_args()

    num_envs, steps_per_env = _load_run_params(args.run_dir)
    samples_per_iter = num_envs * steps_per_env
    ea = EventAccumulator(args.run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars(args.tag)
    if not events:
        raise SystemExit(f"タグ {args.tag} がありません")
    start = events[0].step
    values = {e.step: e.value for e in events}
    last = events[-1].step

    print(f"run: {args.run_dir}")
    print(f"  {num_envs} env × {steps_per_env} steps = {samples_per_iter:,} サンプル/iteration"
          f" (基準 4096 × 48 = {REF_SAMPLES_PER_ITER:,})、Stage2 開始 iteration = {start}、最新 = {last}")
    print(f"  {'iteration':>9} {'開始から':>8} {'累計サンプル':>14} {'基準換算 it':>11} {args.tag:>12} {'基準値':>7}")
    ref_iter = 0
    while True:
        samples = ref_iter * REF_SAMPLES_PER_ITER
        own_iter = start + round(samples / samples_per_iter)
        if own_iter > last:
            break
        window = [values[s] for s in range(own_iter - args.window, own_iter + args.window + 1) if s in values]
        mean = sum(window) / len(window)
        print(f"  {own_iter:>9} {own_iter - start:>+8} {samples:>14,} {ref_iter:>+11} {mean:>12.3f} {_reference_at(ref_iter):>7.3f}")
        ref_iter += args.every
    samples = (last - start) * samples_per_iter
    print(f"  最新: iteration {last} (開始から +{last - start})、累計 {samples:,} サンプル"
          f" = 基準換算 +{samples / REF_SAMPLES_PER_ITER:.0f} iteration、{args.tag} = {values[last]:.3f}"
          f" (基準値 {_reference_at(samples / REF_SAMPLES_PER_ITER):.3f})")


if __name__ == "__main__":
    main()
