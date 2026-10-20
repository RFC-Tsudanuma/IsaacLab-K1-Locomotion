# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ドリブル Stage2 の学習を見張る (学習とは別プロセス。Isaac Sim は起動しない)。

やること (これ以外の理由では学習を止めない):

1. NaN: コンソールログの学習指標 (Mean reward / 各 loss / action noise std) に nan が出たら学習を止める。
2. 回復の判定: iteration が --stop_check_iter (既定 22000) を初めて超えたとき、直近 --window
   iteration の転倒率 (Episode_Termination/base_height) の平均が --max_fall (既定 0.3) 以上なら学習を止める。
   判定は 1 回だけ。
3. 途中エクスポート: カリキュラムが初めて Level --export_level (既定 3) に達した checkpoint
   (model_*_curriculum.json の level が Level-1 以上の最初のもの) を、export_dribble_target.py で
   ONNX + 仕様 JSON に書き出す (別プロセス)。

判断と実行したことは <run>/watchdog.log に書く。

    nohup python scripts/rsl_rl/dribble_watchdog.py --run_dir logs/rsl_rl/k1_dribble_target/<run> \
        --console_log logs/rsl_rl/k1_dribble_target/console_<...>.log &
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

_ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
_NAN_KEYS = ("Mean reward", "loss", "Mean action noise std")


def _log(path: str, msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def _parse_console(path: str) -> tuple[dict[int, float], int | None, str | None, bool]:
    """iteration → 転倒率、最新 iteration、NaN が出た行、学習が完了したか を返す。"""
    falls: dict[int, float] = {}
    it = None
    nan_line = None
    done = False
    with open(path, errors="replace") as f:
        for line in f:
            m = _ITER_RE.search(line)
            if m:
                it = int(m.group(1))
                continue
            s = line.strip()
            if it is not None and s.startswith("Episode_Termination/base_height:"):
                try:
                    falls[it] = float(s.split(":")[1])
                except ValueError:
                    pass
            if nan_line is None and any(k in s for k in _NAN_KEYS) and re.search(r"\bnan\b", s, re.IGNORECASE):
                nan_line = f"iteration {it}: {s}"
            if s.startswith("Training time:"):
                done = True
    return falls, it, nan_line, done


def _training_pids(task: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    pids = []
    for row in out.splitlines()[1:]:
        pid, _, args = row.strip().partition(" ")
        if "scripts/rsl_rl/train.py" in args and task in args and "python3" in args:
            pids.append(int(pid))
    return pids


def _stop_training(task: str, logfile: str, reason: str) -> None:
    pids = _training_pids(task)
    _log(logfile, f"学習を止めます: {reason} (pid {pids})")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--console_log", required=True)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-K1-Dribble-Target-v0")
    parser.add_argument("--stop_check_iter", type=int, default=22000)
    parser.add_argument("--max_fall", type=float, default=0.3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--export_level", type=int, default=3)
    parser.add_argument("--interval_s", type=int, default=120)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    logfile = os.path.join(run_dir, "watchdog.log")
    state_path = os.path.join(run_dir, "watchdog_state.json")
    state = json.load(open(state_path)) if os.path.isfile(state_path) else {"stop_checked": False, "exported": None}
    _log(logfile, f"開始: run={run_dir}, console={args.console_log}, state={state}")
    export_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_dribble_target.py")

    while True:
        falls, it, nan_line, done = _parse_console(args.console_log)

        if nan_line:
            _stop_training(args.task, logfile, f"NaN を検出: {nan_line}")
            break

        if it is not None and not state["stop_checked"] and it >= args.stop_check_iter:
            window = [falls[i] for i in range(args.stop_check_iter - args.window + 1, args.stop_check_iter + 1) if i in falls]
            mean = sum(window) / max(len(window), 1)
            state["stop_checked"] = True
            state["stop_check"] = {"iteration": args.stop_check_iter, "mean_fall": mean, "n": len(window)}
            if mean >= args.max_fall:
                _stop_training(args.task, logfile, f"iteration {args.stop_check_iter} の直近 {len(window)} iteration の転倒率の平均 {mean:.3f} >= {args.max_fall}")
                json.dump(state, open(state_path, "w"), indent=2)
                break
            _log(logfile, f"回復の判定: iteration {args.stop_check_iter} の直近 {len(window)} iteration の転倒率の平均 {mean:.3f} < {args.max_fall} → 続行")

        if state["exported"] is None:
            for cur in sorted(glob.glob(os.path.join(run_dir, "model_*_curriculum.json")),
                              key=lambda p: int(re.search(r"model_(\d+)_curriculum", p).group(1))):
                try:
                    level = json.load(open(cur))["dribble_target_range"]["level"] + 1
                except (OSError, ValueError, KeyError):
                    continue
                if level >= args.export_level:
                    ckpt = cur.replace("_curriculum.json", ".pt")
                    _log(logfile, f"Level {level} に達した最初の checkpoint: {os.path.basename(ckpt)} → エクスポートします")
                    res = subprocess.run([sys.executable, export_script, "--checkpoint", ckpt], capture_output=True, text=True)
                    tail = "\n".join(l for l in res.stdout.splitlines() if l.startswith("[export]") or l.startswith("[verify]"))
                    _log(logfile, f"エクスポート終了 (exit={res.returncode}):\n{tail}\n{res.stderr[-1500:] if res.returncode else ''}")
                    state["exported"] = {"checkpoint": ckpt, "level": level, "exit": res.returncode}
                    break

        json.dump(state, open(state_path, "w"), indent=2)
        if done or (it is not None and not _training_pids(args.task) and done):
            _log(logfile, "学習が終了したので監視を終えます")
            break
        if it is not None and not _training_pids(args.task):
            _log(logfile, f"学習プロセスが見つかりません (最後の iteration {it})。監視を終えます")
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
