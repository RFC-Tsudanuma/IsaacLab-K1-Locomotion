#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""checkpoint のパスから gym のタスク名を引く。

``logs/rsl_rl/<experiment_name>/<run>/model_N.pt`` の ``<experiment_name>`` を、
タスク登録側 (``tasks/manager_based/*/__init__.py``) と RunnerCfg 側
(``*/agents/rsl_rl_ppo_cfg*.py`` の ``self.experiment_name``) を突き合わせて
タスク名に戻す。

**IsaacLab を import しない**のがこのスクリプトの存在理由。gym のレジストリを引けば
確実だが、それには Isaac Sim の起動 (10 秒以上) が要る。単にタスク名を知りたいだけの
場面 (play_walk_kick.sh) でそれを払うのは高いので、ソースを静的に読む。

使い方::

    python3 scripts/rsl_rl/resolve_task.py logs/rsl_rl/k1_walk_kick_360/<run>/model_N.pt
    python3 scripts/rsl_rl/resolve_task.py k1_walk_kick_360          # experiment 名直接
    python3 scripts/rsl_rl/resolve_task.py --list                    # 対応表を全部出す

見つかれば Play タスク名を 1 行で stdout に出して 0 で終了。見つからなければ
stderr に候補を出して 1 で終了する。

.. note::
   静的解析なので、登録の書き方を大きく変えると引けなくなる。対応しているのは 2 形式:

   1. ``gym.register(id="...", kwargs={"rsl_rl_cfg_entry_point": "...:Cls"})``
      (改行や ``(`` での折り返しは吸収する)
   2. ``walk_lob_rough`` のテーブル駆動 ``("<id>", "<EnvCfg>", "<RunnerCfg>")``

   クラス名はファイルを跨いで衝突する (``K1FlatPPORunnerCfg`` が locomotion の
   ``rsl_rl_ppo_cfg.py`` と ``rsl_rl_ppo_cfg_kick.py`` の両方にある) ので、
   **(パッケージ名, モジュール名, クラス名) の 3 つ組で引く**こと。
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

_TASKS_ROOT = pathlib.Path(__file__).resolve().parents[2] / (
    "source/isaaclab_k1_locomotion/isaaclab_k1_locomotion/tasks/manager_based"
)

# "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:K1FooPPORunnerCfg")
_ENTRY = re.compile(r'rsl_rl_cfg_entry_point"\s*:\s*\(?\s*f?"[^"]*\.(\w+):(\w+)"')
_TABLE = re.compile(r'\("([^"]+)",\s*\n?\s*"(\w+)",\s*"(\w+)"\)')


def build_map(root: pathlib.Path = _TASKS_ROOT) -> dict[str, set[str]]:
    """``experiment_name -> {タスク名, ...}`` を作る。"""
    key2exp: dict[tuple[str, str, str], str] = {}
    for f in root.rglob("agents/rsl_rl_ppo_cfg*.py"):
        pkg, mod, cur = f.parent.parent.name, f.stem, None
        for line in f.read_text(encoding="utf-8").split("\n"):
            m = re.match(r"class (\w+)", line.strip())
            if m:
                cur = m.group(1)
            m = re.search(r'self\.experiment_name\s*=\s*"([^"]+)"', line)
            if m and cur:
                key2exp.setdefault((pkg, mod, cur), m.group(1))

    exp2task: dict[str, set[str]] = collections.defaultdict(set)
    for f in root.rglob("__init__.py"):
        pkg, src = f.parent.name, f.read_text(encoding="utf-8")
        for blk in src.split("gym.register(")[1:]:
            tid, ent = re.search(r'id\s*=\s*"([^"]+)"', blk), _ENTRY.search(blk)
            if tid and ent:
                exp = key2exp.get((pkg, ent.group(1), ent.group(2)))
                if exp:
                    exp2task[exp].add(tid.group(1))
        for tid, _env_cls, runner_cls in _TABLE.findall(src):
            exp = key2exp.get((pkg, "rsl_rl_ppo_cfg", runner_cls))
            if exp:
                exp2task[exp] |= {tid, tid.replace("-v0", "-Play-v0")}
    return exp2task


def experiment_from(arg: str) -> str:
    """checkpoint パスか experiment 名から experiment 名を取り出す。

    ``logs/rsl_rl/<exp>/<run>/model_N.pt`` を想定し、``logs/rsl_rl`` の次の要素を採る。
    見つからなければ「.pt の 2 つ上」→「そのままの文字列」の順にフォールバックする。
    """
    p = pathlib.Path(arg)
    parts = list(p.parts)
    for i in range(len(parts) - 2):
        if parts[i] == "logs" and parts[i + 1] == "rsl_rl":
            return parts[i + 2]
    if p.suffix == ".pt" and len(parts) >= 3:
        return parts[-3]
    return arg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", nargs="?", help="checkpoint パス、または experiment 名。")
    ap.add_argument("--list", action="store_true", help="対応表を全部出す。")
    ap.add_argument("--play", action="store_true", default=True, help="Play タスクを返す (既定)。")
    ap.add_argument("--train", dest="play", action="store_false", help="学習用タスクを返す。")
    args = ap.parse_args()

    exp2task = build_map()
    if args.list:
        for exp in sorted(exp2task):
            print(f"{exp:44s} {' '.join(sorted(exp2task[exp]))}")
        return 0
    if not args.target:
        ap.error("target か --list が要る")

    exp = experiment_from(args.target)
    tasks = sorted(exp2task.get(exp, ()))
    want_play = args.play
    hits = [t for t in tasks if t.endswith("-Play-v0") == want_play]

    if not hits:
        print(f"[resolve_task] experiment '{exp}' からタスク名を引けませんでした。", file=sys.stderr)
        if tasks:
            print(f"[resolve_task] 同 experiment の候補: {' '.join(tasks)}", file=sys.stderr)
        else:
            near = [e for e in exp2task if exp in e or e in exp]
            if near:
                print(f"[resolve_task] 名前が近い experiment: {' '.join(sorted(near))}", file=sys.stderr)
            print("[resolve_task] 一覧は --list。--task で明示指定もできます。", file=sys.stderr)
        return 1

    # 同じ experiment に複数タスクがぶら下がる場合 (walk_init 版など) は最短を採る。
    # 派生タスクほど ID が長くなるので、最短 = 素の本命になる。
    print(min(hits, key=len))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
