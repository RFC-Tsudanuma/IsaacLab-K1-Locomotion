#!/usr/bin/env python3
"""TUI to select run directories under logs/ and export their latest checkpoints.

For each checked directory, this finds ``model_<N>.pt`` with the largest ``N`` and
runs ``bash export_policy.sh --checkpoint <model_path>`` sequentially.

Keys:
    j / DOWN     move down
    k / UP       move up
    SPACE        toggle selection
    a            select all
    n            select none
    /            filter (substring match; ESC/Enter to exit filter)
    ENTER        run export for all checked directories
    q / ESC      quit without running
"""

from __future__ import annotations

import curses
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_ROOT = SCRIPT_DIR / "logs" / "rsl_rl" / "k1_flat"
EXPORT_SCRIPT = SCRIPT_DIR / "export_policy.sh"

MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


def latest_checkpoint(run_dir: Path) -> Path | None:
    best_iter = -1
    best_path: Path | None = None
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = MODEL_RE.match(p.name)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_iter:
            best_iter = it
            best_path = p
    return best_path


def collect_runs() -> list[tuple[Path, Path | None, int | None]]:
    runs: list[tuple[Path, Path | None, int | None]] = []
    if not LOGS_ROOT.is_dir():
        return runs
    for d in sorted(LOGS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        ckpt = latest_checkpoint(d)
        it = None
        if ckpt is not None:
            m = MODEL_RE.match(ckpt.name)
            if m:
                it = int(m.group(1))
        runs.append((d, ckpt, it))
    return runs


def safe_addstr(win, y: int, x: int, s: str, w: int, attr: int = 0) -> None:
    # Writing to the bottom-right cell scrolls/advances the cursor and raises
    # ERR. Trim by one and ignore residual errors (e.g. tiny terminals).
    if w <= 0:
        return
    try:
        win.addnstr(y, x, s, max(0, w - 1), attr)
    except curses.error:
        pass


def run_tui(stdscr: "curses._CursesWindow", runs: list[tuple[Path, Path | None, int | None]]) -> list[int]:
    curses.curs_set(0)
    stdscr.keypad(True)
    try:
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
    except curses.error:
        pass

    checked = [False] * len(runs)
    cursor = 0
    top = 0
    filter_str = ""
    in_filter = False

    def visible_indices() -> list[int]:
        if not filter_str:
            return list(range(len(runs)))
        return [i for i, (d, _, _) in enumerate(runs) if filter_str.lower() in d.name.lower()]

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        title = f" export_policy.sh runner — {LOGS_ROOT} "
        safe_addstr(stdscr, 0, 0, title.ljust(w), w, curses.A_REVERSE)

        vis = visible_indices()
        if not vis:
            safe_addstr(stdscr, 2, 2, "(no runs match)", w - 2, curses.color_pair(4))
        else:
            cursor = max(0, min(cursor, len(vis) - 1))
            list_h = h - 4
            if cursor < top:
                top = cursor
            elif cursor >= top + list_h:
                top = cursor - list_h + 1

            for row, vi in enumerate(vis[top : top + list_h]):
                y = 1 + row
                idx = vis[top + row]
                d, ckpt, it = runs[idx]
                box = "[x]" if checked[idx] else "[ ]"
                if ckpt is None:
                    info = "no model_*.pt"
                    color = curses.color_pair(4)
                else:
                    info = f"iter={it}"
                    color = curses.color_pair(2)
                line = f" {box} {d.name:<30} {info}"
                attr = curses.A_REVERSE if (top + row) == cursor else curses.A_NORMAL
                safe_addstr(stdscr, y, 0, line.ljust(w), w, attr | color)

        sel_count = sum(checked)
        status = (
            f"{sel_count} selected | {len(vis)}/{len(runs)} shown"
            f"{' | filter: ' + filter_str if filter_str else ''}"
        )
        safe_addstr(stdscr, h - 2, 0, status.ljust(w), w, curses.color_pair(1) | curses.A_REVERSE)

        if in_filter:
            help_s = f"/{filter_str}_  (ESC to cancel, ENTER to apply)"
        else:
            help_s = "j/k move  SPACE toggle  a all  n none  / filter  ENTER run  q quit"
        safe_addstr(stdscr, h - 1, 0, help_s.ljust(w), w)

        stdscr.refresh()
        ch = stdscr.getch()

        if in_filter:
            if ch in (10, 13, 27):
                in_filter = False
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                filter_str = filter_str[:-1]
            elif 32 <= ch < 127:
                filter_str += chr(ch)
                cursor = 0
                top = 0
            continue

        if ch in (ord("q"), 27):
            return []
        if ch in (ord("j"), curses.KEY_DOWN):
            if vis:
                cursor = min(cursor + 1, len(vis) - 1)
        elif ch in (ord("k"), curses.KEY_UP):
            if vis:
                cursor = max(cursor - 1, 0)
        elif ch == curses.KEY_NPAGE:
            cursor = min(cursor + (h - 4), len(vis) - 1) if vis else 0
        elif ch == curses.KEY_PPAGE:
            cursor = max(cursor - (h - 4), 0)
        elif ch in (curses.KEY_HOME, ord("g")):
            cursor = 0
        elif ch in (curses.KEY_END, ord("G")):
            cursor = max(0, len(vis) - 1)
        elif ch == ord(" "):
            if vis:
                idx = vis[cursor]
                if runs[idx][1] is not None:
                    checked[idx] = not checked[idx]
        elif ch == ord("a"):
            for i, (_, ckpt, _) in enumerate(runs):
                if ckpt is not None and (not filter_str or i in vis):
                    checked[i] = True
        elif ch == ord("n"):
            for i in range(len(checked)):
                if not filter_str or i in vis:
                    checked[i] = False
        elif ch == ord("/"):
            in_filter = True
        elif ch in (10, 13):
            return [i for i, c in enumerate(checked) if c]


def main() -> int:
    if not LOGS_ROOT.is_dir():
        print(f"logs root not found: {LOGS_ROOT}", file=sys.stderr)
        return 1
    if not EXPORT_SCRIPT.is_file():
        print(f"export script not found: {EXPORT_SCRIPT}", file=sys.stderr)
        return 1

    runs = collect_runs()
    if not runs:
        print(f"no run directories under {LOGS_ROOT}", file=sys.stderr)
        return 1

    selected = curses.wrapper(run_tui, runs)
    if not selected:
        print("nothing selected; aborting.")
        return 0

    targets = [(runs[i][0], runs[i][1], runs[i][2]) for i in selected if runs[i][1] is not None]
    print("=" * 60)
    print(f"running export_policy.sh for {len(targets)} run(s):")
    for d, ckpt, it in targets:
        print(f"  - {d.name}  (model_{it}.pt)")
    print("=" * 60)

    failures: list[str] = []
    for n, (d, ckpt, it) in enumerate(targets, 1):
        print(f"\n[{n}/{len(targets)}] {d.name} :: {ckpt.name}")
        rc = subprocess.call(["bash", str(EXPORT_SCRIPT), "--checkpoint", str(ckpt)])
        if rc != 0:
            print(f"  -> FAILED (exit {rc})")
            failures.append(d.name)
        else:
            print(f"  -> OK")

    print("\n" + "=" * 60)
    if failures:
        print(f"done with {len(failures)} failure(s):")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("all exports finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
