#!/usr/bin/env python3
"""Select and retrieve files from a VAST.ai-style SSH destination."""

import argparse
import curses
import fnmatch
import json
import os
import posixpath
import shlex
import subprocess
import sys
import unicodedata
from typing import Optional


DEFAULT_REMOTE_ROOT = "/workspace/IsaacLab-K1-Locomotion"
JSON_MARKER = "__RETREAVE_JSON__"

# Keep the tree and rsync exclusions aligned with retrieve_robot.py.
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".vscode/",
    ".ruff_cache",
    "node_modules",
    ".git",
    "deploy_log",
]

_RESOLVE_REMOTE_ROOT_SCRIPT = f"""
import json
import os
import sys

path = os.path.realpath(os.path.expanduser(sys.argv[1]))
if not os.path.isdir(path):
    raise NotADirectoryError(path)
print({JSON_MARKER!r} + json.dumps(path))
"""

_LIST_REMOTE_DIR_SCRIPT = f"""
import json
import os
import sys

entries = []
with os.scandir(sys.argv[1]) as iterator:
    for entry in iterator:
        entries.append([entry.name, entry.is_dir(follow_symlinks=False)])
print({JSON_MARKER!r} + json.dumps(entries))
"""


class RetrieveError(RuntimeError):
    """Expected error while connecting to or retrieving from the remote host."""


class SshDestination:
    """Parsed VAST_SSH destination and its OpenSSH options."""

    def __init__(self, value: str):
        try:
            parts = shlex.split(value)
        except ValueError as exc:
            raise RetrieveError(f"VAST_SSH を解析できません: {exc}") from exc

        # Accept both the repository convention (arguments only) and a command
        # copied verbatim from the VAST.ai UI.
        if parts and posixpath.basename(parts[0]) == "ssh":
            parts = parts[1:]
        if not parts:
            raise RetrieveError("VAST_SSH が空です。")

        self.target = parts[-1]
        self.options = parts[:-1]
        if self.target.startswith("-"):
            raise RetrieveError(
                "VAST_SSH の末尾に user@host を指定してください。"
            )

    def ssh_command(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *self.options,
            self.target,
            remote_command,
        ]

    def rsync_shell(self) -> str:
        return shlex.join(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=accept-new",
                *self.options,
            ]
        )


def _disp_width(text: str) -> int:
    """Return terminal display width, accounting for wide characters."""
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        for char in text
    )


def _fit_width(text: str, max_cols: int) -> str:
    """Trim text so its terminal display width does not exceed max_cols."""
    if max_cols <= 0:
        return ""
    width = 0
    output = []
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if width + char_width > max_cols:
            break
        output.append(char)
        width += char_width
    return "".join(output)


def _is_excluded(name: str) -> bool:
    return any(
        fnmatch.fnmatch(name, pattern.rstrip("/"))
        for pattern in EXCLUDE_PATTERNS
    )


def _run_remote_json(destination: SshDestination, argv: list[str]):
    remote_command = shlex.join(argv)
    try:
        result = subprocess.run(
            destination.ssh_command(remote_command),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RetrieveError("ssh コマンドが見つかりません。") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or f"終了コード {result.returncode}"
        raise RetrieveError(f"SSH 接続またはリモート操作に失敗しました: {detail}")

    payload = next(
        (
            line[len(JSON_MARKER) :]
            for line in reversed(result.stdout.splitlines())
            if line.startswith(JSON_MARKER)
        ),
        None,
    )
    if payload is None:
        raise RetrieveError("リモートから有効な応答を取得できませんでした。")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RetrieveError("リモートの応答を解析できませんでした。") from exc


def resolve_remote_root(destination: SshDestination, path: str) -> str:
    return _run_remote_json(
        destination,
        ["python3", "-c", _RESOLVE_REMOTE_ROOT_SCRIPT, path],
    )


def list_remote_dir(destination: SshDestination, path: str) -> list[list]:
    entries = _run_remote_json(
        destination,
        ["python3", "-c", _LIST_REMOTE_DIR_SCRIPT, path],
    )
    return sorted(entries, key=lambda entry: (not entry[1], entry[0]))


class RemoteTreeNode:
    """One file or directory in the remote selection tree."""

    def __init__(
        self,
        name: str,
        path: str,
        is_dir: bool,
        destination: SshDestination,
        parent=None,
    ):
        self.name = name
        self.path = path
        self.is_dir = is_dir
        self.destination = destination
        self.parent = parent
        self.children: Optional[list["RemoteTreeNode"]] = None
        self.expanded = False
        self.checked = False
        self.load_error: Optional[str] = None

    def load_children(self) -> None:
        if self.children is not None or not self.is_dir:
            return
        try:
            entries = list_remote_dir(self.destination, self.path)
        except RetrieveError as exc:
            self.children = []
            self.load_error = str(exc)
            return

        self.children = []
        for name, is_dir in entries:
            if _is_excluded(name):
                continue
            self.children.append(
                RemoteTreeNode(
                    name,
                    posixpath.join(self.path, name),
                    is_dir,
                    self.destination,
                    parent=self,
                )
            )


def _collect_checked(node: RemoteTreeNode, remote_root: str, result: list[str]) -> None:
    if node.checked:
        result.append(posixpath.relpath(node.path, remote_root))
        return
    if node.children:
        for child in node.children:
            _collect_checked(child, remote_root, result)


def _flatten(
    nodes: list[RemoteTreeNode], depth: int = 0, acc=None
) -> list[tuple[RemoteTreeNode, int]]:
    if acc is None:
        acc = []
    for node in nodes:
        acc.append((node, depth))
        if node.is_dir and node.expanded and node.children:
            _flatten(node.children, depth + 1, acc)
    return acc


def _run_tui(roots: list[RemoteTreeNode], remote_root: str) -> Optional[list[str]]:
    result: dict = {"paths": None}

    def addline(stdscr, y, x, value, max_x, attr=curses.A_NORMAL):
        available = max_x - 1 - x
        if available <= 0:
            return
        fitted = _fit_width(value, available)
        fitted += " " * (available - _disp_width(fitted))
        try:
            stdscr.addstr(y, x, fitted, attr)
        except curses.error:
            pass

    def draw(stdscr, visible, cursor, top, height, status):
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        header = (
            "↑↓:移動  →/Enter:開閉  Space:選択  a:全展開  "
            "d:回収実行  q:中止"
        )
        addline(stdscr, 0, 0, header, max_x)
        addline(stdscr, 1, 0, "-" * (max_x - 1), max_x)
        for row in range(height):
            index = top + row
            if index >= len(visible):
                break
            node, depth = visible[index]
            marker = ("▾" if node.expanded else "▸") if node.is_dir else " "
            check = "[x]" if node.checked else "[ ]"
            suffix = "/" if node.is_dir else ""
            line = f"{'  ' * depth}{marker} {check} {node.name}{suffix}"
            attr = curses.A_REVERSE if index == cursor else curses.A_NORMAL
            addline(stdscr, 2 + row, 0, line, max_x, attr)
        addline(stdscr, max_y - 1, 0, status, max_x, curses.A_BOLD)
        stdscr.refresh()

    def expand_all(node: RemoteTreeNode) -> Optional[str]:
        if not node.is_dir:
            return None
        node.load_children()
        node.expanded = True
        if node.load_error:
            return node.load_error
        for child in node.children or []:
            error = expand_all(child)
            if error:
                return error
        return None

    def loop(stdscr):
        curses.curs_set(0)
        cursor = 0
        top = 0
        message = ""
        for root in roots:
            root.load_children()
            root.expanded = True

        while True:
            visible = _flatten(roots)
            max_y, _ = stdscr.getmaxyx()
            height = max(1, max_y - 3)
            cursor = max(0, min(cursor, len(visible) - 1))
            if cursor < top:
                top = cursor
            elif cursor >= top + height:
                top = cursor - height + 1

            selected = []
            for root in roots:
                _collect_checked(root, remote_root, selected)
            status = message or f"選択中: {len(selected)} 項目"
            draw(stdscr, visible, cursor, top, height, status)

            key = stdscr.getch()
            message = ""
            node = visible[cursor][0] if visible else None
            if key in (curses.KEY_UP, ord("k")):
                cursor -= 1
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor += 1
            elif key == curses.KEY_PPAGE:
                cursor -= height
            elif key == curses.KEY_NPAGE:
                cursor += height
            elif key in (curses.KEY_RIGHT, ord("l"), curses.KEY_ENTER, 10, 13):
                if node and node.is_dir:
                    if not node.expanded:
                        node.load_children()
                        node.expanded = True
                        message = node.load_error or ""
                    else:
                        node.expanded = False
            elif key in (curses.KEY_LEFT, ord("h")):
                if node and node.is_dir and node.expanded:
                    node.expanded = False
                elif node and node.parent is not None:
                    parent = node.parent
                    for index, (candidate, _depth) in enumerate(_flatten(roots)):
                        if candidate is parent:
                            cursor = index
                            break
            elif key == ord(" "):
                if node:
                    node.checked = not node.checked
            elif key == ord("a"):
                for root in roots:
                    error = expand_all(root)
                    if error:
                        message = error
                        break
            elif key == ord("d"):
                paths = []
                for root in roots:
                    _collect_checked(root, remote_root, paths)
                result["paths"] = paths
                return
            elif key in (ord("q"), 27):
                result["paths"] = None
                return

    curses.wrapper(loop)
    return result["paths"]


def select_remote_paths(
    destination: SshDestination, remote_root: str
) -> Optional[list[str]]:
    label = posixpath.basename(remote_root.rstrip("/")) or remote_root
    root = RemoteTreeNode(label, remote_root, True, destination)
    root.load_children()
    if root.load_error:
        raise RetrieveError(root.load_error)
    return _run_tui([root], remote_root)


def _rsync_base_command(destination: SshDestination) -> list[str]:
    command = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        destination.rsync_shell(),
    ]
    for pattern in EXCLUDE_PATTERNS:
        command.extend(["--exclude", pattern])
    return command


def _remote_source(destination: SshDestination, path: str) -> str:
    return f"{destination.target}:{path}"


def run_rsync(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] 実行: {shlex.join(command)}")
        return

    print(f"実行: {shlex.join(command)}")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RetrieveError("rsync コマンドが見つかりません。") from exc
    except subprocess.CalledProcessError as exc:
        raise RetrieveError(f"rsync に失敗しました（終了コード {exc.returncode}）。") from exc


def retrieve_selected(
    paths: list[str],
    destination: SshDestination,
    remote_root: str,
    local_destination: str,
    dry_run: bool,
) -> None:
    if not paths:
        print("回収対象が選択されていません。")
        return

    os.makedirs(local_destination, exist_ok=True)
    command = _rsync_base_command(destination)
    for relative_path in paths:
        remote_path = posixpath.join(remote_root, relative_path)
        command.append(_remote_source(destination, remote_path))
    command.append(os.path.join(local_destination, ""))
    run_rsync(command, dry_run)


def retrieve_all(
    destination: SshDestination,
    remote_root: str,
    local_destination: str,
    dry_run: bool,
) -> None:
    os.makedirs(local_destination, exist_ok=True)
    command = _rsync_base_command(destination)
    command.append(_remote_source(destination, remote_root.rstrip("/") + "/"))
    command.append(os.path.join(local_destination, ""))
    run_rsync(command, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VAST_SSH で指定した SSH 先からデータを回収します。",
        epilog=(
            "例: VAST_SSH='-p 40712 root@ssh4.vast.ai' ./retreave.py\n"
            "リモート起点は VAST_REMOTE_ROOT でも変更できます。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("VAST_REMOTE_ROOT", DEFAULT_REMOTE_ROOT),
        help=f"リモートの起点（既定: {DEFAULT_REMOTE_ROOT}）",
    )
    parser.add_argument(
        "--dest",
        help="ローカル保存先（既定: retreave.py と同じ場所の retrieved/）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="SSH 接続と選択は行い、rsync は表示だけにする",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="TUI を使わず、リモート起点以下をすべて回収する",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vast_ssh = os.environ.get("VAST_SSH", "").strip()
    if not vast_ssh:
        print(
            "Error: VAST_SSH を指定してください。\n"
            "例: VAST_SSH='-p 40712 root@ssh4.vast.ai' ./retreave.py",
            file=sys.stderr,
        )
        return 2

    try:
        destination = SshDestination(vast_ssh)
        remote_root = resolve_remote_root(destination, args.remote_root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_destination = os.path.abspath(
            os.path.expanduser(args.dest or os.path.join(script_dir, "retrieved"))
        )

        if args.dry_run:
            print("DRY RUN MODE - ファイルはコピーしません")

        print(f"回収元: {destination.target}:{remote_root}")
        print(f"保存先: {local_destination}")

        if args.all:
            retrieve_all(
                destination,
                remote_root,
                local_destination,
                args.dry_run,
            )
            return 0

        selected = select_remote_paths(destination, remote_root)
        if selected is None:
            print("中止しました。回収は行いません。")
            return 0
        if not selected:
            print("回収対象が選択されませんでした。終了します。")
            return 0

        print("回収対象:")
        for relative_path in selected:
            print(f"  - {relative_path}")
        retrieve_selected(
            selected,
            destination,
            remote_root,
            local_destination,
            args.dry_run,
        )
        return 0
    except RetrieveError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except curses.error as exc:
        print(f"Error: TUI を開始できません: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中止しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
