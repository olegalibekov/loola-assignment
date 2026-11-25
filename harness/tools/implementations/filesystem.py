"""Filesystem navigation, search, and workspace-safe mutations."""
from __future__ import annotations

import datetime as dt
import fnmatch
import os
import re
import shutil
from pathlib import Path

from ..base import ToolContext, ToolError
from ..core import _existing
from .text import _regex


SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".idea"}


def _walk(root: Path):
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            yield Path(base) / name


def _check_tree(ctx: ToolContext, root: Path) -> None:
    """Preflight recursive operations before they can follow a nested link as the harness uid."""
    if not ctx.confine_to or not root.is_dir():
        return
    for base, dirs, files in os.walk(root):
        for name in sorted(dirs + files):
            path = Path(base) / name
            ctx.path(str(path))
            if path.is_symlink():
                raise ToolError("policy_denied", "recursive copy/archive operations cannot include symbolic links "
                                "in the sandbox", hint="Copy the intended files explicitly within the workspace.")


def list_dir(ctx: ToolContext, path: str = ".", depth: int = 1) -> str:
    root = _existing(ctx, path)
    lines: list[str] = []

    def visit(folder: Path, level: int) -> None:
        for entry in sorted((p for p in folder.iterdir() if ctx.allowed(p)),
                            key=lambda p: (p.is_file(), p.name)):
            if entry.name in SKIP_DIRS:
                continue
            suffix = "/" if entry.is_dir() else f"  ({entry.stat().st_size} B)"
            lines.append("  " * level + entry.name + suffix)
            if entry.is_dir() and not entry.is_symlink() and level + 1 < depth:
                visit(entry, level + 1)

    visit(root, 0)
    return "\n".join(lines) or "(empty directory)"


def glob_files(ctx: ToolContext, pattern: str, path: str = ".") -> str:
    root = _existing(ctx, path)
    hits = sorted(ctx.rel(p) for p in root.glob(pattern) if not SKIP_DIRS & set(p.parts) and ctx.allowed(p))
    return "\n".join(hits[:500]) or "(no matches)"


def grep(ctx: ToolContext, pattern: str, path: str = ".", include: str = "*", ignore_case: bool = False,
         max_results: int = 50) -> str:
    rx = _regex(pattern, re.I if ignore_case else 0)
    root = _existing(ctx, path)
    files = [root] if root.is_file() else [p for p in _walk(root) if fnmatch.fnmatch(p.name, include)]
    hits: list[str] = []
    total = 0
    for file in files:
        if not ctx.allowed(file):
            continue
        try:
            for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    total += 1
                    if len(hits) < max_results:
                        hits.append(f"{ctx.rel(file)}:{number}: {line[:300]}")
        except (UnicodeDecodeError, OSError):
            continue
    if not hits:
        return "(no matches)"
    more = f"\n[{total} matches total, showing first {len(hits)}]" if total > len(hits) else ""
    return "\n".join(hits) + more


def file_info(ctx: ToolContext, path: str) -> str:
    p = _existing(ctx, path)
    st = p.stat()
    kind = "directory" if p.is_dir() else "file"
    mtime = dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return f"{path}: {kind}, {st.st_size} bytes, mode {oct(st.st_mode & 0o777)}, modified {mtime}"


def move_path(ctx: ToolContext, source: str, destination: str) -> str:
    src, dst = _existing(ctx, source), ctx.path(destination)
    if dst.is_dir():
        ctx.path(str(dst / src.name))
    _check_tree(ctx, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved {source} -> {destination}"


def copy_path(ctx: ToolContext, source: str, destination: str) -> str:
    src, dst = _existing(ctx, source), ctx.path(destination)
    if not src.is_dir() and dst.is_dir():
        ctx.path(str(dst / src.name))
    _check_tree(ctx, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    return f"Copied {source} -> {destination}"


def delete_path(ctx: ToolContext, path: str, recursive: bool = False) -> str:
    p = _existing(ctx, path)
    if p.resolve() in (ctx.workdir, Path.home(), Path("/")):
        raise ToolError("policy_denied", "refusing to delete the working directory, home or /")
    if p.is_dir():
        if not recursive and any(p.iterdir()):
            raise ToolError("invalid_arguments", f"{path} is a non-empty directory", hint="Pass recursive=true.")
        shutil.rmtree(p)
    else:
        p.unlink()
    return f"Deleted {path}"


def make_dir(ctx: ToolContext, path: str) -> str:
    ctx.path(path).mkdir(parents=True, exist_ok=True)
    return f"Directory ready: {path}"
