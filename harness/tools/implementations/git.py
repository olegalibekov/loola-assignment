"""Git repository inspection and commits."""
from __future__ import annotations

from ..base import ToolContext
from ..processes import run_process


def _git(ctx: ToolContext, *args: str) -> str:
    return run_process(["git", "-c", "core.pager=cat", *args], cwd=ctx.workdir, timeout=60)


def git_status(ctx: ToolContext) -> str:
    return _git(ctx, "status", "--short", "--branch")


def git_diff(ctx: ToolContext, path: str = "", staged: bool = False) -> str:
    return _git(ctx, "diff", *(["--staged"] if staged else []), *(["--", path] if path else []))


def git_log(ctx: ToolContext, limit: int = 10) -> str:
    return _git(ctx, "log", f"-{max(1, limit)}", "--oneline", "--decorate")


def git_show(ctx: ToolContext, ref: str = "HEAD") -> str:
    return _git(ctx, "show", "--stat", ref)


def git_commit(ctx: ToolContext, message: str, add_all: bool = True) -> str:
    if add_all:
        added = _git(ctx, "add", "-A")
        if not added.startswith("Exit code: 0"):
            return added
    return _git(ctx, "commit", "-m", message)
