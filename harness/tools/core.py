"""Core tools: always present in the `tools` array, in this fixed order."""
from __future__ import annotations

import difflib
import re

from .base import Tool, ToolContext, ToolError, params
from .processes import run_process

MAX_BASH_TIMEOUT = 600

# Harness-level policy: commands that are never worth forwarding to a shell.
_DENY = [
    (re.compile(r"\brm\s+(-[a-zA-Z]+\s+)*-[a-zA-Z]*[rR][a-zA-Z]*\s+(-[a-zA-Z]+\s+)*(/|~|\$HOME)/?(\s|$)"),
     "recursive delete of / or the home directory"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "formatting a filesystem"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw write to a device"),
    (re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"(^|[;&|]\s*|\bsudo\s+)(shutdown|reboot|halt|poweroff)(\s|$)"), "shutting the machine down"),
]


def policy_violation(command: str) -> str | None:
    return next((why for pattern, why in _DENY if pattern.search(command)), None)


def bash(ctx: ToolContext, command: str, timeout: int | None = None) -> str:
    why = policy_violation(command)
    if why:
        raise ToolError("policy_denied", f"command blocked by harness policy ({why}).",
                        hint="This will not be allowed on retry; reach the goal another way.")
    limit = min(max(1, timeout or ctx.bash_timeout), MAX_BASH_TIMEOUT)
    return run_process(command, cwd=ctx.workdir, timeout=limit, shell=True)


def _existing(ctx: ToolContext, raw: str):
    path = ctx.path(raw)
    if not path.exists():
        siblings = [p.name for p in path.parent.iterdir()] if path.parent.is_dir() else []
        close = difflib.get_close_matches(path.name, siblings, n=3)
        hint = f"Did you mean: {', '.join(close)}?" if close else (
            f"Directory {ctx.rel(path.parent)} contains: {', '.join(sorted(siblings)[:20])}" if siblings
            else f"Directory {ctx.rel(path.parent)} does not exist.")
        raise ToolError("not_found", f"no such file: {raw}", hint=hint)
    return path


def read_text(ctx: ToolContext, raw: str) -> str:
    path = _existing(ctx, raw)
    if path.is_dir():
        raise ToolError("invalid_arguments", f"{raw} is a directory", hint="Use list_dir or bash `ls`.")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError("execution_error", f"{raw} is not UTF-8 text (binary file?)",
                        hint="Inspect it with bash: `file`, `xxd | head`.") from None


def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = 400) -> str:
    lines = read_text(ctx, path).splitlines()
    start = max(1, offset)
    chunk = lines[start - 1:start - 1 + max(1, limit)]
    body = "\n".join(f"{start + i:>5}\t{line}" for i, line in enumerate(chunk))
    end = start + len(chunk) - 1
    if not chunk:
        return f"[{path}: {len(lines)} lines; nothing at offset {start}]"
    if end < len(lines) or start > 1:
        body += f"\n[lines {start}-{end} of {len(lines)}; pass offset/limit for more]"
    return body


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    target = ctx.path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content.encode('utf-8'))} bytes ({content.count(chr(10)) + 1} lines) to {path}"


def edit_file(ctx: ToolContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    text = read_text(ctx, path)
    count = text.count(old_string)
    if old_string == new_string:
        raise ToolError("invalid_arguments", "old_string and new_string are identical")
    if count == 0:
        first = old_string.strip().splitlines()[0] if old_string.strip() else ""
        close = difflib.get_close_matches(first, [l.strip() for l in text.splitlines()], n=1, cutoff=0.6)
        raise ToolError("no_match", f"old_string not found in {path}",
                        hint=(f"Closest line in the file: {close[0]!r}. " if close else "")
                        + "Whitespace and indentation must match exactly; read the file first.")
    if count > 1 and not replace_all:
        raise ToolError("ambiguous_match", f"old_string occurs {count} times in {path}",
                        hint="Add surrounding lines to make it unique, or pass replace_all=true.")
    ctx.path(path).write_text(text.replace(old_string, new_string), encoding="utf-8")
    return f"Edited {path}: {count if replace_all else 1} replacement(s)"


CORE_TOOLS = [
    Tool("bash",
         "Execute a shell command in the working directory and return exit code, stdout and stderr. "
         "No stdin; commands are killed after `timeout` seconds (default 120, max 600).",
         params({"command": "string: The bash command to execute.",
                 "timeout?": "integer: Seconds before the command is killed."}), bash),
    Tool("read_file", "Read a UTF-8 text file with line numbers. Reads up to `limit` lines starting at line `offset`.",
         params({"path": "string: File path, relative to the working directory or absolute.",
                 "offset?": "integer: First line to read, 1-based (default 1).",
                 "limit?": "integer: Maximum number of lines (default 400)."}), read_file),
    Tool("write_file", "Create or overwrite a text file with the given content. Parent directories are created.",
         params({"path": "string: File path.", "content": "string: Full new content of the file."}), write_file),
    Tool("edit_file", "Replace an exact string in a file. Fails if old_string is missing or not unique.",
         params({"path": "string: File path.", "old_string": "string: Exact text to replace.",
                 "new_string": "string: Replacement text.",
                 "replace_all?": "boolean: Replace every occurrence (default false)."}), edit_file),
]
