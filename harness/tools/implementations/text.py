"""Text inspection, comparison, replacement, and hashing."""
from __future__ import annotations

import difflib
import hashlib
import re

from ..base import ToolContext, ToolError
from ..core import _existing, read_text


def _regex(pattern: str, flags: int = 0) -> re.Pattern:
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError("invalid_arguments", f"invalid regular expression: {exc}") from None


def head_tail(ctx: ToolContext, path: str, head: int = 20, tail: int = 20) -> str:
    lines = read_text(ctx, path).splitlines()
    if len(lines) <= head + tail:
        return "\n".join(lines)
    return "\n".join(lines[:head] + [f"[... {len(lines) - head - tail} lines omitted ...]"] + lines[-tail:])


def count_lines(ctx: ToolContext, path: str) -> str:
    text = read_text(ctx, path)
    return f"{path}: {len(text.splitlines())} lines, {len(text.split())} words, {len(text)} chars"


def diff_files(ctx: ToolContext, a: str, b: str) -> str:
    diff = difflib.unified_diff(read_text(ctx, a).splitlines(), read_text(ctx, b).splitlines(), a, b, lineterm="")
    return "\n".join(diff) or "(files are identical)"


def regex_replace(ctx: ToolContext, path: str, pattern: str, replacement: str, count: int = 0) -> str:
    text = read_text(ctx, path)
    new, n = _regex(pattern, re.M).subn(replacement, text, count=count)
    if n:
        ctx.path(path).write_text(new, encoding="utf-8")
    return f"{path}: {n} replacement(s)"


def hash_file(ctx: ToolContext, path: str, algorithm: str = "sha256") -> str:
    return f"{algorithm}:{hashlib.new(algorithm, _existing(ctx, path).read_bytes()).hexdigest()}  {path}"
