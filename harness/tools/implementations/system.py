"""HTTP requests, environment inspection, system utilities, and arithmetic."""
from __future__ import annotations

import ast
import datetime as dt
import operator
import os
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from ..base import ToolContext, ToolError
from ..core import _existing
from .filesystem import _walk


SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)


def http_request(ctx: ToolContext, url: str, method: str = "GET", body: str = "", timeout: int = 30) -> str:
    if not url.startswith(("http://", "https://")):
        raise ToolError("invalid_arguments", "url must start with http:// or https://")
    req = urllib.request.Request(url, data=body.encode() if body else None, method=method.upper(),
                                 headers={"User-Agent": "agent-harness"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return f"HTTP {resp.status}\n{resp.read(200_000).decode('utf-8', 'replace')}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}\n{exc.read(20_000).decode('utf-8', 'replace')}"
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ToolError("execution_error", f"request failed: {getattr(exc, 'reason', exc)}") from None


def which(ctx: ToolContext, command: str) -> str:
    return shutil.which(command) or f"{command}: not found on PATH"


def env_get(ctx: ToolContext, name: str) -> str:
    if SECRET_NAME.search(name):
        return f"{name}=<redacted by harness>" if name in os.environ else f"{name} is not set"
    return f"{name}={os.environ[name]}" if name in os.environ else f"{name} is not set"


def disk_usage(ctx: ToolContext, path: str = ".") -> str:
    root = _existing(ctx, path)
    sizes = Counter()
    for file in _walk(root) if root.is_dir() else [root]:
        if not ctx.allowed(file):
            continue
        rel = file.relative_to(root) if root.is_dir() else Path(file.name)
        sizes[rel.parts[0]] += file.stat().st_size
    lines = [f"{size:>12,} B  {name}" for name, size in sizes.most_common(30)]
    return f"total {sum(sizes.values()):,} B\n" + "\n".join(lines)


def current_time(ctx: ToolContext) -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calc(ctx: ToolContext, expression: str) -> str:
    def ev(node: ast.AST):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ToolError("invalid_arguments", "only numbers and + - * / // % ** ( ) are supported")

    try:
        return f"{expression} = {ev(ast.parse(expression, mode='eval'))}"
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ToolError("invalid_arguments", f"cannot evaluate: {exc}") from None
