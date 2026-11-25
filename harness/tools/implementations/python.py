"""Python execution, tests, and source inspection."""
from __future__ import annotations

import ast
import os
import tempfile

from ..base import ToolContext, ToolError
from ..core import read_text
from ..processes import run_process


def run_pytest(ctx: ToolContext, target: str = "", keyword: str = "", maxfail: int = 0, verbose: bool = False) -> str:
    cmd = [ctx.python, "-m", "pytest", "-p", "no:cacheprovider", "-q" if not verbose else "-v", "--tb=short"]
    if keyword:
        cmd += ["-k", keyword]
    if maxfail:
        cmd += [f"--maxfail={maxfail}"]
    if target:
        cmd += target.split()
    return run_process(cmd, cwd=ctx.workdir, timeout=ctx.bash_timeout)


def run_python(ctx: ToolContext, code: str, timeout: int = 60) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
    os.chmod(fh.name, 0o644)  # mkstemp makes it 0600; in the sandbox the interpreter runs as another user
    try:
        return run_process([ctx.python, fh.name], cwd=ctx.workdir, timeout=min(max(1, timeout), 600))
    finally:
        os.unlink(fh.name)


def py_syntax_check(ctx: ToolContext, path: str) -> str:
    try:
        ast.parse(read_text(ctx, path), filename=path)
    except SyntaxError as exc:
        return f"SyntaxError in {path}:{exc.lineno}:{exc.offset}: {exc.msg}\n    {(exc.text or '').rstrip()}"
    return f"{path}: syntax OK"


def py_outline(ctx: ToolContext, path: str) -> str:
    try:
        tree = ast.parse(read_text(ctx, path), filename=path)
    except SyntaxError as exc:
        raise ToolError("execution_error", f"cannot parse {path}: line {exc.lineno}: {exc.msg}") from None
    lines: list[str] = []

    def visit(node: ast.AST, indent: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(f"{'  ' * indent}{child.lineno:>4}: def {child.name}({ast.unparse(child.args)})")
            elif isinstance(child, ast.ClassDef):
                lines.append(f"{'  ' * indent}{child.lineno:>4}: class {child.name}")
                visit(child, indent + 1)

    visit(tree, 0)
    return "\n".join(lines) or "(no functions or classes)"
