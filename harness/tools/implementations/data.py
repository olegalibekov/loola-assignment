"""CSV, JSON, SQLite, and structured log analysis."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics
from collections import Counter

from ..base import ToolContext, ToolError
from ..core import _existing, read_text


def _csv_rows(ctx: ToolContext, path: str) -> tuple[list[str], list[dict]]:
    with _existing(ctx, path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def csv_summary(ctx: ToolContext, path: str) -> str:
    columns, rows = _csv_rows(ctx, path)
    out = [f"{path}: {len(rows)} rows, {len(columns)} columns"]
    for col in columns:
        values = [(r.get(col) or "").strip() for r in rows]
        filled = [v for v in values if v]
        numbers = [n for n in map(_num, filled) if n is not None]
        if filled and len(numbers) == len(filled):
            out.append(f"- {col}: numeric, min={min(numbers):g} max={max(numbers):g} "
                       f"mean={statistics.fmean(numbers):.4g}, {len(values) - len(filled)} empty")
        else:
            top = ", ".join(f"{v!r}x{c}" for v, c in Counter(filled).most_common(5))
            out.append(f"- {col}: {len(set(filled))} distinct, {len(values) - len(filled)} empty; top: {top}")
    return "\n".join(out)


def csv_aggregate(ctx: ToolContext, path: str, group_by: str, value: str = "", agg: str = "count") -> str:
    columns, rows = _csv_rows(ctx, path)
    for col in filter(None, (group_by, value)):
        if col not in columns:
            raise ToolError("invalid_arguments", f"no column '{col}' in {path}", hint=f"Columns: {', '.join(columns)}")
    if agg != "count" and not value:
        raise ToolError("invalid_arguments", f"agg='{agg}' needs a `value` column")
    groups: dict[str, list[float]] = {}
    for row in rows:
        number = _num(row.get(value, "")) if value else 1.0
        if number is not None:
            groups.setdefault(row[group_by], []).append(number)
    fn = {"count": len, "sum": sum, "mean": statistics.fmean, "min": min, "max": max}[agg]
    lines = [f"{key!r}: {fn(vals):g}" for key, vals in sorted(groups.items())]
    return f"{agg}({value or '*'}) by {group_by} (raw values, no normalisation):\n" + "\n".join(lines)


def json_query(ctx: ToolContext, path: str, pointer: str = "") -> str:
    try:
        node = json.loads(read_text(ctx, path))
    except json.JSONDecodeError as exc:
        raise ToolError("execution_error", f"{path} is not valid JSON: {exc}") from None
    for part in filter(None, pointer.split(".")):
        try:
            node = node[int(part)] if isinstance(node, list) else node[part]
        except (KeyError, IndexError, ValueError, TypeError):
            keys = list(node)[:20] if isinstance(node, dict) else f"list of {len(node)}" if isinstance(node, list) else type(node).__name__
            raise ToolError("not_found", f"'{part}' not found while resolving '{pointer}'", hint=f"Available here: {keys}") from None
    if isinstance(node, list) and len(node) > 20:
        return f"list of {len(node)} items; first 3:\n" + json.dumps(node[:3], indent=1, ensure_ascii=False)
    return json.dumps(node, indent=1, ensure_ascii=False)


def sqlite_query(ctx: ToolContext, database: str, sql: str) -> str:
    conn = sqlite3.connect(ctx.path(database))
    denied = False

    def authorize(action, arg1, arg2, db_name, trigger):
        nonlocal denied
        # ATTACH also implements VACUUM INTO. Its filename can be a SQL expression, so merely
        # validating the `database` argument cannot confine these secondary file accesses.
        if action == sqlite3.SQLITE_ATTACH and arg1 != "":
            denied = True
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    if ctx.confine_to:
        conn.set_authorizer(authorize)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(101)
        conn.commit()
    except sqlite3.Error as exc:
        if denied:
            raise ToolError("policy_denied", "ATTACH and VACUUM INTO are disabled in the sandbox",
                            hint="Query the intended workspace database directly with the database argument.") from None
        raise ToolError("execution_error", f"SQLite error: {exc}") from None
    finally:
        conn.close()
    if cur.description is None:
        return f"OK, {cur.rowcount} row(s) affected"
    header = " | ".join(d[0] for d in cur.description)
    body = "\n".join(" | ".join(str(v) for v in row) for row in rows[:100])
    return f"{header}\n{body}" + ("\n[more than 100 rows; add LIMIT]" if len(rows) > 100 else "")


_LEVEL = re.compile(r"\b(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b")


_SEVERITY = ["CRITICAL", "FATAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "TRACE"]


def log_levels(ctx: ToolContext, path: str, show: int = 5) -> str:
    lines = read_text(ctx, path).splitlines()
    by_level: dict[str, list[tuple[int, str]]] = {}
    for number, line in enumerate(lines, 1):
        match = _LEVEL.search(line[:120])
        if match:
            by_level.setdefault(match.group(1), []).append((number, line))
    out = [f"{path}: {len(lines)} lines"]
    out += [f"- {level}: {len(by_level[level])}" for level in _SEVERITY if level in by_level]
    top = next((level for level in _SEVERITY if level in by_level), None)
    if top:
        out.append(f"First {min(show, len(by_level[top]))} {top} line(s):")
        out += [f"{number}: {line[:400]}" for number, line in by_level[top][:show]]
    return "\n".join(out)
