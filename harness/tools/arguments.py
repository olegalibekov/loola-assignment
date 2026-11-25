"""Parse, repair, and validate tool-call arguments against tool schemas."""
from __future__ import annotations

import json
import re

from .base import Tool, ToolError


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_arguments(raw) -> tuple[dict | None, str | None, bool]:
    """Parse tool-call arguments. Returns (args, error, repaired).

    Cheap, unambiguous repairs are done here instead of spending a model turn:
    dict passed as-is, empty string -> {}, markdown fences stripped, double-encoded JSON.
    Anything else is reported back to the model with the parser position.
    """
    if isinstance(raw, dict):
        return raw, None, False
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None, raw is not None and raw != ""
    if not isinstance(raw, str):
        return None, f"arguments must be a JSON object, got {type(raw).__name__}", False
    repaired = False
    text = raw
    for _ in range(2):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            stripped = _FENCE.sub("", text.strip())
            if stripped != text:
                text, repaired = stripped, True
                continue
            return None, f"arguments are not valid JSON: {exc.msg} at char {exc.pos}", False
        if isinstance(value, str):  # double-encoded
            text, repaired = value, True
            continue
        if isinstance(value, dict):
            return value, None, repaired
        return None, f"arguments must be a JSON object, got {type(value).__name__}", False
    return None, "arguments are not valid JSON", False


_TRUE, _FALSE = {"true", "1", "yes"}, {"false", "0", "no"}


def _coerce(value, prop: dict):
    """Return (ok, value, coerced). Lossless type fixes only ("5" -> 5, "true" -> True)."""
    typ = prop.get("type")
    coerced = False
    if typ == "string":
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return False, value, False
        if not isinstance(value, str):
            value, coerced = str(value), True
    elif typ == "integer":
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            value, coerced = int(value), True
        elif isinstance(value, float) and value.is_integer():
            value, coerced = int(value), True
        if isinstance(value, bool) or not isinstance(value, int):
            return False, value, False
    elif typ == "number":
        if isinstance(value, str):
            try:
                value, coerced = float(value), True
            except ValueError:
                return False, value, False
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, value, False
    elif typ == "boolean":
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            value, coerced = value.strip().lower() in _TRUE, True
        if not isinstance(value, bool):
            return False, value, False
    elif typ == "array":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            value, coerced = (parsed if isinstance(parsed, list) else [value]), True
        if not isinstance(value, list):
            return False, value, False
    elif typ == "object":
        if isinstance(value, str):
            try:
                value, coerced = json.loads(value), True
            except json.JSONDecodeError:
                return False, value, False
        if not isinstance(value, dict):
            return False, value, False
    if "enum" in prop and value not in prop["enum"]:
        return False, value, False
    return True, value, coerced


def validate_arguments(tool: Tool, args: dict) -> tuple[dict, int]:
    """Validate against the tool schema. Returns (clean_args, coerced_count) or raises ToolError."""
    props = tool.parameters.get("properties", {})
    problems: list[str] = []
    clean: dict = {}
    coerced = 0
    for name in tool.parameters.get("required", []):
        if name not in args or args[name] is None:
            problems.append(f"missing required argument '{name}'")
    for name, value in args.items():
        if name not in props:
            problems.append(f"unexpected argument '{name}'")
            continue
        if value is None:
            continue  # explicit null for an optional argument == omitted
        ok, value, did = _coerce(value, props[name])
        if not ok:
            expected = "one of " + ", ".join(props[name]["enum"]) if "enum" in props[name] else props[name]["type"]
            problems.append(f"argument '{name}' must be {expected}, got {json.dumps(args[name], ensure_ascii=False)[:80]}")
            continue
        coerced += did
        clean[name] = value
    if problems:
        raise ToolError("invalid_arguments", f"{tool.name}: " + "; ".join(problems),
                        hint=f"Schema: {json.dumps(tool.parameters, ensure_ascii=False)}")
    return clean, coerced
