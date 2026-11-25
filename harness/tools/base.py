"""Shared tool primitives: definitions, execution context, errors, and schema DSL."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class ToolError(Exception):
    """A failure the model can act on. `kind` is the error class shown to the model."""

    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind, self.message, self.hint = kind, message, hint

    def render(self) -> str:
        text = f"ERROR [{self.kind}]: {self.message}"
        return f"{text}\nHint: {self.hint}" if self.hint else text


@dataclass
class ToolContext:
    workdir: Path
    bash_timeout: int = 120
    python: str = sys.executable
    # Sandbox mode: in-process file tools may only touch these roots. They run inside the harness
    # process (which holds the API key), so without this `read_file /proc/self/environ` would leak it.
    confine_to: tuple[Path, ...] = ()

    def path(self, raw: str) -> Path:
        p = Path(raw).expanduser()
        p = p if p.is_absolute() else self.workdir / p
        if self.confine_to:
            real = p.resolve()  # follows symlinks, so a link out of the workspace is caught too
            if not any(real == root or root in real.parents for root in self.confine_to):
                raise ToolError("policy_denied", f"{raw} is outside the workspace; file tools are confined to it.",
                                hint="Work with files under the working directory.")
        return p

    def allowed(self, p: Path) -> bool:
        real = p.resolve()
        return not self.confine_to or any(real == root or root in real.parents for root in self.confine_to)

    def rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(self.workdir))
        except ValueError:
            return str(p)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str] = field(repr=False, compare=False)
    category: str = "core"

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def signature(self) -> str:
        required = set(self.parameters.get("required", []))
        args = [name if name in required else f"{name}?" for name in self.parameters.get("properties", {})]
        return f"{self.name}({', '.join(args)})"


def params(spec: dict[str, str] | None = None) -> dict:
    """Schema DSL: {"path": "string: File path.", "limit?": "integer: Max lines."}.

    A trailing "?" marks an optional argument. "enum[a|b]" declares a string enum.
    """
    properties: dict = {}
    required: list[str] = []
    for key, desc in (spec or {}).items():
        name = key.rstrip("?")
        typ, _, text = desc.partition(":")
        typ = typ.strip()
        prop: dict
        if typ.startswith("enum["):
            prop = {"type": "string", "enum": typ[5:-1].split("|")}
        elif typ == "array":
            prop = {"type": "array", "items": {"type": "string"}}
        else:
            prop = {"type": typ}
        prop["description"] = text.strip()
        properties[name] = prop
        if not key.endswith("?"):
            required.append(name)
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
