"""Skills: markdown playbooks in skills/<name>/SKILL.md, loaded into context on demand.

Only the one-line index lives in the (static) system prompt. A body enters the context
as a tool result, i.e. at the tail of the conversation, so loading never touches the
cached prefix.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from ..tools.base import ToolError


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def _parse(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        header, _, body = text[3:].partition("\n---")
        for line in header.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                meta[key.strip()] = value.strip()
    return Skill(meta.get("name", path.parent.name), meta.get("description", ""), body.strip())


class SkillStore:
    def __init__(self, directory: Path | None):
        files = sorted(directory.glob("*/SKILL.md")) if directory and directory.is_dir() else []
        self.skills = {s.name: s for s in map(_parse, files)}  # sorted => stable prompt bytes
        self.loaded: list[str] = []  # every skill ever loaded, in order
        self.in_context: set[str] = set()  # bodies still visible to the model

    def index(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())

    def all_bodies(self) -> str:
        return "\n\n".join(f"### Skill: {s.name}\n{s.body}" for s in self.skills.values())

    def load(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            close = difflib.get_close_matches(name, list(self.skills), n=3, cutoff=0.4)
            raise ToolError("unknown_skill", f"no skill named '{name}'",
                            hint=f"Did you mean: {', '.join(close)}?" if close else f"Available: {', '.join(self.skills)}")
        if name in self.in_context:
            return f"Skill '{name}' is already loaded earlier in this conversation; follow those instructions."
        if name not in self.loaded:
            self.loaded.append(name)
        self.in_context.add(name)
        return f"# Skill: {skill.name}\n{skill.body}"

    def on_compaction(self) -> None:
        """Bodies were summarised away; allow reloading."""
        self.in_context.clear()
