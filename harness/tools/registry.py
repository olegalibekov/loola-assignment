"""Tool registry: what the model sees in `tools`, and the pipeline every call goes through.

Loading modes (Config.tool_mode):
  eager    - all schemas in `tools` from turn one. Simple, but every request pays for all of them.
  append   - schemas are appended to `tools` when `tool_search` finds them. Saves tokens, but chat
             templates render `tools` ahead of the conversation, so each load invalidates the cache.
  dispatch - `tools` is constant for the whole run. Schemas arrive as *tool results* (tail of the
             context) and catalog tools are invoked through `use_tool`. Nothing in the prefix moves.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..runtime.config import Config
from ..context.skills import SkillStore
from . import CATALOG_TOOLS, CORE_TOOLS, Tool, ToolContext, ToolError, params, validate_arguments

TOOL_SEARCH, USE_TOOL, LOAD_SKILL = "tool_search", "use_tool", "load_skill"


@dataclass
class Outcome:
    text: str
    ok: bool = True
    kind: str = "ok"  # error class when ok is False
    tool: str = ""  # tool that actually ran (after use_tool / auto-routing)
    intercepted: list[str] = field(default_factory=list)  # deterministic fixes the harness applied


def cap_output(text: str, limit: int, spill: Path | None) -> tuple[str, bool]:
    """Keep head and tail of an oversized output; the full text is spilled to a file."""
    if len(text) <= limit:
        return text, False
    head, tail = int(limit * 0.6), int(limit * 0.4)
    where = ""
    if spill is not None:
        spill.parent.mkdir(parents=True, exist_ok=True)
        spill.write_text(text, encoding="utf-8")
        where = f" Full output saved to {spill} - read parts of it with grep / sed -n 'A,Bp'."
    marker = f"\n[... {len(text) - head - tail} of {len(text)} chars omitted by the harness.{where}]\n"
    return text[:head] + marker + text[-tail:], True


class ToolRegistry:
    def __init__(self, cfg: Config, skills: SkillStore, core: list[Tool] | None = None,
                 catalog: list[Tool] | None = None):
        self.cfg = cfg
        self.skills = skills
        roots = tuple(p.resolve() for p in (cfg.workdir, cfg.run_dir) if p is not None) if cfg.confine_file_tools else ()
        self.ctx = ToolContext(workdir=cfg.workdir, bash_timeout=cfg.bash_timeout, confine_to=roots)
        if roots and os.geteuid() == 0:
            os.umask(0)  # sandbox: files written by in-process tools stay writable for the unprivileged tool user
        self.core = {t.name: t for t in (CORE_TOOLS if core is None else core)}
        cat = CATALOG_TOOLS if catalog is None else catalog
        self.catalog = {t.name: t for t in sorted(cat, key=lambda t: (t.category, t.name))}
        self.appended: list[str] = []  # append mode: catalog tools promoted into `tools`, in load order
        self.seen: list[str] = []  # catalog tools whose schema the model has been shown or has used
        self.meta = self._meta_tools()
        self._spill_seq = 0

    # ------------------------------------------------------------------ what the model sees

    def _meta_tools(self) -> dict[str, Tool]:
        if self.cfg.tool_mode == "eager":
            return {}
        meta = [Tool(TOOL_SEARCH,
                     "Find catalog tools by keywords or exact names and get their JSON schemas. "
                     + ("Found tools become directly callable." if self.cfg.tool_mode == "append"
                        else f"Call a found tool with `{USE_TOOL}`."),
                     params({"query": "string: Keywords or tool names, e.g. 'csv aggregate' or 'grep git_commit'.",
                             "limit?": "integer: Maximum tools to return (default 5)."}), self._tool_search, "meta")]
        if self.cfg.tool_mode == "dispatch":
            meta.append(Tool(USE_TOOL, f"Call a catalog tool by name with arguments matching its schema (see `{TOOL_SEARCH}`).",
                             params({"name": "string: Catalog tool name.",
                                     "arguments?": "object: Arguments for that tool."}), self._unreachable, "meta"))
        if self.skills.skills:
            meta.append(Tool(LOAD_SKILL, "Load a skill (a short playbook) into the conversation by name.",
                             params({"name": "string: Skill name from the skills list."}),
                             lambda ctx, name: self.skills.load(name), "meta"))
        return {t.name: t for t in meta}

    def native_tools(self) -> list[dict]:
        """The `tools` array. Order is fixed; in dispatch mode the content never changes."""
        tools = list(self.core.values()) + list(self.meta.values())
        if self.cfg.tool_mode == "eager":
            tools += list(self.catalog.values())
        elif self.cfg.tool_mode == "append":
            tools += [self.catalog[name] for name in self.appended]  # append-only: earlier entries keep their bytes
        return [t.schema() for t in tools]

    def catalog_index(self) -> str:
        """Compact signatures, grouped by category: ~10% of the size of the full schemas.

        Enough for the model to call most tools without a `tool_search` round trip; the full
        schema is one search away, and a validation error always carries it.
        """
        by_category: dict[str, list[str]] = {}
        for tool in self.catalog.values():
            by_category.setdefault(tool.category, []).append(tool.signature())
        return "\n".join(f"- {category}: {', '.join(names)}" for category, names in by_category.items())

    def pinned_state(self) -> str:
        """Harness-owned facts that must survive compaction (never left to the summariser)."""
        lines = []
        if self.skills.loaded:
            lines.append("- Skills loaded earlier (bodies were dropped; call load_skill again if you need one): "
                         + ", ".join(self.skills.loaded))
        if self.seen and self.cfg.tool_mode != "eager":
            lines.append("- Catalog tools already discovered: " + "; ".join(self.catalog[n].signature() for n in self.seen))
        if self.cfg.run_dir and self._spill_seq:
            lines.append(f"- Full copies of truncated tool outputs: {self.cfg.run_dir / 'outputs'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ meta tools

    def _unreachable(self, ctx, **_):  # use_tool is unwrapped in execute()
        raise AssertionError("use_tool is resolved by the registry")

    def search(self, query: str, limit: int = 5) -> list[Tool]:
        terms = [t for t in re.split(r"[^a-z0-9_]+", query.lower()) if t]
        exact = [self.catalog[t] for t in terms if t in self.catalog]
        scored = []
        for tool in self.catalog.values():
            if tool in exact:
                continue
            name_words = set(tool.name.split("_"))
            text = tool.description.lower()
            score = sum(4 * (t in name_words) + 3 * (t == tool.category) + 2 * (t in tool.name and t not in name_words)
                        + (t in text) for t in terms)
            if score:
                scored.append((-score, tool.name, tool))
        ranked = exact + [tool for _, _, tool in sorted(scored)]
        return ranked[:max(1, limit, len(exact))]

    def _tool_search(self, ctx, query: str, limit: int = 5) -> str:
        found = self.search(query, limit)
        if not found:
            return f"No catalog tool matches '{query}'. Catalog:\n{self.catalog_index()}"
        for tool in found:
            if tool.name not in self.seen:
                self.seen.append(tool.name)
        if self.cfg.tool_mode == "append":
            self.appended += [t.name for t in found if t.name not in self.appended]
            return "Loaded - call these directly:\n" + "\n".join(f"- {t.signature()}: {t.description}" for t in found)
        lines = [json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters},
                            ensure_ascii=False) for t in found]
        return f"Call any of these with {USE_TOOL}(name, arguments):\n" + "\n".join(lines)

    # ------------------------------------------------------------------ execution pipeline

    def execute(self, name: str, args: dict) -> Outcome:
        """resolve -> validate/coerce -> run -> cap. Never raises for anything a tool did."""
        outcome = Outcome("", tool=name)
        try:
            tool, args = self._resolve(name, args, outcome)
            outcome.tool = tool.name
            args, coerced = validate_arguments(tool, args)
            if coerced:
                outcome.intercepted.append("coerced_argument")
            text = tool.func(self.ctx, **args)
        except ToolError as exc:
            outcome.ok, outcome.kind, text = False, exc.kind, exc.render()
        except Exception as exc:  # a bug in a tool or an OS error: report it, keep the loop alive
            outcome.ok, outcome.kind = False, "execution_error"
            text = f"ERROR [execution_error]: {type(exc).__name__}: {exc}"
        if self.cfg.cap_tool_output:
            spill = None
            if self.cfg.run_dir is not None:
                self._spill_seq += 1
                # Unknown tool names come from the model and may contain paths or be enormous.
                label = re.sub(r"[^a-zA-Z0-9_-]", "_", outcome.tool or "tool")[:64]
                spill = self.cfg.run_dir / "outputs" / f"{self._spill_seq:04d}_{label}.txt"
            text, capped = cap_output(text, self.cfg.tool_output_chars, spill)
            if capped:
                outcome.intercepted.append("capped_output")
            elif spill is not None:
                self._spill_seq -= 1
        outcome.text = text if text.strip() else "(no output)"
        return outcome

    def _resolve(self, name: str, args: dict, outcome: Outcome) -> tuple[Tool, dict]:
        if name == USE_TOOL and USE_TOOL in self.meta:
            inner = args.get("name")
            if not isinstance(inner, str) or not inner:
                raise ToolError("invalid_arguments", f"{USE_TOOL} needs a string `name`",
                                hint=f'Example: {USE_TOOL}({{"name": "grep", "arguments": {{"pattern": "TODO"}}}})')
            inner_args = args.get("arguments") or {}
            if isinstance(inner_args, str):
                try:
                    inner_args = json.loads(inner_args)
                    outcome.intercepted.append("coerced_argument")
                except json.JSONDecodeError:
                    inner_args = None
            if not isinstance(inner_args, dict):
                raise ToolError("invalid_arguments", f"{USE_TOOL}.arguments must be a JSON object")
            extra = {k: v for k, v in args.items() if k not in ("name", "arguments")}
            if extra and not inner_args:  # arguments passed flat next to `name`: unambiguous, accept
                inner_args = extra
                outcome.intercepted.append("flattened_arguments")
            name, args = inner, inner_args
            if name in self.core or name in self.meta:
                outcome.intercepted.append("auto_routed")
        if name in self.core:
            return self.core[name], args
        if name in self.meta:
            return self.meta[name], args
        if name in self.catalog:
            callable_now = self.cfg.tool_mode == "eager" or name in self.appended or outcome.tool == USE_TOOL
            if not callable_now:
                # The model called a real catalog tool natively without loading it. The intent is
                # unambiguous, so run it instead of burning a turn on an "unknown tool" error.
                outcome.intercepted.append("auto_routed")
            if name not in self.seen:
                self.seen.append(name)
            return self.catalog[name], args
        known = list(self.core) + list(self.meta) + list(self.catalog)
        close = difflib.get_close_matches(name, known, n=3, cutoff=0.5)
        raise ToolError("unknown_tool", f"no tool named '{name}'",
                        hint=(f"Did you mean: {', '.join(close)}? " if close else "")
                        + (f"Use {TOOL_SEARCH} to look up catalog tools." if TOOL_SEARCH in self.meta else ""))
