"""Conversation state and cache-aware compaction.

Invariants (they are what keeps the provider's prefix cache valid):
  * between compactions the message list is append-only - nothing already sent is edited,
    reordered or dropped;
  * messages[0] (system prompt) never changes, and the task text stays the leading bytes of
    messages[1] for the whole run;
  * compaction is a rare, single event with hysteresis (trigger high, land low), so the cache
    is rebuilt once per compaction instead of once per turn;
  * which level runs is decided before anything is rewritten, so a summary request is always the
    untouched history plus one message (a cache hit);
  * a tool call and its results are never separated.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from ..runtime.config import Config
from ..llm.client import serialize_prompt

SUMMARY_INSTRUCTION = """\
[harness] The context window is almost full, so the messages above are about to be replaced by a summary. \
Do not call tools in this reply. Write the handoff summary that lets you continue the task with nothing \
else to rely on. Use exactly these sections:

## Done
Completed steps and their outcome (reference the task's own step numbers if it has them).
## Files
Every file created or modified, with a one-line note on its current state.
## Key facts
Exact values that will be needed later, copied verbatim: identifiers, numbers, names, paths, commands that worked.
## Problems
Errors hit and how they were resolved; approaches that failed and must not be retried.
## Unverified
Task requirements that are not done yet, or are done but have not been checked against the task text with a tool. One per line. Write `none` only if every requirement has been checked.
## Next
What remains, in order, starting with the very next action.

Be specific and terse (under 500 words). If an earlier <context_summary> exists above, merge it in - \
nothing from it may be lost. No preamble."""

SUMMARY_BLOCK = """\
<context_summary number="{n}">
Earlier messages of this conversation were compacted. This summary replaces them.

{summary}

Harness state (authoritative):
{pinned}
Continue from "Next". Do not redo completed steps; if unsure about the state of a file, check it with a tool. Before the final answer, clear every item under "Unverified" by checking it with a tool.
</context_summary>"""

ELIDE_MIN_CHARS = 500
ELIDED = "[harness: output elided to save context ({n} chars). First line: {first}{last}]"
ELIDED_LINE = 100  # chars of the first / last line kept in a stub


def elide_outputs(messages: list[dict]) -> tuple[list[dict], int]:
    """Level 1 compaction: replace large tool outputs by a deterministic stub. Returns (copy, count).

    Assistant messages are left byte-identical (the model still sees every action it took, and
    provider signatures attached to tool calls stay valid). Skill bodies are instructions, not
    observations, so `load_skill` results are kept.
    """
    keep = {call.get("id") for m in messages for call in m.get("tool_calls") or []
            if (call.get("function") or {}).get("name") == "load_skill"}
    out, count = [], 0
    for message in messages:
        content = message.get("content")
        if (message.get("role") == "tool" and isinstance(content, str) and len(content) > ELIDE_MIN_CHARS
                and message.get("tool_call_id") not in keep):
            lines = [line for line in content.splitlines() if line.strip()] or [""]
            last = f" | Last line: {lines[-1][:ELIDED_LINE]}" if len(lines) > 1 else ""
            message = {**message, "content": ELIDED.format(n=len(content), first=lines[0][:ELIDED_LINE], last=last)}
            count += 1
        out.append(message)
    return out, count


def section(text: str, title: str) -> str:
    match = re.search(rf"^##\s*{re.escape(title)}\s*\n(.*?)(?=^##\s|\Z)", text, re.S | re.M)
    return match.group(1).strip() if match else ""


def mechanical_summary(messages: list[dict], limit: int = 40) -> str:
    """Model-free fallback: a log of the calls made and how each one ended."""
    results = {m.get("tool_call_id"): m.get("content") or "" for m in messages if m.get("role") == "tool"}
    lines = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            first = (results.get(call.get("id"), "").strip().splitlines() or [""])[0]
            lines.append(f"- {fn.get('name')}({str(fn.get('arguments'))[:120]}) -> {first[:120]}")
    omitted = f"({len(lines) - limit} earlier calls omitted)\n" if len(lines) > limit else ""
    return ("## Done\n(The summariser was unavailable; this is the raw call log.)\n" + omitted
            + "\n".join(lines[-limit:]) + "\n## Next\nRe-check the current state with tools, then continue the task.")


class Context:
    def __init__(self, system_prompt: str, task: str, cfg: Config):
        self.cfg = cfg
        self.task = task
        self.messages: list[dict] = [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": task}]
        self.compactions = 0  # summaries written so far
        self.unverified = ""  # last "Unverified" section, carried over if a later summary drops it
        self.chars_per_token = 3.6  # replaced by the server's real ratio after the first response

    def append(self, message: dict) -> None:
        self.messages.append(message)

    # ------------------------------------------------------------------ size accounting

    def calibrate(self, prompt_chars: int, prompt_tokens: int) -> None:
        """Learn chars-per-token from real usage, so estimates track this model's tokenizer."""
        if prompt_tokens > 0 and prompt_chars > 0:
            self.chars_per_token = min(8.0, max(1.5, prompt_chars / prompt_tokens))

    def tokens(self, tools: list[dict], messages: list[dict] | None = None) -> int:
        text = serialize_prompt(tools, self.messages if messages is None else messages)
        return int(len(text) / self.chars_per_token) + 1

    def _message_tokens(self, messages: list[dict]) -> int:
        return int(len(json.dumps(messages, ensure_ascii=False)) / self.chars_per_token)

    def needs_compaction(self, tools: list[dict]) -> bool:
        return self.tokens(tools) > self.cfg.compact_trigger * self.cfg.usable_window

    def overflows(self, tools: list[dict]) -> bool:
        return self.tokens(tools) > self.cfg.usable_window

    # ------------------------------------------------------------------ compaction

    def cut_index(self, keep_nothing: bool = False) -> int | None:
        """First message of the tail that stays verbatim. Always on a turn boundary."""
        if keep_nothing:
            return len(self.messages) if len(self.messages) > 2 else None
        budget = self.cfg.compact_keep_tail * self.cfg.usable_window
        boundaries = [i for i in range(3, len(self.messages)) if self.messages[i]["role"] != "tool"]
        if not boundaries:
            return None
        for index in boundaries:
            if self._message_tokens(self.messages[index:]) <= budget:
                return index
        return boundaries[-1]  # even the last turn is over budget: keep just that one

    def compact(self, summarize: Callable[[list[dict]], str], pinned: str, tools: list[dict],
                keep_nothing: bool = False) -> dict | None:
        """Shrink messages[2:cut]. Returns an event dict, or None if there is nothing to cut.

        Level 1 (no model call): stub old large tool outputs. Used when that alone brings the prompt
        under `compact_target`. Level 2: replace the whole zone with a model-written summary folded
        into the task message. `summarize` receives the head *as it was sent before* plus one
        instruction, so the summary request itself is a cache hit.
        """
        cut = self.cut_index(keep_nothing)
        if cut is None:
            return None
        head, tail = self.messages[:cut], self.messages[cut:]

        if self.cfg.compact_elide and not keep_nothing:
            zone, count = elide_outputs(head[2:])
            candidate = head[:2] + zone + tail
            if count and self.tokens(tools, candidate) <= self.cfg.compact_target * self.cfg.usable_window:
                self.messages = candidate
                return {"level": "elide", "elided_messages": count, "dropped_messages": 0,
                        "kept_messages": len(tail), "fallback": False}

        dropped_tokens = self._message_tokens(head[2:])
        summary = summarize(head + [{"role": "user", "content": SUMMARY_INSTRUCTION}]).strip()
        fallback = not summary
        if fallback:
            summary = mechanical_summary(head)
        # Open items must never be lost silently: if this summary has no "Unverified" section but an
        # earlier one did, the harness carries the earlier one over verbatim.
        open_items = section(summary, "Unverified")
        if open_items:
            self.unverified = open_items
        elif self.unverified:
            summary += f"\n\n## Unverified\n(carried over by the harness from the previous summary)\n{self.unverified}"
        self.compactions += 1
        block = SUMMARY_BLOCK.format(n=self.compactions, summary=summary,
                                     pinned=(pinned + "\n") if pinned else "- (nothing to restore)\n")
        # One user message (task + summary) keeps strict user/assistant alternation for picky
        # chat templates, and the task text remains a cached prefix.
        self.messages = [self.messages[0], {"role": "user", "content": f"{self.task}\n\n{block}"}] + tail
        return {"level": "summary", "number": self.compactions, "dropped_messages": len(head) - 2,
                "kept_messages": len(tail), "dropped_tokens_est": dropped_tokens, "summary_chars": len(summary),
                "fallback": fallback}
