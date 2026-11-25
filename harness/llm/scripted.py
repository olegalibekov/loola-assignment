"""A scripted stand-in for the LLM client: deterministic replies, real token accounting by size.

Used by the unit tests and by the offline replay benchmark (bench/replay.py). It lets the
harness mechanics - prompt size, prefix reuse, compaction - be measured without a model.
"""
from __future__ import annotations

import copy
import json
from typing import Callable, Iterable

from .client import LLMResponse, Usage, serialize_prompt

CHARS_PER_TOKEN = 4.0


def say(text: str, finish: str = "stop") -> LLMResponse:
    return LLMResponse(content=text, finish_reason=finish)


def call(name: str, /, _text: str = "", _id: str | None = None, **arguments) -> LLMResponse:
    return calls([(name, arguments)], text=_text, ids=[_id] if _id else None)


def calls(items: list[tuple[str, dict | str]], text: str = "", ids: list[str] | None = None,
          finish: str = "tool_calls") -> LLMResponse:
    tool_calls = []
    for index, (name, arguments) in enumerate(items):
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        tool_calls.append({"id": ids[index] if ids else None, "type": "function",
                           "function": {"name": name, "arguments": raw}})
    return LLMResponse(content=text, tool_calls=tool_calls, finish_reason=finish)


class ScriptedLLM:
    def __init__(self, script: Iterable[LLMResponse] | Callable[[list[dict], list[dict]], LLMResponse],
                 summary: str = "## Done\nwork so far\n## Next\ncontinue"):
        self._next = script if callable(script) else iter(script).__next__
        self._callable = callable(script)
        self.summary = summary
        self.requests: list[dict] = []
        self._counter = 0

    def chat(self, messages: list[dict], tools: list[dict], *, tool_choice: str = "auto",
             max_tokens: int | None = None) -> LLMResponse:
        self.requests.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools),
                              "tool_choice": tool_choice})
        if tool_choice == "none":
            resp = LLMResponse(content=self.summary)
        else:
            resp = copy.deepcopy(self._next(messages, tools) if self._callable else self._next())
        for tool_call in resp.tool_calls:
            if not tool_call.get("id"):
                self._counter += 1
                tool_call["id"] = f"call_{self._counter}"
        out = resp.content + "".join(str(c["function"]["arguments"]) for c in resp.tool_calls)
        resp.usage = Usage(prompt_tokens=int(len(serialize_prompt(tools, messages)) / CHARS_PER_TOKEN),
                           completion_tokens=int(len(out) / CHARS_PER_TOKEN) + 1)
        return resp
