"""Run metrics, JSONL trace, and provider-independent prefix-cache accounting."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.client import LLMResponse


class PrefixTracker:
    """How much of each request is a byte-for-byte prefix of the previous one.

    A KV-cache (llama.cpp slots, vLLM prefix caching, provider prompt caching) can reuse at
    most that shared prefix, so this is an upper bound on cache hits that does not depend on
    the provider reporting `cached_tokens`.
    """

    def __init__(self) -> None:
        self.previous = ""
        self.total = 0
        self.reused = 0

    def observe(self, prompt: str) -> tuple[int, int]:
        shared = len(os.path.commonprefix([self.previous, prompt]))
        self.previous = prompt
        self.total += len(prompt)
        self.reused += shared
        return shared, len(prompt)


@dataclass
class Metrics:
    status: str = "running"
    error: str = ""
    turns: int = 0
    requests: int = 0
    api_attempts: int = 0  # includes failed HTTP attempts, retries, and summary requests
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # as reported by the server (0 if it does not report)
    peak_prompt_tokens: int = 0
    prefix_chars_total: int = 0
    prefix_chars_reused: int = 0
    reusable_prompt_tokens: int = 0  # prompt tokens covered by the shared prefix (upper bound on cache hits)
    tool_calls: int = 0
    tool_usage: Counter = field(default_factory=Counter)  # calls by the tool that actually ran
    tool_errors: Counter = field(default_factory=Counter)  # errors returned to the model, by class
    intercepted: Counter = field(default_factory=Counter)  # problems fixed by the harness, by class
    llm_retries: int = 0
    compactions: list = field(default_factory=list)
    wall_seconds: float = 0.0
    started: float = field(default_factory=time.monotonic, repr=False)

    def record_request(self, resp: LLMResponse, shared: int, total: int) -> None:
        usage = resp.usage
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cached_tokens += usage.cached_tokens
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, usage.prompt_tokens)
        self.prefix_chars_total += total
        self.prefix_chars_reused += shared
        if total:
            self.reusable_prompt_tokens += round(usage.prompt_tokens * shared / total)
        self.llm_retries += resp.retries
        if resp.retries:
            self.intercepted["llm_retry"] += resp.retries

    def to_dict(self) -> dict:
        self.wall_seconds = round(time.monotonic() - self.started, 2)
        data = {k: v for k, v in self.__dict__.items() if k != "started"}
        data["tool_usage"] = dict(self.tool_usage)
        data["tool_errors"] = dict(self.tool_errors)
        data["intercepted"] = dict(self.intercepted)
        data["total_tokens"] = self.prompt_tokens + self.completion_tokens
        data["prefix_reuse"] = round(self.prefix_chars_reused / self.prefix_chars_total, 4) if self.prefix_chars_total else 0.0
        data["uncached_prompt_tokens_est"] = self.prompt_tokens - self.reusable_prompt_tokens
        return data


class Trace:
    """Append-only JSONL event log (one file per run). A no-op without a run directory."""

    def __init__(self, run_dir: Path | None):
        self.path = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.path = run_dir / "trace.jsonl"
            self.path.write_text("", encoding="utf-8")

    def event(self, name: str, /, **data) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "event": name, **data}, ensure_ascii=False) + "\n")
