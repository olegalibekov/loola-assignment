"""OpenAI-compatible chat client.

Transport problems never reach the model: they are retried here (429 / 5xx / network)
or surfaced as a typed exception the loop knows how to handle (overflow, fatal).
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import requests

from ..runtime.config import Config

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
OVERFLOW_MARKERS = (
    "context length", "context_length", "maximum context", "context window", "too many tokens",
    "prompt is too long", "exceeds the available context", "exceed_context", "reduce the length",
    "max_model_len", "maximum input length",
)


class LLMError(Exception):
    """Retries exhausted or an unusable response."""


class FatalLLMError(LLMError):
    """Auth / config / bad request: retrying or asking the model cannot fix it."""


class ContextOverflowError(LLMError):
    """The prompt does not fit the model window. The loop answers with compaction."""


class BudgetExceeded(LLMError):
    def __init__(self, status: str):
        self.status = status
        super().__init__("API attempt budget exhausted" if status == "request_limit" else "run time budget exhausted")


class RunBudget:
    """Shared across normal requests, summaries and transport retries.

    The deadline is cooperative: socket timeouts bound inactivity, not total download time.
    Blocking tool work must enforce its own timeout; results are checked before further work.
    """
    def __init__(self, max_attempts: int = 0, max_seconds: float = 0.0, clock=None):
        self.max_attempts, self.max_seconds = max_attempts, max_seconds
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.attempts = 0

    def remaining(self) -> float | None:
        if not self.max_seconds:
            return None
        return max(0.0, self.max_seconds - (self.clock() - self.started))

    def check_time(self) -> None:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise BudgetExceeded("time_limit")

    def check_attempt(self) -> None:
        self.check_time()
        if self.max_attempts and self.attempts >= self.max_attempts:
            raise BudgetExceeded("request_limit")

    def begin_attempt(self) -> None:
        self.check_attempt()
        self.attempts += 1

    def sleep_delay(self, delay: float) -> float:
        self.check_attempt()
        remaining = self.remaining()
        if remaining is not None and delay >= remaining:
            raise BudgetExceeded("time_limit")
        return delay

    def timeout(self, configured: float) -> float:
        self.check_time()
        remaining = self.remaining()
        return configured if remaining is None else min(configured, remaining)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    retries: int = 0
    latency: float = 0.0
    extra: dict = field(default_factory=dict)  # provider fields to echo back with the assistant message
    retry_reasons: list[str] = field(default_factory=list)  # why each retry happened, e.g. "HTTP 429"


def serialize_prompt(tools: list[dict], messages: list[dict]) -> str:
    """Canonical text of what the model sees, tools first (as chat templates render them).

    Used for prefix-stability accounting and for token estimation. Insertion order is kept
    on purpose: the same objects must always serialise to the same bytes.
    """
    return json.dumps(tools, ensure_ascii=False) + "\n" + json.dumps(messages, ensure_ascii=False)


def _parse_usage(data: dict) -> Usage:
    raw = data.get("usage")
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("usage must be an object")
    raw = raw or {}
    details = raw.get("prompt_tokens_details")
    timings = data.get("timings")
    if any(value is not None and not isinstance(value, dict) for value in (details, timings)):
        raise ValueError("token details and timings must be objects")
    details = details or {}
    cached = (
        details.get("cached_tokens")  # OpenAI, vLLM (--enable-prompt-tokens-details), OpenRouter
        or raw.get("cache_read_input_tokens")  # Anthropic-style
        or (timings or {}).get("cache_n")  # llama.cpp server
        or 0
    )
    return Usage(int(raw.get("prompt_tokens") or 0), int(raw.get("completion_tokens") or 0), int(cached))


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):  # some servers return content parts
        if any(not isinstance(part, dict) or not isinstance(part.get("text", ""), str) for part in content):
            raise ValueError("content parts must be objects with string text")
        return "".join(part.get("text", "") for part in content).strip()
    if not isinstance(content, str):
        raise ValueError("content must be text or a list of parts")
    return content.strip()


class LLMClient:
    def __init__(self, cfg: Config, session: requests.Session | None = None, sleep=time.sleep):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.sleep = sleep
        self.budget: RunBudget | None = None
        self.headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            self.headers["Authorization"] = f"Bearer {cfg.api_key}"

    def chat(self, messages: list[dict], tools: list[dict], *, tool_choice: str = "auto",
             max_tokens: int | None = None) -> LLMResponse:
        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_output_tokens,
        }
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        if tools:
            # `tools` is sent even for tool_choice="none" (summaries): dropping it would change
            # the rendered prefix and miss the KV-cache.
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        payload.update(self.cfg.extra_body)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        started = time.monotonic()
        last_error = "no attempt made"
        reasons: list[str] = []
        for attempt in range(self.cfg.max_retries + 1):
            if attempt:
                delay = self._backoff(attempt, retry_after)
                self.sleep(self.budget.sleep_delay(delay) if self.budget else delay)
            retry_after = None
            if self.budget:
                self.budget.begin_attempt()
            try:
                http = self.session.post(f"{self.cfg.base_url}/chat/completions", data=body,
                                         headers=self.headers,
                                         timeout=self.budget.timeout(self.cfg.request_timeout)
                                         if self.budget else self.cfg.request_timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                reasons.append(type(exc).__name__)
                continue

            if http.status_code == 200:
                try:
                    data = http.json()
                    if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                        raise ValueError("response must have a choices array")
                    choice = data["choices"][0]
                    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                        raise ValueError("choice and message must be objects")
                    message = choice["message"]
                    tool_calls = message.get("tool_calls")
                    tool_calls = [] if tool_calls is None else tool_calls
                    if not isinstance(tool_calls, list):
                        raise ValueError("tool_calls must be an array")
                    for call in tool_calls:
                        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                            raise ValueError("tool calls must contain function objects")
                        if not isinstance(call["function"].get("name"), str):
                            raise ValueError("tool names must be strings")
                        if call.get("id") is not None and not isinstance(call["id"], str):
                            raise ValueError("tool call ids must be strings")
                        if call.get("type", "function") != "function":
                            raise ValueError("unsupported tool call type")
                    finish = choice.get("finish_reason")
                    if finish is not None and not isinstance(finish, str):
                        raise ValueError("finish_reason must be a string")
                    return LLMResponse(
                        content=_text(message.get("content")),
                        tool_calls=tool_calls,
                        finish_reason=finish or "stop",
                        usage=_parse_usage(data),
                        retries=attempt,
                        latency=time.monotonic() - started,
                        extra={k: message[k] for k in self.cfg.passthrough_fields if message.get(k) is not None},
                        retry_reasons=reasons,
                    )
                except (ValueError, KeyError, IndexError, TypeError, OverflowError) as exc:
                    last_error = f"malformed response body ({type(exc).__name__})"
                    reasons.append("malformed body")
                    continue

            detail = http.text[:500]
            if http.status_code in (400, 413, 422) and any(m in detail.lower() for m in OVERFLOW_MARKERS):
                raise ContextOverflowError(detail)
            if http.status_code in RETRY_STATUSES:
                last_error = f"HTTP {http.status_code}: {detail}"
                reasons.append(f"HTTP {http.status_code}")
                retry_after = _retry_after(http)
                continue
            raise FatalLLMError(f"HTTP {http.status_code}: {detail}")

        raise LLMError(f"giving up after {self.cfg.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _backoff(attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 120.0)
        return min(60.0, 1.5 * 2 ** (attempt - 1)) * (0.75 + random.random() / 2)


def _retry_after(http: requests.Response) -> float | None:
    try:
        return float(http.headers.get("Retry-After", ""))
    except ValueError:
        return None
