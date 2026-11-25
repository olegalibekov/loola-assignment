import json

import pytest
import requests

from harness.llm.client import BudgetExceeded, ContextOverflowError, FatalLLMError, LLMClient, LLMError, RunBudget, serialize_prompt


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.sent.append({"url": url, "body": json.loads(data), "headers": headers})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def ok(content="hi", tool_calls=None, usage=None, finish="stop", **extra):
    return FakeResponse(200, {"choices": [{"message": {"content": content, "tool_calls": tool_calls},
                                           "finish_reason": finish}],
                              "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2}, **extra})


def client(make_cfg, replies, **cfg):
    sleeps = []
    session = FakeSession(replies)
    return LLMClient(make_cfg(api_key="k", **cfg), session=session, sleep=sleeps.append), session, sleeps


MESSAGES = [{"role": "user", "content": "x"}]
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]


def test_success_parses_content_tools_and_usage(make_cfg):
    tool_calls = [{"id": "1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
    usage = {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 80}}
    llm, session, _ = client(make_cfg, [ok("  text ", tool_calls, usage, finish="tool_calls")])
    resp = llm.chat(MESSAGES, TOOLS)
    assert (resp.content, resp.finish_reason, resp.tool_calls) == ("text", "tool_calls", tool_calls)
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.cached_tokens) == (100, 7, 80)
    sent = session.sent[0]
    assert sent["url"].endswith("/chat/completions") and sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"]["tools"] == TOOLS and sent["body"]["tool_choice"] == "auto"


def test_llama_cpp_cache_counter_is_understood(make_cfg):
    llm, _, _ = client(make_cfg, [ok(timings={"cache_n": 42})])
    assert llm.chat(MESSAGES, TOOLS).usage.cached_tokens == 42


def test_no_tools_key_when_there_are_no_tools(make_cfg):
    llm, session, _ = client(make_cfg, [ok()])
    llm.chat(MESSAGES, [])
    assert "tools" not in session.sent[0]["body"] and "tool_choice" not in session.sent[0]["body"]


def test_transient_failures_are_retried_with_backoff(make_cfg):
    replies = [FakeResponse(429, "slow down", {"Retry-After": "3"}), FakeResponse(503, "busy"),
               requests.ConnectionError("reset"), FakeResponse(200, "<html>gateway</html>"), ok("finally")]
    llm, session, sleeps = client(make_cfg, replies)
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.content == "finally" and resp.retries == 4 and len(session.sent) == 5
    assert resp.retry_reasons == ["HTTP 429", "HTTP 503", "ConnectionError", "malformed body"]
    assert sleeps[0] == 3.0 and len(sleeps) == 4


def test_retries_are_bounded(make_cfg):
    llm, session, _ = client(make_cfg, [FakeResponse(500, "boom")] * 3, max_retries=2)
    with pytest.raises(LLMError, match="giving up after 3 attempts"):
        llm.chat(MESSAGES, TOOLS)
    assert len(session.sent) == 3


def test_total_attempt_budget_stops_retries_before_sleep(make_cfg):
    llm, session, sleeps = client(make_cfg, [FakeResponse(503, "busy"), ok()], max_retries=5)
    llm.budget = RunBudget(max_attempts=1)
    with pytest.raises(BudgetExceeded) as exc:
        llm.chat(MESSAGES, TOOLS)
    assert exc.value.status == "request_limit"
    assert len(session.sent) == 1 and not sleeps and llm.budget.attempts == 1


def test_deadline_limits_retry_waits_and_http_timeout(make_cfg):
    llm, session, sleeps = client(make_cfg, [FakeResponse(429, "busy", {"Retry-After": "60"})])
    llm.budget = RunBudget(max_seconds=2, clock=lambda: 0)
    original = session.post
    timeouts = []

    def post(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return original(*args, **kwargs)

    session.post = post
    with pytest.raises(BudgetExceeded) as exc:
        llm.chat(MESSAGES, TOOLS)
    assert exc.value.status == "time_limit"
    assert timeouts == [2] and not sleeps and len(session.sent) == 1


@pytest.mark.parametrize("body", [
    [],
    {"choices": {}},
    {"choices": []},
    {"choices": [None]},
    {"choices": [{"message": "invalid shape"}]},
    {"choices": [{"message": []}]},
    {"choices": [{"message": {"content": {"text": "invalid shape"}}}]},
    {"choices": [{"message": {"content": [{"text": None}]}}]},
    {"choices": [{"message": {"tool_calls": {}}}]},
    {"choices": [{"message": {"tool_calls": [None]}}]},
    {"choices": [{"message": {"tool_calls": [{"function": "bad"}]}}]},
    {"choices": [{"message": {"tool_calls": [{"function": {"name": 7}}]}}]},
    {"choices": [{"message": {"tool_calls": [{"id": [], "function": {"name": "bash"}}]}}]},
    {"choices": [{"message": {}, "finish_reason": []}]},
    {"choices": [{"message": {}}], "usage": []},
    {"choices": [{"message": {}}], "usage": {"prompt_tokens_details": []}},
    {"choices": [{"message": {}}], "usage": {"prompt_tokens": "not a number"}},
    {"choices": [{"message": {}}], "usage": {"prompt_tokens": float("inf")}},
    {"choices": [{"message": {}}], "timings": []},
])
def test_malformed_response_shapes_are_retried_without_changing_the_request(make_cfg, body):
    llm, session, sleeps = client(make_cfg, [FakeResponse(200, body), ok("recovered")])
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.content == "recovered" and resp.retries == 1
    assert resp.retry_reasons == ["malformed body"] and len(sleeps) == 1
    assert session.sent[0] == session.sent[1]


def test_malformed_response_exhaustion_raises_llm_error(make_cfg):
    body = {"choices": [{"message": {"tool_calls": ["invalid call"]}}]}
    llm, session, _ = client(make_cfg, [FakeResponse(200, body)] * 2, max_retries=1)
    with pytest.raises(LLMError, match="giving up after 2 attempts: malformed response body"):
        llm.chat(MESSAGES, TOOLS)
    assert len(session.sent) == 2


def test_text_content_parts_and_function_extras_survive_parsing(make_cfg):
    tool_call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"},
                 "extra_content": {"google": {"thought_signature": "SIG=="}}}
    llm, _, _ = client(make_cfg, [ok([{"type": "text", "text": "part one "},
                                     {"type": "text", "text": "part two"}], [tool_call])])
    response = llm.chat(MESSAGES, TOOLS)
    assert response.content == "part one part two" and response.tool_calls == [tool_call]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_and_config_errors_fail_fast(make_cfg, status):
    llm, session, sleeps = client(make_cfg, [FakeResponse(status, "nope")])
    with pytest.raises(FatalLLMError):
        llm.chat(MESSAGES, TOOLS)
    assert len(session.sent) == 1 and not sleeps


def test_context_overflow_is_a_distinct_error(make_cfg):
    body = {"error": {"message": "This model's maximum context length is 8192 tokens."}}
    llm, _, _ = client(make_cfg, [FakeResponse(400, body)])
    with pytest.raises(ContextOverflowError):
        llm.chat(MESSAGES, TOOLS)


def test_other_bad_requests_are_fatal_not_overflow(make_cfg):
    llm, _, _ = client(make_cfg, [FakeResponse(400, {"error": {"message": "unknown field foo"}})])
    with pytest.raises(FatalLLMError) as exc:
        llm.chat(MESSAGES, TOOLS)
    assert not isinstance(exc.value, ContextOverflowError)


def test_prompt_serialisation_is_deterministic():
    assert serialize_prompt(TOOLS, MESSAGES) == serialize_prompt(json.loads(json.dumps(TOOLS)), list(MESSAGES))


def test_temperature_can_be_omitted(make_cfg):
    llm, session, _ = client(make_cfg, [ok(), ok()])
    llm.chat(MESSAGES, TOOLS)
    assert session.sent[0]["body"]["temperature"] == 0.1
    llm.cfg.temperature = None
    llm.chat(MESSAGES, TOOLS)
    assert "temperature" not in session.sent[1]["body"]


def test_temperature_env_parsing(monkeypatch):
    from harness.runtime.config import Config
    monkeypatch.setenv("LLM_TEMPERATURE", "none")
    assert Config.from_env().temperature is None
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    assert Config.from_env().temperature == 0.7
    monkeypatch.delenv("LLM_TEMPERATURE")
    assert Config.from_env().temperature == 0.1


def test_provider_fields_are_captured_for_echo(make_cfg):
    body = {"choices": [{"message": {"content": "x", "reasoning_details": [{"type": "reasoning.encrypted", "data": "abc"}],
                                     "refusal": None, "some_vendor_noise": 1}, "finish_reason": "stop"}], "usage": {}}
    llm, _, _ = client(make_cfg, [FakeResponse(200, body)])
    assert llm.chat(MESSAGES, TOOLS).extra == {"reasoning_details": [{"type": "reasoning.encrypted", "data": "abc"}]}
