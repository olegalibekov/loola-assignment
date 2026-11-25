import json

import pytest

from harness.llm.client import ContextOverflowError, FatalLLMError
from harness.runtime.loop import MAX_CONTINUATIONS, MAX_EMPTY_REPLIES, REPEAT_STOP, Agent
from harness.llm.scripted import ScriptedLLM, call, calls, say


def run(make_cfg, script, task="do it", **overrides):
    cfg = make_cfg(**overrides)
    llm = script if isinstance(script, ScriptedLLM) else ScriptedLLM(script)
    agent = Agent(cfg, llm=llm, echo=None)
    return agent.run(task), llm, agent


def roles(messages):
    return [m["role"] for m in messages]


def test_finishes_on_plain_reply_and_writes_artifacts(make_cfg):
    result, llm, agent = run(make_cfg, [call("bash", command="echo hi"), say("done")])
    assert result.ok and result.final_text == "done" and result.metrics["turns"] == 2
    assert (agent.cfg.run_dir / "metrics.json").exists()
    events = [json.loads(line)["event"] for line in (agent.cfg.run_dir / "trace.jsonl").read_text().splitlines()]
    assert events == ["start", "llm", "tool", "llm", "end"]


def test_parallel_tool_calls_become_one_assistant_message(make_cfg):
    """The original loop appended one assistant message per call, which is malformed history."""
    script = [calls([("bash", {"command": "echo a"}), ("bash", {"command": "echo b"})], text="two at once"), say("ok")]
    _, llm, _ = run(make_cfg, script)
    sent = llm.requests[1]["messages"]
    assert roles(sent) == ["system", "user", "assistant", "tool", "tool"]
    assistant = sent[2]
    assert len(assistant["tool_calls"]) == 2
    assert [m["tool_call_id"] for m in sent[3:]] == [c["id"] for c in assistant["tool_calls"]]
    assert "a" in sent[3]["content"] and "b" in sent[4]["content"]


def test_history_is_append_only_between_turns(make_cfg):
    script = [call("bash", command="echo 1"), call("bash", command="echo 2"), call("bash", command="echo 3"), say("ok")]
    result, llm, _ = run(make_cfg, script)
    for earlier, later in zip(llm.requests, llm.requests[1:]):
        assert later["messages"][:len(earlier["messages"])] == earlier["messages"]
        assert later["tools"] == earlier["tools"]
    assert result.metrics["prefix_reuse"] > 0.6


def test_invalid_json_arguments_go_back_to_the_model_and_history_stays_valid(make_cfg):
    script = [calls([("bash", '{"command": "echo hi"')]), say("ok")]
    result, llm, _ = run(make_cfg, script)
    sent = llm.requests[1]["messages"]
    assert sent[3]["content"].startswith("ERROR [invalid_json]")
    json.loads(sent[2]["tool_calls"][0]["function"]["arguments"])  # replayable
    assert result.metrics["tool_errors"] == {"invalid_json": 1}


@pytest.mark.parametrize("arguments", ["", " \n\t"])
def test_empty_arguments_are_replayed_as_valid_json(make_cfg, arguments):
    script = [calls([("list_dir", arguments)]), say("ok")]
    result, llm, _ = run(make_cfg, script)
    sent = llm.requests[1]["messages"]
    assert result.ok and not result.metrics["tool_errors"]
    assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == {}
    assert result.metrics["intercepted"]["repaired_json"] == 1


def test_truncated_tool_call_is_not_executed(make_cfg):
    script = [calls([("write_file", '{"path": "big.txt", "content": "aaaa')], finish="length"), say("ok")]
    result, llm, agent = run(make_cfg, script)
    assert "ERROR [truncated_call]" in llm.requests[1]["messages"][3]["content"]
    assert not (agent.cfg.workdir / "big.txt").exists()


def test_truncated_text_reply_is_continued(make_cfg):
    result, llm, _ = run(make_cfg, [say("part one", finish="length"), say("part two")])
    assert result.ok and result.final_text == "part two"
    assert roles(llm.requests[1]["messages"])[-2:] == ["assistant", "user"]
    assert result.metrics["intercepted"]["continued_truncated_reply"] == 1


def test_empty_reply_gets_a_nudge_instead_of_ending_the_run(make_cfg):
    result, llm, _ = run(make_cfg, [say(""), call("bash", command="echo hi"), say("done")])
    assert result.ok and result.final_text == "done"
    assert "empty" in llm.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("text, finish, limit, recovery, exhausted", [
    ("", "stop", MAX_EMPTY_REPLIES, "empty_reply", "empty_reply_limit"),
    ("partial", "length", MAX_CONTINUATIONS, "continued_truncated_reply", "truncated_reply_limit"),
    ("", "length", MAX_CONTINUATIONS, "continued_truncated_reply", "truncated_reply_limit"),
])
def test_exhausted_reply_recovery_is_a_counted_failure(make_cfg, text, finish, limit, recovery, exhausted):
    result, llm, agent = run(make_cfg, [say(text, finish=finish)] * (limit + 1))
    assert not result.ok and result.status == "llm_error"
    assert len(llm.requests) == limit + 1 and result.final_text == ""
    assert result.metrics["intercepted"][recovery] == limit
    assert result.metrics["intercepted"][exhausted] == 1
    assert "limit" in result.metrics["error"]
    assert json.loads((agent.cfg.run_dir / "metrics.json").read_text())["status"] == "llm_error"


def test_repeated_identical_calls_are_warned_then_stopped(make_cfg):
    result, llm, _ = run(make_cfg, [call("bash", command="false")] * 20)
    assert result.status == "stuck" and result.metrics["turns"] == REPEAT_STOP
    assert "take a different approach" in llm.requests[3]["messages"][-1]["content"]
    assert "take a different approach" not in llm.requests[2]["messages"][-1]["content"]


def test_max_turns(make_cfg):
    script = (call("bash", command=f"echo {i}") for i in range(100))
    result, _, _ = run(make_cfg, script, max_turns=4)
    assert result.status == "max_turns" and result.metrics["turns"] == 4


def test_tool_error_kinds_and_interceptions_are_counted(make_cfg):
    script = [call("nope"), call("calc", expression="1+1"), call("read_file", path="missing.txt"), say("ok")]
    result, _, _ = run(make_cfg, script)
    assert result.metrics["tool_errors"] == {"unknown_tool": 1, "not_found": 1}
    assert result.metrics["intercepted"] == {"auto_routed": 1}


def test_fatal_llm_error_ends_the_run_cleanly(make_cfg):
    def boom(messages, tools):
        raise FatalLLMError("HTTP 401: bad key")

    result, _, _ = run(make_cfg, ScriptedLLM(boom))
    assert result.status == "llm_error" and "401" in result.metrics["error"]


@pytest.mark.parametrize("finish, status", [("error", "llm_error"), ("content_filter", "content_filter"),
                                          ("surprise", "llm_error")])
@pytest.mark.parametrize("with_calls", [False, True])
def test_unsuccessful_finish_reason_never_runs_tools_or_finishes(make_cfg, finish, status, with_calls):
    response = (calls([("write_file", {"path": "unexpected.txt", "content": "bad"})], finish=finish)
                if with_calls else say("looks finished", finish=finish))
    result, _, agent = run(make_cfg, [response])
    assert result.status == status and not result.ok and result.final_text == ""
    assert result.metrics["tool_calls"] == 0 and result.metrics["intercepted"]
    assert not (agent.cfg.workdir / "unexpected.txt").exists()


def test_keyboard_interrupt_writes_final_metrics_and_trace(make_cfg):
    def interrupted(messages, tools):
        raise KeyboardInterrupt

    result, _, agent = run(make_cfg, ScriptedLLM(interrupted))
    assert result.status == "interrupted" and not result.ok
    assert result.metrics["intercepted"]["interrupted"] == 1
    assert result.metrics["api_attempts"] == 1
    assert json.loads((agent.cfg.run_dir / "metrics.json").read_text())["status"] == "interrupted"
    assert json.loads((agent.cfg.run_dir / "trace.jsonl").read_text().splitlines()[-1])["event"] == "end"


def test_api_attempt_budget_includes_overflow_and_summary(make_cfg):
    steps = iter([call("list_dir"), ContextOverflowError("context length exceeded"), say("not reached")])

    def script(messages, tools):
        step = next(steps)
        if isinstance(step, Exception):
            raise step
        return step

    result, llm, _ = run(make_cfg, ScriptedLLM(script), max_api_attempts=3)
    assert result.status == "request_limit" and not result.ok
    assert result.metrics["api_attempts"] == 3 and len(llm.requests) == 3
    assert llm.requests[-1]["tool_choice"] == "none"
    assert result.metrics["intercepted"]["request_limit"] == 1


def test_deadline_crossed_during_response_prevents_tool_execution(make_cfg, monkeypatch):
    from harness.llm.client import RunBudget
    now = [0.0]
    budget = RunBudget(max_seconds=5, clock=lambda: now[0])
    monkeypatch.setattr("harness.runtime.loop.RunBudget", lambda *args: budget)

    def slow_response(messages, tools):
        now[0] = 6
        return call("write_file", path="unexpected.txt", content="bad")

    result, _, agent = run(make_cfg, ScriptedLLM(slow_response), max_wall_seconds=5)
    assert result.status == "time_limit" and result.metrics["api_attempts"] == 1
    assert result.metrics["requests"] == 1 and result.metrics["tool_calls"] == 0
    assert result.metrics["intercepted"]["time_limit"] == 1
    assert not (agent.cfg.workdir / "unexpected.txt").exists()


# ---------------------------------------------------------------- compaction inside the loop

BIG = "python3 -c \"print('z' * 6000)\""
SMALL_WINDOW = dict(context_window=9000, max_output_tokens=1000, enforce_window=True, compact_elide=False)  # level 2
ELIDE_WINDOW = {**SMALL_WINDOW, "compact_elide": True}


def noisy_script(turns):
    return [call("bash", command=f"{BIG} # {i}") for i in range(turns)] + [say("finished")]


def test_without_compaction_the_window_overflows(make_cfg):
    result, _, _ = run(make_cfg, noisy_script(12), compaction=False, **SMALL_WINDOW)
    assert result.status == "context_overflow"


def test_compaction_keeps_the_run_alive_and_the_prefix_stable(make_cfg):
    result, llm, agent = run(make_cfg, noisy_script(12), task="THE TASK", **SMALL_WINDOW)
    assert result.ok and len(result.metrics["compactions"]) >= 1
    assert result.metrics["peak_prompt_tokens"] <= agent.cfg.usable_window

    system = llm.requests[0]["messages"][0]
    for request in llm.requests:
        messages = request["messages"]
        assert messages[0] == system  # the system prompt is never rewritten
        assert messages[1]["content"].startswith("THE TASK")  # the task stays the cached prefix
        assert request["tools"] == llm.requests[0]["tools"]
        # every tool result still has its call, every call its result
        pending = set()
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                pending.add(tool_call["id"])
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending
                pending.discard(message["tool_call_id"])
        assert not pending or request["tool_choice"] == "none" or messages[-1]["role"] != "assistant"

    summaries = [r for r in llm.requests if r["tool_choice"] == "none"]
    assert summaries and "handoff summary" in summaries[0]["messages"][-1]["content"]
    after = llm.requests[llm.requests.index(summaries[0]) + 1]["messages"]
    assert "<context_summary" in after[1]["content"] and after[2]["role"] == "assistant"


def test_summary_request_reuses_the_cached_prefix(make_cfg):
    _, llm, _ = run(make_cfg, noisy_script(12), **SMALL_WINDOW)
    index = next(i for i, r in enumerate(llm.requests) if r["tool_choice"] == "none")
    summary, previous = llm.requests[index]["messages"], llm.requests[index - 1]["messages"]
    head = summary[:-1]
    assert previous[:len(head)] == head  # head of the summary request == prefix of the previous request


def test_compaction_is_rare_not_per_turn(make_cfg):
    result, _, _ = run(make_cfg, noisy_script(30), **SMALL_WINDOW)
    assert result.ok
    assert 2 <= len(result.metrics["compactions"]) <= 12  # hysteresis: several turns between compactions


def test_server_side_overflow_triggers_compaction_and_retry(make_cfg):
    state = {"raised": False, "i": 0}
    steps = noisy_script(6)

    def script(messages, tools):
        if len(messages) > 10 and not state["raised"]:
            state["raised"] = True
            raise ContextOverflowError("maximum context length exceeded")
        state["i"] += 1
        return steps[state["i"] - 1]

    result, llm, _ = run(make_cfg, ScriptedLLM(script), context_window=10**6)
    assert result.ok and result.metrics["compactions"][0]["reason"] == "overflow"
    assert result.metrics["intercepted"]["overflow_recovered"] == 1


@pytest.mark.parametrize("summary_overflows", [False, True])
def test_overflow_after_first_tool_turn_compacts_the_whole_turn(make_cfg, monkeypatch, summary_overflows):
    steps = iter([call("list_dir"), ContextOverflowError("context length exceeded"), say("recovered")])

    def script(messages, tools):
        step = next(steps)
        if isinstance(step, Exception):
            raise step
        return step

    llm = ScriptedLLM(script)
    original_chat = llm.chat

    def chat(messages, tools, **kwargs):
        if summary_overflows and kwargs.get("tool_choice") == "none":
            raise ContextOverflowError("summary also exceeds context length")
        return original_chat(messages, tools, **kwargs)

    monkeypatch.setattr(llm, "chat", chat)
    result, _, _ = run(make_cfg, llm, task="ORIGINAL TASK", context_window=10**6)
    assert result.ok and result.final_text == "recovered"
    assert result.metrics["tool_calls"] == 1
    assert result.metrics["intercepted"]["overflow_recovered"] == 1
    event, = result.metrics["compactions"]
    assert event["kept_messages"] == 0 and event["dropped_messages"] == 2
    assert event["fallback"] == summary_overflows
    recovered = llm.requests[-1]["messages"]
    assert len(recovered) == 2 and recovered[0] == llm.requests[0]["messages"][0]
    assert recovered[1]["content"].startswith("ORIGINAL TASK\n\n<context_summary")


def test_initial_prompt_overflow_fails_without_useless_compactions(make_cfg):
    def script(messages, tools):
        raise ContextOverflowError("initial prompt does not fit")

    result, llm, _ = run(make_cfg, ScriptedLLM(script))
    assert result.status == "context_overflow" and not result.ok
    assert len(llm.requests) == 1 and not result.metrics["compactions"]


def test_pinned_state_survives_compaction(make_cfg):
    script = [call("load_skill", name="log-triage"), call("tool_search", query="grep")] + noisy_script(12)
    _, llm, _ = run(make_cfg, script, **SMALL_WINDOW)
    compacted = next(r["messages"][1]["content"] for r in llm.requests if "<context_summary" in r["messages"][1]["content"])
    assert "log-triage" in compacted and "grep(pattern" in compacted


def test_provider_fields_are_echoed_back_unchanged(make_cfg):
    """Gemini thought signatures and OpenRouter reasoning blocks must survive the round trip."""
    first = calls([("bash", {"command": "echo a"}), ("bash", {"command": "echo b"})])
    first.tool_calls[0]["extra_content"] = {"google": {"thought_signature": "SIG=="}}
    first.tool_calls[0]["index"] = 0
    first.extra = {"reasoning_details": [{"type": "reasoning.encrypted", "data": "blob"}]}
    _, llm, _ = run(make_cfg, [first, say("ok")])
    assistant = llm.requests[1]["messages"][2]
    assert assistant["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "blob"}]
    assert assistant["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG=="}}
    assert "extra_content" not in assistant["tool_calls"][1] and "index" not in assistant["tool_calls"][0]
    assert [m["role"] for m in llm.requests[1]["messages"][2:]] == ["assistant", "tool", "tool"]  # calls first, then results


def test_level_one_compaction_runs_first_and_calls_no_model(make_cfg):
    result, llm, _ = run(make_cfg, noisy_script(12), task="THE TASK", **ELIDE_WINDOW)
    levels = [c["level"] for c in result.metrics["compactions"]]
    assert result.ok and levels and levels[0] == "elide"
    assert sum(r["tool_choice"] == "none" for r in llm.requests) == levels.count("summary")  # elide costs no model call
    for request in llm.requests:
        assert request["messages"][0] == llm.requests[0]["messages"][0]
        assert request["messages"][1]["content"].startswith("THE TASK")
    assert result.metrics["peak_prompt_tokens"] <= 8000


def test_long_run_escalates_from_elide_to_summary(make_cfg):
    result, _, _ = run(make_cfg, noisy_script(40), **ELIDE_WINDOW)
    levels = [c["level"] for c in result.metrics["compactions"]]
    assert result.ok and "elide" in levels and "summary" in levels


def test_skill_stays_loaded_across_level_one_compaction(make_cfg):
    script = [call("load_skill", name="log-triage")] + noisy_script(8) + [call("load_skill", name="log-triage")]
    script[-2], script[-1] = script[-1], script[-2]  # load again just before the final answer
    _, llm, agent = run(make_cfg, script, **ELIDE_WINDOW)
    levels = [c["level"] for c in agent.metrics.compactions]
    if "summary" not in levels:  # body still in context: the second load is answered with a pointer
        assert "already loaded" in llm.requests[-1]["messages"][-1]["content"]


def test_process_rules_are_in_every_system_prompt(make_cfg):
    for mode in ("eager", "append", "dispatch"):
        _, llm, _ = run(make_cfg, [say("done")], tool_mode=mode)
        system = llm.requests[0]["messages"][0]["content"]
        assert "## Before you finish" in system and "Passing tests are not proof" in system
