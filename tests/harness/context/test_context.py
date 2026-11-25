from harness.context.manager import ELIDE_MIN_CHARS, Context, elide_outputs, mechanical_summary, section
from harness.runtime.metrics import PrefixTracker


def turn(i, size=2000):
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * size}]


def filled(make_cfg, turns=10, **cfg):
    cfg.setdefault("compact_elide", False)  # most tests here exercise level 2 (summary)
    ctx = Context("SYSTEM", "TASK", make_cfg(context_window=5000, max_output_tokens=1000, **cfg))
    for i in range(turns):
        for message in turn(i):
            ctx.append(message)
    return ctx


def test_cut_is_on_a_turn_boundary_and_respects_the_tail_budget(make_cfg):
    ctx = filled(make_cfg)
    cut = ctx.cut_index()
    assert ctx.messages[cut]["role"] == "assistant" and cut >= 3
    assert ctx._message_tokens(ctx.messages[cut:]) <= ctx.cfg.compact_keep_tail * ctx.cfg.usable_window


def test_compact_folds_summary_into_task_message(make_cfg):
    ctx = filled(make_cfg)
    seen = {}

    def summarize(messages):
        seen["messages"] = messages
        return "## Done\nstuff"

    before = list(ctx.messages)
    event = ctx.compact(summarize, pinned="- Skills loaded earlier: a", tools=[])
    assert seen["messages"][:-1] == before[:len(seen["messages"]) - 1]  # head sent unchanged
    assert ctx.messages[0] == before[0]
    assert ctx.messages[1]["content"].startswith("TASK\n\n<context_summary number=\"1\">")
    assert "## Done\nstuff" in ctx.messages[1]["content"] and "Skills loaded earlier: a" in ctx.messages[1]["content"]
    assert ctx.messages[2:] == before[-event["kept_messages"]:]
    assert event["dropped_messages"] + event["kept_messages"] == len(before) - 2 and not event["fallback"]


def test_second_compaction_replaces_the_first_summary(make_cfg):
    ctx = filled(make_cfg)
    ctx.compact(lambda m: "first", pinned="", tools=[])
    for i in range(10, 20):
        for message in turn(i):
            ctx.append(message)
    ctx.compact(lambda m: "second", pinned="", tools=[])
    task_message = ctx.messages[1]["content"]
    assert task_message.count("<context_summary") == 1 and "second" in task_message and 'number="2"' in task_message


def test_empty_summary_falls_back_to_mechanical_log(make_cfg):
    ctx = filled(make_cfg)
    event = ctx.compact(lambda m: "  ", pinned="", tools=[])
    assert event["fallback"] and "- bash({}) -> xxxx" in ctx.messages[1]["content"]


def test_nothing_to_cut_returns_none(make_cfg):
    ctx = Context("S", "T", make_cfg())
    assert ctx.compact(lambda m: "s", pinned="", tools=[]) is None
    for message in turn(0):
        ctx.append(message)
    assert ctx.compact(lambda m: "s", pinned="", tools=[]) is None  # a single turn has no head to summarise


def test_keep_nothing_drops_the_whole_tail(make_cfg):
    ctx = filled(make_cfg)
    ctx.compact(lambda m: "all", pinned="", tools=[], keep_nothing=True)
    assert len(ctx.messages) == 2 and ctx.messages[1]["role"] == "user"


def test_calibration_tracks_the_server_tokenizer(make_cfg):
    ctx = filled(make_cfg)
    baseline = ctx.tokens([])
    ctx.calibrate(prompt_chars=10_000, prompt_tokens=5_000)  # 2 chars per token
    assert ctx.chars_per_token == 2.0 and ctx.tokens([]) > baseline * 1.5


def test_mechanical_summary_lists_calls_with_results():
    messages = turn(1, size=5) + turn(2, size=5)
    assert mechanical_summary(messages).count("- bash(") == 2


def test_prefix_tracker_measures_shared_prefix():
    tracker = PrefixTracker()
    assert tracker.observe("abcdef") == (0, 6)
    assert tracker.observe("abcdefgh") == (6, 8)
    assert tracker.observe("abXdefgh") == (2, 8)
    assert (tracker.reused, tracker.total) == (8, 22)


# ---------------------------------------------------------------- level 1: elide without a model call

def test_level_one_stubs_old_outputs_and_needs_no_model(make_cfg):
    ctx = filled(make_cfg, compact_elide=True)
    before = list(ctx.messages)

    def summarize(messages):
        raise AssertionError("level 1 must not call the model")

    event = ctx.compact(summarize, pinned="", tools=[])
    assert event["level"] == "elide" and event["elided_messages"] > 0 and ctx.compactions == 0
    assert len(ctx.messages) == len(before) and ctx.messages[:2] == before[:2]
    for old, new in zip(before, ctx.messages):
        if old["role"] != "tool":
            assert new == old  # assistant messages (and any provider signatures on them) are untouched
    tail = ctx.messages[-event["kept_messages"]:]
    assert tail == before[-event["kept_messages"]:]  # recent turns stay verbatim
    stub = next(m["content"] for m in ctx.messages if m["role"] == "tool" and m["content"].startswith("[harness:"))
    assert "2000 chars" in stub and "First line: xxx" in stub
    assert ctx.tokens([]) <= ctx.cfg.compact_target * ctx.cfg.usable_window


def test_level_one_falls_through_to_summary_without_touching_history_first(make_cfg):
    ctx = filled(make_cfg, compact_elide=True, compact_target=0.05)  # level 1 can never reach this target
    before = list(ctx.messages)
    seen = {}

    def summarize(messages):
        seen["head"] = messages[:-1]
        return "## Done\nx"

    event = ctx.compact(summarize, pinned="", tools=[])
    assert event["level"] == "summary"
    assert seen["head"] == before[:len(seen["head"])]  # the summary request saw the original, un-elided history


def test_elide_keeps_skill_bodies_and_small_outputs():
    big = "y" * (ELIDE_MIN_CHARS + 1)
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "load_skill", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": big},
        {"role": "tool", "tool_call_id": "b", "content": "Exit code: 0\n" + big + "\nlast line"},
        {"role": "tool", "tool_call_id": "c", "content": "short"},
    ]
    out, count = elide_outputs(messages)
    assert count == 1 and out[1]["content"] == big and out[3]["content"] == "short"
    assert out[2]["content"].startswith("[harness: output elided") and "First line: Exit code: 0 | Last line: last line" in out[2]["content"]
    assert messages[2]["content"].startswith("Exit code")  # input not mutated
    assert elide_outputs(out)[1] == 0  # idempotent: stubs are short


# ---------------------------------------------------------------- unverified requirements survive

def test_unverified_section_is_carried_over_when_a_later_summary_drops_it(make_cfg):
    ctx = filled(make_cfg)
    ctx.compact(lambda m: "## Done\na\n## Unverified\n- step 7 not checked\n- report value\n## Next\nb", pinned="", tools=[])
    assert ctx.unverified == "- step 7 not checked\n- report value"
    for i in range(10, 20):
        for message in turn(i):
            ctx.append(message)
    ctx.compact(lambda m: "## Done\neverything\n## Next\nfinish", pinned="", tools=[])  # the model forgot the section
    text = ctx.messages[1]["content"]
    assert "carried over by the harness" in text and "- step 7 not checked" in text and "- report value" in text


def test_unverified_section_is_replaced_when_the_model_updates_it(make_cfg):
    ctx = filled(make_cfg)
    ctx.compact(lambda m: "## Unverified\n- a\n## Next\nx", pinned="", tools=[])
    for i in range(10, 20):
        for message in turn(i):
            ctx.append(message)
    ctx.compact(lambda m: "## Unverified\nnone\n## Next\nx", pinned="", tools=[])
    assert ctx.unverified == "none" and "carried over" not in ctx.messages[1]["content"]


def test_section_parser():
    text = "## Done\nx\n## Unverified\n- one\n- two\n\n## Next\ny"
    assert section(text, "Unverified") == "- one\n- two" and section(text, "Next") == "y" and section(text, "Missing") == ""
