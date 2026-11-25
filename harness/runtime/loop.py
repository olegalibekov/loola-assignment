"""The agent loop.

Division of labour for failures:
  * anything the model can fix by changing its next action goes back to it as a tool result
    (`ERROR [kind]: ... Hint: ...`);
  * anything it cannot fix - transport, window overflow, output truncation, malformed history,
    runaway loops, budgets - is handled here, deterministically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import Config
from ..context.manager import Context
from ..llm.client import BudgetExceeded, ContextOverflowError, FatalLLMError, LLMClient, LLMError, LLMResponse, RunBudget, serialize_prompt
from .metrics import Metrics, PrefixTracker, Trace
from ..context.prompt import build_system_prompt
from ..tools.registry import Outcome, ToolRegistry
from ..context.skills import SkillStore
from ..tools import parse_arguments

MAX_CONTINUATIONS = 3  # replies cut off by max_tokens that we ask the model to continue
MAX_EMPTY_REPLIES = 2
REPEAT_WARN, REPEAT_STOP = 3, 6  # identical consecutive tool calls
MIN_MESSAGES_BETWEEN_COMPACTIONS = 6


class _ResponseStopped(FatalLLMError):
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


@dataclass
class RunResult:
    status: str  # finished | max_turns | stuck | context_overflow | llm_error | content_filter | interrupted | request_limit | time_limit
    final_text: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "finished"


@dataclass
class _Call:
    id: str
    name: str
    args: dict | None
    error: str | None
    repaired: bool
    replay_arguments: str  # what goes back into history: always valid JSON
    extras: dict = field(default_factory=dict)  # provider fields on the call (e.g. thought signatures)

    def as_history(self) -> dict:
        return {**self.extras, "id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.replay_arguments}}


class Agent:
    def __init__(self, cfg: Config, llm=None, echo=print):
        self.cfg = cfg
        self.llm = llm or LLMClient(cfg)
        self.echo = echo or (lambda *_: None)
        self.skills = SkillStore(cfg.skills_dir)
        self.registry = ToolRegistry(cfg, self.skills)
        self.metrics = Metrics()
        self.trace = Trace(cfg.run_dir)
        self.prefix = PrefixTracker()
        self._last_compaction_len = 0
        self._repeat_key: str | None = None
        self._repeat_count = 0
        self.budget = RunBudget(cfg.max_api_attempts, cfg.max_wall_seconds)

    # ------------------------------------------------------------------ public

    def run(self, task: str) -> RunResult:
        self.budget = RunBudget(self.cfg.max_api_attempts, self.cfg.max_wall_seconds)
        if isinstance(self.llm, LLMClient):
            self.llm.budget = self.budget
        ctx = Context(build_system_prompt(self.cfg, self.registry, self.skills), task, self.cfg)
        self.trace.event("start", tool_mode=self.cfg.tool_mode, compaction=self.cfg.compaction,
                         context_window=self.cfg.context_window, model=self.cfg.model)
        status, final = "max_turns", ""
        try:
            status, final = self._loop(ctx)
        except BudgetExceeded as exc:
            status, self.metrics.error = exc.status, str(exc)
            self.metrics.intercepted[exc.status] += 1
        except KeyboardInterrupt:
            status, self.metrics.error = "interrupted", "run interrupted"
            self.metrics.intercepted["interrupted"] += 1
        except _ResponseStopped as exc:
            status, self.metrics.error = exc.status, str(exc)
        except ContextOverflowError as exc:
            status, self.metrics.error = "context_overflow", str(exc)[:300]
        except LLMError as exc:
            status, self.metrics.error = "llm_error", str(exc)[:300]
        self.metrics.status = status
        self.metrics.api_attempts = self.budget.attempts
        data = self.metrics.to_dict()
        self.trace.event("end", **data)
        if self.cfg.run_dir is not None:
            (self.cfg.run_dir / "metrics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return RunResult(status, final, data)

    # ------------------------------------------------------------------ loop

    def _loop(self, ctx: Context) -> tuple[str, str]:
        continuations = empty_replies = 0
        for turn in range(1, self.cfg.max_turns + 1):
            self.budget.check_time()
            self.metrics.turns = turn
            self.echo(f"\n{'=' * 60}\nTurn {turn}\n{'=' * 60}")
            resp = self._request(ctx)
            content, raw_calls = resp.content, resp.tool_calls
            if content:
                self.echo(f"\n{content}")

            if not raw_calls:
                if resp.finish_reason == "length":
                    if continuations >= MAX_CONTINUATIONS:
                        self.metrics.intercepted["truncated_reply_limit"] += 1
                        raise LLMError("reply remained truncated after the continuation limit")
                    continuations += 1
                    self.metrics.intercepted["continued_truncated_reply"] += 1
                    ctx.append({**resp.extra, "role": "assistant", "content": content or "(output limit reached)"})
                    ctx.append({"role": "user", "content": "[harness] Your reply was cut off by the output-token "
                                "limit. Continue exactly where you stopped; do not repeat what you already wrote."})
                    continue
                if not content:
                    if empty_replies >= MAX_EMPTY_REPLIES:
                        self.metrics.intercepted["empty_reply_limit"] += 1
                        raise LLMError("model returned empty replies after the recovery limit")
                    empty_replies += 1
                    self.metrics.intercepted["empty_reply"] += 1
                    ctx.append({"role": "assistant", "content": "(empty reply)"})
                    ctx.append({"role": "user", "content": "[harness] Your reply was empty. Call a tool to continue "
                                "the task, or give the final answer as plain text if it is complete."})
                    continue
                self.echo("\nAgent finished")
                return "finished", content

            calls = [self._normalise(call, turn, index) for index, call in enumerate(raw_calls)]
            # ONE assistant message carrying ALL tool calls, then one tool message per call.
            # Provider extras (reasoning blocks, thought signatures) are echoed back byte-for-byte:
            # some APIs reject the next request without them.
            ctx.append({**resp.extra, "role": "assistant", "content": content or None,
                        "tool_calls": [c.as_history() for c in calls]})
            stuck = False
            for call in calls:
                self.budget.check_time()
                outcome = self._execute(call, truncated=resp.finish_reason == "length")
                text, stuck_now = self._guard_repeats(call, outcome.text)
                stuck = stuck or stuck_now
                ctx.append({"role": "tool", "tool_call_id": call.id, "content": text})
                self.budget.check_time()
                self.echo(f"  [{call.name}] {json.dumps(call.args, ensure_ascii=False)[:200] if call.args is not None else '<unparseable>'}"
                          f"\n   -> {text[:400]}{'...' if len(text) > 400 else ''}")
            if stuck:
                self.metrics.error = f"same tool call repeated {REPEAT_STOP} times in a row"
                return "stuck", ""
        return "max_turns", ""

    # ------------------------------------------------------------------ LLM requests

    def _chat(self, messages: list[dict], tools: list[dict], kind: str, **kwargs) -> LLMResponse:
        """Every request goes through here so usage and prefix reuse are always accounted."""
        prompt = serialize_prompt(tools, messages)
        self.budget.check_time()
        if not isinstance(self.llm, LLMClient):
            self.budget.begin_attempt()  # stand-ins have one attempt per chat invocation
        resp = self.llm.chat(messages, tools, **kwargs)
        shared, total = self.prefix.observe(prompt)
        self.metrics.record_request(resp, shared, total)
        self.trace.event("llm", kind=kind, prompt_tokens=resp.usage.prompt_tokens,
                         completion_tokens=resp.usage.completion_tokens, cached_tokens=resp.usage.cached_tokens,
                         prefix_reuse=round(shared / total, 4) if total else 0, finish=resp.finish_reason,
                         retries=resp.retries, retry_reasons=resp.retry_reasons, latency=round(resp.latency, 2),
                         messages=len(messages), tools=len(tools))
        self.budget.check_time()
        if resp.finish_reason == "content_filter":
            self.metrics.intercepted["content_filter"] += 1
            raise _ResponseStopped("content_filter", "provider filtered the response")
        if resp.finish_reason not in ("stop", "length", "tool_calls"):
            self.metrics.intercepted["invalid_finish_reason"] += 1
            raise _ResponseStopped("llm_error", f"unsupported provider finish reason: {resp.finish_reason!r}")
        if resp.finish_reason == "tool_calls" and not resp.tool_calls:
            self.metrics.intercepted["invalid_finish_reason"] += 1
            raise _ResponseStopped("llm_error", "provider reported tool_calls without any calls")
        return resp

    def _request(self, ctx: Context) -> LLMResponse:
        tools = self.registry.native_tools()
        if (self.cfg.compaction and ctx.needs_compaction(tools)
                and len(ctx.messages) - self._last_compaction_len >= MIN_MESSAGES_BETWEEN_COMPACTIONS):
            self._compact(ctx, tools, reason="threshold")
        for attempt in range(3):
            overflow: str | None = None
            if self.cfg.enforce_window and ctx.overflows(tools):
                overflow = (f"prompt of ~{ctx.tokens(tools)} tokens exceeds the usable window of "
                            f"{self.cfg.usable_window} (simulated limit)")
            else:
                try:
                    resp = self._chat(ctx.messages, tools, kind="turn")
                    ctx.calibrate(len(serialize_prompt(tools, ctx.messages)), resp.usage.prompt_tokens)
                    return resp
                except ContextOverflowError as exc:
                    overflow = str(exc)
            # Overflow is the harness's problem, not the model's: compact and retry, harder each time.
            if not self.cfg.compaction or attempt == 2:
                raise ContextOverflowError(overflow)
            compacted = self._compact(ctx, tools, reason="overflow", keep_nothing=attempt == 1)
            if not compacted and attempt == 0:
                # A single tool turn has no older turn to cut. Summarise that whole turn
                # before giving up; the untouched system/task-only prompt still cannot shrink.
                compacted = self._compact(ctx, tools, reason="overflow", keep_nothing=True)
            if not compacted:
                raise ContextOverflowError(overflow)
            self.metrics.intercepted["overflow_recovered"] += 1
        raise ContextOverflowError("unreachable")

    def _compact(self, ctx: Context, tools: list[dict], reason: str, keep_nothing: bool = False) -> bool:
        before = ctx.tokens(tools)

        def summarize(messages: list[dict]) -> str:
            if self.cfg.enforce_window and ctx.tokens(tools, messages) > self.cfg.context_window:
                return ""  # the summary request itself would not fit: mechanical fallback
            try:
                response = self._chat(messages, tools, kind="summary", tool_choice="none",
                                      max_tokens=self.cfg.summary_max_tokens)
                if response.finish_reason not in ("stop", "length") or response.tool_calls:
                    self.metrics.intercepted["invalid_summary_reply"] += 1
                    return ""
                return response.content
            except BudgetExceeded:
                raise
            except FatalLLMError:
                raise
            except LLMError:  # overflow or exhausted retries: fall back to the mechanical summary
                return ""

        event = ctx.compact(summarize, self.registry.pinned_state(), tools, keep_nothing=keep_nothing)
        if event is None:
            return False
        if event["level"] == "summary":  # skill bodies were summarised away; level 1 keeps them
            self.skills.on_compaction()
        self._last_compaction_len = len(ctx.messages)
        event.update(reason=reason, tokens_before=before, tokens_after=ctx.tokens(tools), turn=self.metrics.turns)
        self.metrics.compactions.append(event)
        self.trace.event("compaction", **event)
        self.echo(f"\n[harness] compacted context ({event['level']}, {reason}): ~{before} -> ~{event['tokens_after']} tokens")
        return True

    # ------------------------------------------------------------------ tool calls

    def _normalise(self, raw: dict, turn: int, index: int) -> _Call:
        fn = raw.get("function") or {}
        args, error, repaired = parse_arguments(fn.get("arguments"))
        original = fn.get("arguments")
        if isinstance(original, str) and not original.strip():
            repaired = True
        # History must stay replayable: servers reject assistant messages whose arguments are not JSON.
        replay = original if isinstance(original, str) and error is None and not repaired else json.dumps(
            args or {}, ensure_ascii=False)
        extras = {k: v for k, v in raw.items() if k not in ("id", "type", "function", "index")}
        return _Call(raw.get("id") or f"call_{turn}_{index}", str(fn.get("name") or ""), args, error, repaired, replay,
                     extras)

    def _execute(self, call: _Call, truncated: bool) -> Outcome:
        self.metrics.tool_calls += 1
        if call.error is not None:
            if truncated:
                kind = "truncated_call"
                text = (f"ERROR [{kind}]: your tool call was cut off by the output-token limit "
                        f"({self.cfg.max_output_tokens}) before its arguments were complete. Nothing was executed.\n"
                        "Hint: do the work in smaller pieces - write large files in several steps.")
            else:
                kind = "invalid_json"
                text = (f"ERROR [{kind}]: {call.error}. Nothing was executed.\n"
                        "Hint: send the arguments as one JSON object matching the tool schema.")
            outcome = Outcome(text, ok=False, kind=kind, tool=call.name)
        else:
            outcome = self.registry.execute(call.name, call.args or {})
            if call.repaired:
                outcome.intercepted.append("repaired_json")
        self.metrics.tool_usage[outcome.tool or call.name] += 1
        if not outcome.ok:
            self.metrics.tool_errors[outcome.kind] += 1
        for fix in outcome.intercepted:
            self.metrics.intercepted[fix] += 1
        self.trace.event("tool", name=call.name, resolved=outcome.tool, ok=outcome.ok, kind=outcome.kind,
                         intercepted=outcome.intercepted, chars=len(outcome.text))
        return outcome

    def _guard_repeats(self, call: _Call, text: str) -> tuple[str, bool]:
        """Detect a model stuck on one call. Nudge first (appended text), stop the run later."""
        key = call.name + "\0" + json.dumps(call.args, sort_keys=True, ensure_ascii=False)
        self._repeat_count = self._repeat_count + 1 if key == self._repeat_key else 1
        self._repeat_key = key
        if self._repeat_count >= REPEAT_STOP:
            return text, True
        if self._repeat_count >= REPEAT_WARN:
            self.metrics.intercepted["repeat_warning"] += 1
            text += (f"\n[harness] This exact call has now been made {self._repeat_count} times in a row with the "
                     "same arguments. Repeating it will not change the result - take a different approach.")
        return text, False
