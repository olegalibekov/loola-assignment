# Agent-assisted development record

This is an edited summary of the implementation and review work performed with Claude Code and Codex.
The task briefs below are normalized summaries, not verbatim prompts. Repeated troubleshooting,
credentials, personal details, and unrelated discussion are omitted. Findings and validation results
retain their original scope; unfinished work is identified explicitly.

## 1. Inspect the baseline and define acceptance criteria

**Task brief:** Inspect the supplied agent loop. Identify correctness issues and establish what a
successful implementation must demonstrate.

The baseline has one shell tool and an append-only conversation. Its limitations include unguarded
argument parsing, one assistant message per tool call, no HTTP retries, and unbounded tool output.
The original behavior was retained as the benchmark reference, with instrumentation isolated by
`# [bench]` annotations. A separate retry variant distinguishes transport resilience from loop behavior.

Acceptance criteria were made explicit: discover tools and skills on demand; preserve a stable request
prefix between compactions; recover from context pressure; measure token usage and task success;
and verify the generated workspace independently of the agent's final answer.

## 2. Analyze prefix-cache behavior and choose the loading strategy

**Task brief:** Analyze which request changes invalidate the shared prefix. Design lazy loading around
stable schemas and deterministic serialization.

Three strategies provide observable controls: eager schemas, append-only schema loading, and a fixed
dispatch interface. In dispatch mode, tool discovery and skill content arrive as tool results at the
end of the conversation. The native tool definitions remain constant. A compact catalog index supplies
names and signatures without including every full schema and skill body in every request.

The system prompt excludes timestamps, working-directory values, and per-run counters. History is
append-only between compactions. Each model reply produces one assistant message containing all of
its calls, followed by one tool result per call. Stored arguments are always valid JSON.

## 3. Implement context compaction with explicit invariants

**Task brief:** Implement bounded context growth while preserving task state, tool-call boundaries,
and the stable system/task prefix.

Tool output is capped before entering history. Compaction uses a high trigger and a lower target to
avoid repeated rewrites near the window boundary. The first level replaces old large outputs with
deterministic stubs. When that is insufficient, the second level requests a structured summary.

Cuts occur at complete turn boundaries. The system prompt and original task prefix stay unchanged;
recent messages are retained verbatim. Loaded skills, discovered tools, and references to saved
outputs are carried forward deterministically. The summary request retains the same tool definitions
and appends an instruction to the existing conversation.

Compaction necessarily changes the summarized portion of history. The design preserves the earlier
static prefix and makes this rewrite infrequent; it does not claim that all cached context survives.

## 4. Harden execution and classify errors by who can recover

**Task brief:** Review failure paths. Return actionable errors to the model and handle infrastructure
failures deterministically in the harness.

Missing files, invalid arguments, ambiguous edits, and other correctable tool errors return structured
`ERROR [kind]` messages with hints. Transport retries, context overflow, truncated responses, malformed
history, and repeated-call loops are handled by the harness. Authentication and configuration failures
terminate the run rather than being retried indefinitely.

Additional fixes include strict provider completion-status handling, shared API-attempt budgets,
a cooperative run deadline, interruption metrics, bounded subprocess output capture, finite pipe-drain
and process-reap waits, and safe filenames for saved outputs. The time budget is explicitly cooperative;
it does not preempt arbitrary in-process work.

Docker uses separate agent and checker images. The agent image excludes the task generator, hidden
checker, and reference replay. Tool processes run under an unprivileged identity with a scrubbed
environment; filesystem operations enforce workspace boundaries.

## 5. Compare baseline and improved variants with reproducible evidence

**Task brief:** Compare token usage and task completion on the same long, multi-step task. Separate
mechanical replay evidence from real-model performance.

The benchmark generates a deterministic workspace and evaluates it with 12 independent checks.
Scripted replay exercises real tools through a fixed trajectory, allowing comparisons of loading,
output caps, compaction, and reusable-prefix length. Its token counts are estimates; replay success
is not evidence of model reasoning performance.

The model runner records provider usage and checker results. Controlled variants isolate loading
(`eager-managed` versus `full`), compaction (`full-nocompact` versus `full`), and output capping
(`full-no-cap` versus `full`). Source manifests, environment metadata, and Docker image IDs make each
series auditable. A local per-key lock prevents overlapping series, and source drift stops a series.

Saved reports remain in [bench/results](../../bench/results); the [README](../../README.md) explains
which measurements belong to the earlier implementation. Later real-model attempts encountered
HTTP 503 and 429 responses. Those failures do not establish an advantage for compaction or loading.
The baseline logs do not retain enough provider detail to identify the precise exhausted quota.

## 6. Refactor by responsibility and verify the result

**Task brief:** Review module boundaries, organize related files into subfolders, and verify that
structural changes preserve behavior.

The harness is grouped into runtime, context, LLM, and tool packages. Catalog implementations are
separated by operation type. Benchmark code is grouped into task, runners, and sandbox packages;
tests mirror these responsibilities. Public CLI entry points remain small.

Before and after the file moves, snapshots confirmed identical system prompts and ordered tool
schemas in all three loading modes. The task text, generated workspace, and checker function bodies
also matched. This avoids attributing a benchmark change to a directory refactor.

| Validation performed | Observed result | Scope |
|---|---|---|
| Full offline test suite | 214 passed | Harness, tools, and benchmark logic |
| Docker selfcheck | 20 probes passed | Tested isolation and execution boundaries, using a fake key |
| Docker checker with scripted reference solution | 12/12 checks passed | Pipeline verification, not model success |
| CLI help checks | 8 entry points passed | Preserved command interfaces |
| Offline wheel build | Build succeeded; all 29 harness modules included | Package contents |
| Prompt and task snapshots | Matched before and after moves | Refactor invariants |

GitHub Actions configuration covers Python 3.10, 3.12, and 3.14 plus Docker selfcheck. These local
results do not imply that the remote CI matrix has run successfully.

## Review handoff

The implementation, assignment answers, tests, and this session record are available for review.
The remaining evidence gap is a completed comparable real-model series for the current version.
GitHub publication and remote CI validation have not been confirmed in this record.

Credentials, IDE settings, virtual environments, caches, and local run workspaces are excluded from
the review archive. Benchmark reports are retained as supporting evidence. Superseded session exports
were archived outside the repository when this consolidated record was prepared.
