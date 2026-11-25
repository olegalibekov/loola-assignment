# AGENTS.md

Guidance for coding agents (and humans) working in this repository.

## What this is

A small coding-agent harness in Python, grown from the single-file loop in `baseline/agent_original.py`.
It adds two mechanisms and a benchmark that measures them:

- **dynamic loading** of tools and skills, so dozens of descriptions are not in every prompt;
- **context compaction** when the window fills up;
- both designed so the provider's **prompt prefix cache (KV-cache) stays valid**.

README.md (Russian) is the user-facing write-up: design, results, answers to the assignment's questions.

## Layout

The code is grouped into `harness/{runtime,context,llm,tools}` and `bench/{task,runners,sandbox}`.
Tests mirror those areas. Keep CLI entry points small: `harness/cli.py`, `bench/run.py`,
`bench/replay.py`, and `bench/sandbox/__main__.py` delegate to the implementation packages.

| Path | Role |
|---|---|
| `harness/runtime/loop.py` | Agent loop; decides what goes back to the model vs. what the harness handles |
| `harness/llm/client.py` | OpenAI-compatible client: retries, error classes, usage parsing |
| `harness/context/manager.py` | Message list, token estimation, compaction |
| `harness/tools/registry.py` | `tools` array per mode, `tool_search` / `use_tool`, execution pipeline, output cap |
| `harness/tools/base.py` | Tool definitions, execution context, errors and schema DSL |
| `harness/tools/arguments.py` | Argument parsing, repair, coercion and validation |
| `harness/tools/processes.py` | Child environment, privilege drop, bounded subprocess execution |
| `harness/tools/core.py`, `harness/tools/catalog.py` | Always-on tools and the ordered lazy catalog declarations |
| `harness/tools/implementations/` | Catalog implementations grouped by filesystem, text, Python, Git, data, archives and system operations |
| `harness/context/skills.py`, `skills/*/SKILL.md` | Skill store and the playbooks themselves |
| `harness/context/prompt.py` | Static system prompt builder |
| `harness/runtime/metrics.py` | Metrics, JSONL trace, prefix-reuse tracker |
| `harness/llm/scripted.py` | Scripted LLM stand-in for tests and the offline replay |
| `bench/task/workspace.py`, `bench/task/check.py` | Benchmark task generator and its 12-check verifier |
| `bench/runners/replay.py` | Offline replay: reference solution through every harness variant, no model |
| `bench/runners/model.py` | Real-model benchmark runner |
| `bench/runners/provenance.py` | Source manifests, safe configuration metadata and local per-key series lock |
| `bench/sandbox/docker.py`, `Dockerfile` | Docker sandbox for agent and checker; `python -m bench.sandbox` proves the key is unreachable |
| `baseline/agent_original.py` | The assignment's script. Only `# [bench]` lines may differ from the original |
| `tests/harness/`, `tests/tools/`, `tests/bench/` | Tests grouped by the subsystem they exercise; shared fixtures in `tests/conftest.py` |
| `docs/session-log/session.md` | Consolidated edited development record, decisions, validation and remaining evidence gaps |

## Commands

```bash
uv pip install --python .venv/bin/python -e ".[dev]"   # or: pip install -e ".[dev]"
.venv/bin/python -m pytest                             # ~1 min, no network
.venv/bin/python -m bench.replay                       # offline numbers, no model
.venv/bin/python -m bench.run --runs 5                 # real model; needs .env (see .env.example)
.venv/bin/python -m bench.sandbox                      # build the image, check the key cannot leak
.venv/bin/python -m bench.run --docker --runs 5        # same benchmark inside the sandbox
.venv/bin/python -m harness "task text" --workdir path # run the agent
```

## Invariants - do not break these

They are what keeps the prompt cache valid; `tests/harness/runtime/test_loop.py` and `tests/tools/test_registry.py` enforce them.

1. **The system prompt is static.** `build_system_prompt` is a pure function of config, catalog and skill
   directory. No timestamps, working directory, counters or per-run ids.
2. **`tools` is constant in `dispatch` mode** and append-only in `append` mode. Never reorder or remove entries.
3. **History is append-only between compactions.** To tell the model something, append a message or add text
   to the tool result being appended now. Never edit, trim or drop a message that was already sent.
4. **Compaction is the only rewrite**, it is rare (trigger high, land low), it cuts on a turn boundary, and it
   keeps `messages[0]` and the task text at the start of `messages[1]` byte-identical.
5. **One assistant message per model reply**, carrying all its tool calls, followed by one `tool` message per call.
   Arguments stored in history are always valid JSON.
6. **Serialisation is deterministic.** Do not introduce sets, unordered iteration or `sort_keys` changes on
   anything that ends up in a request.

## While a benchmark is running

Every run starts a fresh `python -m harness` process, so an edit to `harness/` or `skills/` changes the runs
that have not started yet. Do not edit them, and do not start a second benchmark on the same key, until the
series finishes. All runs quoted together in the README must come from one code version.

Skills are generic playbooks. They must never name this benchmark's bugs, data or answers
(`tests/bench/task/test_bench.py::test_skills_do_not_leak_task_specifics`).

## Error-handling rule

Before adding an error path, decide who can fix it:

- the model can, by changing its next action -> return it as a tool result: `ERROR [kind]: message` + `Hint:`.
  Raise `ToolError(kind, message, hint)`; never return a harness traceback.
- only the harness can (transport, window, truncation, malformed history, runaway loops) -> handle it in
  `runtime/loop.py` / `llm/client.py` and count it in `metrics.intercepted`.
- nobody can (auth, bad config) -> fail fast with a run status, no retries.

## Conventions

- Python 3.10+, standard library plus `requests`. Do not add dependencies without a strong reason.
- Tools are plain functions `func(ctx: ToolContext, **args) -> str`, declared with the `params({...})` DSL.
  A new catalog tool needs: an implementation in the relevant `harness/tools/implementations/` module,
  a `_t(...)` entry in `harness/tools/catalog.py`'s `CATALOG_TOOLS`, and a test under `tests/tools/`.
  Keep catalog declarations in their existing order; implementations must not import the catalog or registry.
- A new skill is a folder `skills/<name>/SKILL.md` with `name` and `description` front matter.
- Every behaviour change comes with a test. Tests must not need a network or a model: use `ScriptedLLM`.
- Never execute a destructive command to test the deny-list; test `policy_violation()` as a pure function.

## Benchmark honesty

- Never hand-edit files under `bench/results/`. Numbers in README must be copied from a results file.
- `bench/runners/replay.py` measures harness mechanics with a scripted trajectory and estimated tokens (chars/4). It
  says nothing about model success - label it as such wherever it is quoted.
- Output of `bench/runners/mock_server.py` is a pipeline smoke test, not a result. Do not commit it.
- If the checker or the task changes, old results are no longer comparable: re-run or delete them.

## Secrets

`.env` is git-ignored. Never commit keys, never print them, never put them in the session log.
