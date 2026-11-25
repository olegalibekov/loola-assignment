"""Real-model benchmark: run the long task through each variant, check the result, compare tokens.

    python -m bench.run --runs 5                                  # original, eager, full
    python -m bench.run --variants eager,full --window 32000 --enforce-window
    python -m bench.run --variants full --runs 1 --jobs 1

Needs LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (environment or .env). Every run gets a fresh
workspace; artifacts (conversation log, trace, metrics, checker verdict, final workspace) are
kept under runs/<stamp>/<variant>/<n>/, and the aggregated table under bench/results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # started as a file (IDE "Run"), not with `-m`: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import os
import statistics
import subprocess
import time
from dataclasses import replace

from harness.runtime.config import REPO_ROOT, Config

from bench.sandbox import docker as sandbox
from bench.task.check import check
from bench.runners.provenance import SourceChangedError, runtime_metadata, safe_config, series_lock, source_manifest, verify_source
from bench.task.workspace import DEFAULT_SEED, TASK, make_workspace

HARNESS = [sys.executable, "-m", "harness"]
CORE_AND_META = {"bash", "read_file", "write_file", "edit_file", "tool_search", "use_tool", "load_skill"}
VARIANTS: dict[str, dict] = {
    # the assignment's script as given: one tool, no skills, nothing managed, no retries
    "original": {"cmd": [sys.executable, str(REPO_ROOT / "baseline" / "agent_original.py")],
                 "note": "assignment script as given (bash only, no HTTP retries)"},
    # the same script with transport retries only, so its loop can be measured on a flaky endpoint
    "original-retry": {"cmd": [sys.executable, str(REPO_ROOT / "baseline" / "agent_original.py")],
                       "env": {"BENCH_HTTP_RETRIES": "8"},
                       "note": "assignment script + HTTP retries on 429/5xx (the only change)"},
    # the naive way to add dozens of tools and skills: everything in the prompt, nothing managed
    "eager": {"cmd": HARNESS + ["--tool-mode", "eager", "--no-compaction", "--no-cap-tool-output"],
              "note": "all 39 tool schemas + 10 skill bodies up front, no compaction, no output cap"},
    "eager-managed": {"cmd": HARNESS + ["--tool-mode", "eager", "--compaction", "--cap-tool-output"],
                      "note": "eager tools and skills + output cap + compaction (loading control for full)"},
    "append": {"cmd": HARNESS + ["--tool-mode", "append", "--compaction", "--cap-tool-output"],
               "note": "lazy tools appended to `tools` (ablation: cache-unfriendly loading)"},
    "full-nocompact": {"cmd": HARNESS + ["--tool-mode", "dispatch", "--no-compaction", "--cap-tool-output"],
                       "note": "dispatch loading + output cap, compaction off (control for compaction)"},
    "full": {"cmd": HARNESS + ["--tool-mode", "dispatch", "--compaction", "--cap-tool-output"],
             "note": "dispatch loading + lazy skills + output cap + compaction"},
    "full-no-cap": {"cmd": HARNESS + ["--tool-mode", "dispatch", "--compaction", "--no-cap-tool-output"],
                    "note": "dispatch loading + compaction, output cap off (output-cap control for full)"},
}


def run_one(variant: str, index: int, root: Path, args) -> dict:
    run_dir = root / variant / str(index)
    workspace = make_workspace(run_dir / "workspace", args.seed)
    spec = VARIANTS[variant]
    cmd = list(spec["cmd"])
    env = {**os.environ, "RUN_DIR": str(run_dir), "PYTHONPATH": str(REPO_ROOT),
           "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
           "MAX_TURNS": str(args.max_turns), "CONTEXT_WINDOW": str(args.window),
           "ENFORCE_WINDOW": "1" if args.enforce_window else "0",
           "LLM_MAX_RETRIES": os.environ.get("LLM_MAX_RETRIES", "8"), **spec.get("env", {})}
    for option, variable in (("max_api_attempts", "MAX_API_ATTEMPTS"), ("max_wall_seconds", "MAX_WALL_SECONDS")):
        value = getattr(args, option, None)
        if value is not None:
            env[variable] = str(value)
    if args.docker:
        # Same variants, but inside the sandbox container: only this run's workspace and run directory
        # are mounted, and the checker below runs in a second container without network or key.
        original = variant.startswith("original")
        cmd = sandbox.agent_command(workspace, run_dir, env, [] if original else cmd[len(HARNESS):], TASK,
                                    entrypoint=["python", "-m", "bench.sandbox.baseline"] if original else None)
    elif variant.startswith("original"):
        cmd.append(TASK)
    else:
        cmd += ["--run-dir", str(run_dir), "--workdir", str(workspace), TASK]
    started = time.time()
    with (run_dir / "conversation.log").open("w", encoding="utf-8") as log:
        try:
            exit_code = subprocess.run(cmd, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            exit_code = -1
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metrics = {"status": "timeout" if exit_code == -1 else "crashed"}
    if args.docker:
        proc = subprocess.run(sandbox.checker_command(workspace, run_dir, args.seed), capture_output=True, text=True)
        try:
            verdict = json.loads(proc.stdout)
        except json.JSONDecodeError:
            verdict = {"success": False, "passed": 0, "total": 12, "checks": {"checker_crashed": False}}
    else:
        verdict = check(workspace, args.seed)
    usage = metrics.get("tool_usage") or {}
    (run_dir / "check.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    row = {"variant": variant, "run": index, "exit_code": exit_code, "wall_seconds": round(time.time() - started, 1),
           **{k: metrics.get(k) for k in ("status", "error", "turns", "requests", "api_attempts", "prompt_tokens", "completion_tokens",
                                          "cached_tokens", "peak_prompt_tokens", "prefix_reuse",
                                          "uncached_prompt_tokens_est", "tool_calls", "tool_errors", "intercepted")},
           "compactions": len(metrics.get("compactions") or []),
           "summaries": sum(c.get("level") == "summary" for c in metrics.get("compactions") or []),
           "elisions": sum(c.get("level") == "elide" for c in metrics.get("compactions") or []),
           "skill_loads": usage.get("load_skill", 0), "tool_searches": usage.get("tool_search", 0),
           "catalog_calls": sum(n for name, n in usage.items() if name not in CORE_AND_META),
           "llm_retries": (metrics.get("intercepted") or {}).get("llm_retry", metrics.get("http_retries", 0)),
           "checks_passed": verdict["passed"], "checks_total": verdict["total"], "success": verdict["success"],
           "failed_checks": [name for name, ok in verdict["checks"].items() if not ok]}
    print(f"[{variant} #{index}] status={row['status']} checks={row['checks_passed']}/{row['checks_total']} "
          f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} turns={row['turns']} "
          f"{row['wall_seconds']}s", flush=True)
    return row


def _values(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]


def _mid(rows: list[dict], key: str, pattern: str = "{:,.0f}") -> str:
    """Median with the range, e.g. `241,064 (198,112..301,560)`."""
    values = _values(rows, key)
    if not values:
        return "n/a"
    if len(values) == 1 or min(values) == max(values):
        return pattern.format(statistics.median(values))
    return f"{pattern.format(statistics.median(values))} ({pattern.format(min(values))}..{pattern.format(max(values))})"


def summarise(rows: list[dict], variants: list[str]) -> str:
    head = ("| variant | N | success | checks | prompt tokens | cached by server | reusable prefix | completion tokens | "
            "peak prompt | turns | compactions | elide / summary | skill loads | catalog calls | LLM retries | statuses |\n"
            + "|---" * 16 + "|\n")
    body = ""
    for variant in variants:
        mine = [r for r in rows if r["variant"] == variant]
        if not mine:
            continue
        statuses = ", ".join(f"{s} {sum(str(r['status']) == s for r in mine)}" for s in sorted({str(r['status']) for r in mine}))
        prompt, cached = sum(_values(mine, "prompt_tokens")), sum(_values(mine, "cached_tokens"))
        body += (f"| {variant} | {len(mine)} | {sum(r['success'] for r in mine)}/{len(mine)} | "
                 f"{_mid(mine, 'checks_passed')} / {mine[0]['checks_total']} | {_mid(mine, 'prompt_tokens')} | "
                 f"{cached / prompt:.0%} | " if prompt else f"| {variant} | {len(mine)} | {sum(r['success'] for r in mine)}/{len(mine)} | {_mid(mine, 'checks_passed')} / {mine[0]['checks_total']} | n/a | n/a | ")
        body += (f"{_mid(mine, 'prefix_reuse', '{:.0%}')} | {_mid(mine, 'completion_tokens')} | "
                 f"{_mid(mine, 'peak_prompt_tokens')} | {_mid(mine, 'turns')} | {_mid(mine, 'compactions')} | "
                 f"{sum(_values(mine, 'elisions'))} / {sum(_values(mine, 'summaries'))} | "
                 f"{sum(_values(mine, 'skill_loads'))} | {sum(_values(mine, 'catalog_calls'))} | "
                 f"{sum(_values(mine, 'llm_retries'))} | {statuses} |\n")
    return head + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="original,original-retry,eager,full", help=f"Comma-separated: {', '.join(VARIANTS)}")
    parser.add_argument("--runs", type=int, default=5, help="Runs per variant.")
    parser.add_argument("--jobs", type=int, choices=(1,), default=1, help="One run at a time per API key.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window", type=int, default=128_000, help="Context window in tokens.")
    parser.add_argument("--enforce-window", action="store_true", help="Treat the window as a hard limit.")
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per run.")
    parser.add_argument("--max-api-attempts", type=int, help="Per-harness-run API budget including retries/summaries; 0 disables.")
    parser.add_argument("--max-wall-seconds", type=float, help="Per-harness-run cooperative deadline; 0 disables.")
    parser.add_argument("--label", default="", help="Suffix for the results file name.")
    parser.add_argument("--docker", action="store_true",
                        help="Run every agent and the checker inside the Docker sandbox (see bench/sandbox/docker.py).")
    args = parser.parse_args()
    if args.runs < 1 or args.max_turns < 1 or args.timeout < 1:
        parser.error("runs, max-turns and timeout must be positive")

    try:
        cfg = Config.from_env(max_api_attempts=args.max_api_attempts, max_wall_seconds=args.max_wall_seconds,
                              context_window=args.window, enforce_window=args.enforce_window, max_turns=args.max_turns)
        # Load .env first, then validate the effective retry default used by child processes.
        cfg = replace(cfg, max_retries=int(os.environ.get("LLM_MAX_RETRIES", "8")))
    except ValueError as exc:
        parser.error(f"invalid configuration: {exc}")
    if not cfg.model:
        print("LLM_MODEL is not set - copy .env.example to .env and fill it in.", file=sys.stderr)
        return 2
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    if not variants:
        parser.error("select at least one variant")

    try:
        with series_lock(cfg.api_key):
            return run_series(args, cfg, variants)
    except (SourceChangedError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def run_series(args, cfg: Config, variants: list[str]) -> int:
    frozen = source_manifest(REPO_ROOT)
    runtime = {"runner": runtime_metadata()}

    if args.docker:
        try:
            sandbox.host_identity()
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not sandbox.docker_available():
            print("Docker is not running - start it or drop --docker.", file=sys.stderr)
            return 2
        sandbox.build_image(REPO_ROOT)  # cached layers make this quick; it guarantees the image matches the code
        runtime["docker_images"] = {
            role: subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
            for role, image in (("agent", sandbox.IMAGE), ("checker", sandbox.IMAGE + "-checker"))
        }
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}" + (f"-{args.label}" if args.label else "")
    root = REPO_ROOT / "runs" / stamp
    root.mkdir(parents=True)
    settings = {"model": cfg.model, "sandbox": "docker" if args.docker else "none", "runs": args.runs,
                "seed": args.seed, "window": args.window,
                "enforce_window": args.enforce_window, "max_turns": args.max_turns, "timeout": args.timeout,
                "max_output_tokens": cfg.max_output_tokens, "temperature": cfg.temperature,
                "source_sha256": frozen["digest"], "runtime": runtime, "config": safe_config(cfg),
                "variants": {name: {"args": VARIANTS[name]["cmd"][len(HARNESS):],
                                     "env": VARIANTS[name].get("env", {})} for name in variants}}
    (root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (root / "source-manifest.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    jobs = [(variant, index) for index in range(1, args.runs + 1) for variant in variants]
    rows = []
    for variant, index in jobs:
        before = verify_source(REPO_ROOT, frozen)
        run_dir = root / variant / str(index)
        run_dir.mkdir(parents=True)
        (run_dir / "source-before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
        try:
            row = run_one(variant, index, root, args)
        finally:
            after = source_manifest(REPO_ROOT)
            (run_dir / "source-after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
            verify_source(REPO_ROOT, frozen)
        row["source_sha256"] = frozen["digest"]
        rows.append(row)
        with (root / "rows.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")
    table = summarise(rows, variants)
    notes = "\n".join(f"- `{v}`: {VARIANTS[v]['note']}" for v in variants)
    report = (f"# Benchmark {stamp}\n\nSettings: `{json.dumps(settings)}`\n\n{table}\n{notes}\n\n"
              "Cells are the median over runs with (min..max). Token counts come from the server's `usage`. "
              "`cached by server` = cached_tokens / prompt_tokens over all runs. `reusable prefix` = share of each "
              "request that is byte-identical to the start of the previous one (upper bound on cache hits). "
              "`elide / summary`, `skill loads`, `catalog calls` and `LLM retries` are totals over all runs of the variant. "
              "`elide` = level-1 compactions (old outputs stubbed, no model call); `summary` = level-2.\n")
    results_dir = REPO_ROOT / "bench" / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{stamp}.md").write_text(report, encoding="utf-8")
    (results_dir / f"{stamp}.json").write_text(json.dumps({"settings": settings, "rows": rows}, indent=2), encoding="utf-8")
    print("\n" + report + f"\nArtifacts: {root}\nResults: {results_dir / (stamp + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
