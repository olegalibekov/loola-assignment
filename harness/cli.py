"""Command line entry point: python -m harness "task" [options]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runtime.config import TOOL_MODES, Config
from .runtime.loop import Agent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harness", description="Minimal coding agent with dynamic tools and compaction.")
    p.add_argument("task", nargs="*", help="Task for the agent (or use --task-file).")
    p.add_argument("--task-file", type=Path, help="Read the task from a file.")
    p.add_argument("--workdir", type=Path, help="Directory the agent works in (default: cwd).")
    p.add_argument("--run-dir", type=Path, help="Where to write trace.jsonl, metrics.json and spilled outputs.")
    p.add_argument("--tool-mode", choices=TOOL_MODES, help="How tool schemas reach the model (default: dispatch).")
    p.add_argument("--compaction", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cap-tool-output", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--context-window", type=int, help="Model context window in tokens.")
    p.add_argument("--enforce-window", action=argparse.BooleanOptionalAction, default=None,
                   help="Refuse to send prompts larger than the window (simulate a small n_ctx).")
    p.add_argument("--max-turns", type=int)
    p.add_argument("--max-api-attempts", type=int, help="Total API attempts, including retries/summaries; 0 disables.")
    p.add_argument("--max-wall-seconds", type=float, help="Cooperative run deadline in seconds; 0 disables.")
    p.add_argument("--quiet", action="store_true", help="Do not echo the conversation.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = args.task_file.read_text(encoding="utf-8") if args.task_file else " ".join(args.task)
    if not task.strip():
        print("No task provided. Exiting.", file=sys.stderr)
        return 2
    try:
        cfg = Config.from_env(workdir=args.workdir, run_dir=args.run_dir, tool_mode=args.tool_mode,
                              compaction=args.compaction, cap_tool_output=args.cap_tool_output,
                              context_window=args.context_window, enforce_window=args.enforce_window,
                              max_turns=args.max_turns, max_api_attempts=args.max_api_attempts,
                              max_wall_seconds=args.max_wall_seconds)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    if not cfg.model:
        print("LLM_MODEL is not set (see .env.example).", file=sys.stderr)
        return 2
    result = Agent(cfg, echo=None if args.quiet else print).run(task)
    m = result.metrics
    print(f"\nstatus={result.status} turns={m['turns']} requests={m['requests']} prompt_tokens={m['prompt_tokens']} "
          f"completion_tokens={m['completion_tokens']} cached_tokens={m['cached_tokens']} "
          f"prefix_reuse={m['prefix_reuse']:.0%} compactions={len(m['compactions'])}", file=sys.stderr)
    if result.status != "finished" and m.get("error"):
        print(f"error: {m['error']}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
