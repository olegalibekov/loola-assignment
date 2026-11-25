"""Run the unchanged assignment loop with Docker's tool identity and environment.

Only shell execution is adapted. The original prompt, loop, HTTP behavior, output format,
and timeout handling are retained so the baseline remains comparable.
"""
from __future__ import annotations

import subprocess
import sys

from harness.tools.processes import child_env, child_identity


def tool_identity() -> dict:
    identity = child_identity()
    if not identity or identity.get("user", 0) <= 0:
        raise ValueError("The baseline sandbox requires a non-root HARNESS_TOOL_UID and a root launcher.")
    return identity


def run_bash(command: str) -> str:
    identity = tool_identity()
    env = {**child_env(), "HOME": "/home/agent"}
    try:
        command_result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120,
                                        env=env, **identity)
        out = command_result.stdout + (f"\nSTDERR:\n{command_result.stderr}" if command_result.stderr else "")
        return f"Exit code: {command_result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"


def main(argv: list[str] | None = None) -> int:
    try:
        tool_identity()  # fail before the first model request if privilege separation is unavailable
    except (TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from baseline import agent_original as baseline

    baseline.run_bash = run_bash
    prompt = " ".join(sys.argv[1:] if argv is None else argv)
    if not prompt.strip():
        print("No task provided. Exiting.")
        return 1
    try:
        baseline.agent_loop(prompt)
    except OverflowError as exc:
        baseline.METRICS["status"], baseline.METRICS["error"] = "context_overflow", str(exc)
    except Exception as exc:
        baseline.METRICS["status"], baseline.METRICS["error"] = "crashed", f"{type(exc).__name__}: {exc}"[:300]
        print(f"\n💥 {baseline.METRICS['error']}")
    baseline.write_metrics()
    return 0 if baseline.METRICS["status"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
