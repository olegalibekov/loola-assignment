"""Docker sandbox for benchmark runs: command builders and image management.

Two kinds of containers, both from the image built by the repo's Dockerfile:

  agent   - has the network (it must reach the LLM endpoint) and the API key, but only in the
            harness process. Model-issued commands run as the host uid with a scrubbed
            environment; in-process file tools are confined to /workspace and /run.
  checker - no network, no key, runs as the host uid. It executes code the agent wrote, so it
            gets nothing worth stealing and nowhere to send it.

Only the per-run workspace and run directory are mounted; the repository and `.env` are not.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):  # started as a file: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMAGE = "loola-harness-sandbox"
# settings forwarded into the agent container when they are set on the host
FORWARDED = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_EXTRA_BODY", "LLM_TIMEOUT",
             "LLM_MAX_RETRIES", "MAX_OUTPUT_TOKENS", "MAX_TURNS", "CONTEXT_WINDOW", "ENFORCE_WINDOW",
             "BENCH_HTTP_RETRIES", "MAX_API_ATTEMPTS", "MAX_WALL_SECONDS")


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def image_exists(image: str = IMAGE) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def build_image(root: Path, image: str = IMAGE) -> None:
    subprocess.run(["docker", "build", "-q", "--target", "agent", "-t", image, str(root)], check=True)
    subprocess.run(["docker", "build", "-q", "--target", "checker", "-t", image + "-checker", str(root)], check=True)


def _mounts(workspace: Path, run_dir: Path) -> list[str]:
    return ["-v", f"{workspace}:/workspace", "-v", f"{run_dir}:/run"]


def host_identity() -> tuple[int, int]:
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("Run the Docker benchmark as a non-root host user; UID 0 cannot isolate tool commands from the API key.")
    return uid, gid


def agent_command(workspace: Path, run_dir: Path, env: dict[str, str], harness_args: list[str], task: str,
                  entrypoint: list[str] | None = None, image: str = IMAGE) -> list[str]:
    """`docker run` for one agent run.

    Secrets are passed as `-e NAME` (value taken from the caller's environment), never on the
    command line, so they do not show up in `ps` or in the recorded command.
    """
    uid, gid = host_identity()
    cmd = ["docker", "run", "--rm", "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
           "--cap-add", "DAC_OVERRIDE", "--cap-add", "KILL", "--security-opt", "no-new-privileges", "--pids-limit", "512",
           "--memory", "2g", *_mounts(workspace, run_dir),
           "-e", f"HARNESS_TOOL_UID={uid}", "-e", f"HARNESS_TOOL_GID={gid}", "-e", "RUN_DIR=/run"]
    for name in FORWARDED:
        if name in env:
            cmd += ["-e", name]
    if entrypoint:
        cmd += ["--entrypoint", entrypoint[0], image, *entrypoint[1:], task]
    else:
        cmd += [image, *harness_args, task]
    return cmd


def checker_command(workspace: Path, run_dir: Path, seed: int, image: str = IMAGE) -> list[str]:
    uid, gid = host_identity()
    return ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{uid}:{gid}", "-e", "HOME=/tmp", *_mounts(workspace, run_dir),
            "--entrypoint", "python", image + "-checker", "-m", "bench.task.check", "/workspace", "--seed", str(seed)]


# Runs inside the container (as the harness would: root, key in its environment) and tries every
# route the model has to the key. Prints one line per probe; exit code 1 if the key leaked anywhere.
PROBE = r"""
import json, os, sys
from pathlib import Path
from harness.runtime.config import Config
from harness.tools.registry import ToolRegistry
from harness.context.skills import SkillStore
from bench.sandbox.baseline import run_bash as baseline_bash

secret = os.environ["LLM_API_KEY"]
cfg = Config.from_env(workdir=Path("/workspace"), run_dir=Path("/run"))
reg = ToolRegistry(cfg, SkillStore(cfg.skills_dir))
probes = [
    ("bash: env", "bash", {"command": "env"}),
    ("bash: /proc/<harness>/environ", "bash", {"command": f"cat /proc/{os.getpid()}/environ /proc/1/environ 2>&1 | tr '\\0' '\\n'"}),
    ("bash: whoami", "bash", {"command": "id -u; id -g"}),
    ("bash: write to workspace", "bash", {"command": "echo ok > /workspace/by_bash.txt && cat /workspace/by_bash.txt"}),
    ("bash: timeout group", "bash", {"command": "sleep 30 & wait", "timeout": 1}),
    ("bash: hidden checker unavailable", "bash", {"command": "test ! -e /app/bench/check.py && test ! -e /app/bench/workspace.py && test ! -e /app/bench/replay.py && test ! -e /app/bench/task/check.py && test ! -e /app/bench/task/workspace.py && test ! -e /app/bench/runners/replay.py && echo isolated"}),
    ("read_file: /proc/self/environ", "read_file", {"path": "/proc/self/environ"}),
    ("read_file: /proc/1/environ", "read_file", {"path": "/proc/1/environ"}),
    ("grep: /proc", "grep", {"pattern": "LLM_API_KEY", "path": "/proc/self"}),
    ("bash: make nested proc link", "bash", {"command": "mkdir -p /workspace/link_probe && ln -s /proc/1/environ /workspace/link_probe/environ"}),
    ("grep: nested proc link", "grep", {"pattern": "LLM_API_KEY", "path": "/workspace/link_probe"}),
    ("copy_path: nested proc link", "copy_path", {"source": "/workspace/link_probe", "destination": "/workspace/probe_copy"}),
    ("env_get", "env_get", {"name": "LLM_API_KEY"}),
    ("run_python: os.environ", "run_python", {"code": "import os; print(sorted(k for k in os.environ if 'LLM' in k or 'KEY' in k))"}),
    ("write_file then bash edit", "write_file", {"path": "by_tool.txt", "content": "a\n"}),
    ("bash: append to tool-written file", "bash", {"command": "echo b >> by_tool.txt && cat by_tool.txt"}),
]
leaked = False
for label, tool, args in probes:
    out = reg.execute(tool, args)
    hit = secret in out.text
    failed = label == "bash: timeout group" and getattr(out, "kind", None) != "timeout"
    failed = failed or (label == "bash: hidden checker unavailable" and out.text != "Exit code: 0\nisolated\n")
    leaked = leaked or hit or failed
    safe = out.text.replace(secret, "<redacted>")
    print(("LEAK " if hit else "FAIL " if failed else "ok   ") + label + " -> " + " ".join(safe.split())[:110])
for label, command in [
    ("baseline: env", "env"),
    ("baseline: /proc/<harness>/environ", f"cat /proc/{os.getpid()}/environ /proc/1/environ 2>&1 | tr '\\0' '\\n'"),
    ("baseline: uid", "id -u"),
    ("baseline: write to workspace", "echo ok > /workspace/by_baseline.txt && cat /workspace/by_baseline.txt"),
]:
    out = baseline_bash(command)
    hit = secret in out
    failed = label == "baseline: uid" and out != f"Exit code: 0\n{os.environ['HARNESS_TOOL_UID']}\n"
    failed = failed or (label == "baseline: write to workspace" and out != "Exit code: 0\nok\n")
    leaked = leaked or hit or failed
    safe = out.replace(secret, "<redacted>")
    print(("LEAK " if hit else "FAIL " if failed else "ok   ") + label + " -> " + " ".join(safe.split())[:110])
print("uid of harness:", os.geteuid())
sys.exit(1 if leaked else 0)
"""


def selfcheck(image: str = IMAGE) -> int:
    """Build the image and show that a model-issued command cannot reach the API key."""
    repo = Path(__file__).resolve().parents[2]
    try:
        uid, gid = host_identity()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not docker_available():
        print("Docker is not running.", file=sys.stderr)
        return 2
    build_image(repo, image)
    with tempfile.TemporaryDirectory() as tmp:
        work, run = Path(tmp) / "workspace", Path(tmp) / "run"
        work.mkdir()
        run.mkdir()
        cmd = ["docker", "run", "--rm", "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
               "--cap-add", "DAC_OVERRIDE", "--cap-add", "KILL", "--security-opt", "no-new-privileges", *_mounts(work, run),
               "-e", f"HARNESS_TOOL_UID={uid}", "-e", f"HARNESS_TOOL_GID={gid}",
               "-e", "LLM_API_KEY", "-e", "LLM_MODEL=probe", "--entrypoint", "python", image, "-c", PROBE]
        env = {**os.environ, "LLM_API_KEY": "sk-selfcheck-0123456789-not-a-real-key"}
        return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(selfcheck())
