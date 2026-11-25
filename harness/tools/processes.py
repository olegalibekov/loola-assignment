"""Run tool processes with scrubbed environments, bounded output, and timeout cleanup."""
from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import time
from pathlib import Path

from .base import ToolError


_CHILD_ENV = {"PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb", "PYTHONUNBUFFERED": "1",
              "GIT_TERMINAL_PROMPT": "0"}
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
_GIT_CONFIG_ENTRY = re.compile(r"GIT_CONFIG_(?:KEY|VALUE)_\d+\Z")


def _child_git_config() -> dict[str, str]:
    """Keep Git's numbered config pairs intact without forwarding secret settings.

    GIT_CONFIG_KEY_n contains an option name, not a credential. Filtering just those
    variables leaves GIT_CONFIG_COUNT/values behind and prevents every Git command
    from starting. Rebuild the pairs together, retaining their order.
    """
    try:
        count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        return {}
    if count < 0 or count > len(os.environ):
        return {}
    result: dict[str, str] = {}
    kept = 0
    for index in range(count):
        key = os.environ.get(f"GIT_CONFIG_KEY_{index}")
        value = os.environ.get(f"GIT_CONFIG_VALUE_{index}")
        if key is None or value is None:
            return {}  # do not pass an incomplete config array to Git
        if (_SECRET_ENV.search(key) or key.lower().endswith(".extraheader")
                or re.search(r"://[^/\s]*@", key + " " + value)):
            continue
        result[f"GIT_CONFIG_KEY_{kept}"] = key
        result[f"GIT_CONFIG_VALUE_{kept}"] = value
        kept += 1
    if "GIT_CONFIG_COUNT" in os.environ:
        result["GIT_CONFIG_COUNT"] = str(kept)
    return result


def child_env() -> dict[str, str]:
    """Environment for commands the model runs: no harness settings, nothing that looks like a secret.

    This keeps the API key out of `env` / `printenv`; it is not a sandbox (see README, limitations).
    """
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("LLM_", "HARNESS_")) and not _SECRET_ENV.search(k)
             and k != "GIT_CONFIG_COUNT" and not _GIT_CONFIG_ENTRY.fullmatch(k)}
    return {**clean, **_child_git_config(), **_CHILD_ENV}


def child_identity() -> dict:
    """In the sandbox (harness is root, HARNESS_TOOL_UID set) commands run as an unprivileged user.

    A different uid cannot read /proc/<harness pid>/environ, which is where the API key lives.
    Outside the sandbox this returns nothing and commands run as the current user.
    """
    uid, gid = os.environ.get("HARNESS_TOOL_UID"), os.environ.get("HARNESS_TOOL_GID")
    if not uid or os.geteuid() != 0:
        return {}
    return {"user": int(uid), "group": int(gid or uid), "extra_groups": []}


PROCESS_OUTPUT_LIMIT = 1024 * 1024  # per stream, before the smaller model-facing registry cap
PROCESS_DRAIN_GRACE = 0.25
PROCESS_REAP_GRACE = 1.0


class _ProcessOutput:
    """Keep a bounded head and tail while consuming arbitrarily large output."""

    def __init__(self, limit: int):
        self.head_limit = limit * 3 // 5
        self.tail_limit = limit - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def append(self, data: bytes) -> None:
        self.total += len(data)
        take = min(len(data), self.head_limit - len(self.head))
        self.head.extend(data[:take])
        self.tail.extend(data[take:])
        if len(self.tail) > self.tail_limit:
            del self.tail[:-self.tail_limit]

    def text(self) -> str:
        omitted = self.total - len(self.head) - len(self.tail)
        marker = (f"\n[... {omitted} bytes omitted by process capture limit; "
                  "redirect output to a file to retain it ...]\n".encode() if omitted else b"")
        return (bytes(self.head) + marker + bytes(self.tail)).decode("utf-8", "replace")


def run_process(command, *, cwd: Path, timeout: int, shell: bool = False,
                capture_limit: int = PROCESS_OUTPUT_LIMIT) -> str:
    """Run a command in its own process group; kill the whole group on timeout.

    stdin is closed so interactive prompts fail fast instead of hanging; pagers are disabled.
    Capture is bounded in memory, and descendants holding inherited pipes cannot extend the timeout.
    """
    env = child_env()
    identity = child_identity()
    if identity:
        env["HOME"] = "/home/agent"
    try:
        proc = subprocess.Popen(command, shell=shell, cwd=cwd, env=env, **identity,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
    except FileNotFoundError as exc:
        raise ToolError("not_found", f"executable not found: {exc.filename}") from None
    stdout, stderr = _ProcessOutput(max(2, capture_limit)), _ProcessOutput(max(2, capture_limit))
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, stdout)
    selector.register(proc.stderr, selectors.EVENT_READ, stderr)
    deadline = time.monotonic() + timeout
    timed_out = False
    kill_error: OSError | None = None

    def kill_group() -> None:
        nonlocal kill_error
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            kill_error = exc

    try:
        while selector.get_map() or proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if timed_out:
                    break
                timed_out = True
                kill_group()
                deadline = time.monotonic() + PROCESS_DRAIN_GRACE
                continue
            if selector.get_map():
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 64 * 1024)
                    if chunk:
                        key.data.append(chunk)
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
            else:
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass  # the next iteration kills the group at the deadline
    finally:
        selector.close()
        proc.stdout.close()
        proc.stderr.close()
        if proc.poll() is None:
            kill_group()
        try:
            proc.wait(timeout=PROCESS_REAP_GRACE)
        except subprocess.TimeoutExpired:
            # Never wait indefinitely on an uninterruptible child. Popen retains it for later reaping.
            pass
    out, err = stdout.text(), stderr.text()
    if kill_error is not None:
        raise ToolError("execution_error", f"could not stop command process group {proc.pid}: {kill_error}. "
                        "The command may still be running and may already have changed files.",
                        hint="Stop the remaining process and check its effects before retrying; "
                        "the sandbox must permit signaling tool processes.") from None
    if timed_out:
        partial = ((out or "") + (err or ""))[-1500:]
        raise ToolError(
            "timeout", f"command timed out after {timeout}s; the process group was killed. It may already have "
                       "changed files - check the working directory before running it again.",
            hint="Use a narrower command, pass a larger `timeout`, or start long jobs in the background "
                 "(`nohup cmd > log 2>&1 &`) and poll the log."
                 + (f"\nLast output before the kill:\n{partial}" if partial.strip() else ""),
        ) from None
    text = f"Exit code: {proc.returncode}\n{out}"
    if err:
        text += f"\nSTDERR:\n{err}"
    return text
