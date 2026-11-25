import os
import signal
import sys
import time

import pytest

from harness.tools.base import ToolError
from harness.tools.processes import run_process


def test_large_process_output_preserves_head_tail_with_bounded_capture(tmp_path):
    code = """
import os
for fd, prefix in [(1, b'OUT'), (2, b'ERR')]:
    os.write(fd, prefix + b'_HEAD\\n')
    for _ in range(128):
        os.write(fd, b'x' * 65536)
    os.write(fd, b'\\n' + prefix + b'_TAIL\\n')
"""
    output = run_process([sys.executable, "-c", code], cwd=tmp_path, timeout=10, capture_limit=16384)
    assert output.startswith("Exit code: 0\nOUT_HEAD\n")
    assert "OUT_TAIL" in output and "ERR_HEAD" in output and output.endswith("ERR_TAIL\n")
    assert output.count("bytes omitted by process capture limit") == 2
    assert len(output) < 34000


def test_timeout_does_not_wait_for_escaped_descendant_pipes(tmp_path):
    pid_file = tmp_path / "escaped.pid"
    # The descendant has a finite lifetime and is also explicitly cleaned up in the test.
    code = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], start_new_session=True)
with open(sys.argv[1], 'w') as stream:
    stream.write(str(child.pid))
print('partial before timeout', flush=True)
time.sleep(10)
"""
    started = time.monotonic()
    try:
        with pytest.raises(ToolError) as exc:
            run_process([sys.executable, "-c", code, str(pid_file)], cwd=tmp_path, timeout=0.5)
        assert time.monotonic() - started < 3
        assert exc.value.kind == "timeout"
        assert "may already have changed files" in exc.value.message
        assert "partial before timeout" in exc.value.hint
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_process_without_open_pipes_is_still_timed_out(tmp_path):
    code = "import os, time; os.close(1); os.close(2); time.sleep(10)"
    started = time.monotonic()
    with pytest.raises(ToolError, match="timed out"):
        run_process([sys.executable, "-c", code], cwd=tmp_path, timeout=0.2)
    assert time.monotonic() - started < 3


def test_signal_permission_error_is_a_bounded_tool_error(tmp_path, monkeypatch):
    def denied(pid, signum):
        raise PermissionError("signaling denied")

    monkeypatch.setattr("harness.tools.processes.os.killpg", denied)
    started = time.monotonic()
    with pytest.raises(ToolError) as exc:
        run_process([sys.executable, "-c", "import time; time.sleep(0.5)"], cwd=tmp_path, timeout=0.1)
    assert time.monotonic() - started < 3
    assert exc.value.kind == "execution_error" and "could not stop" in exc.value.message
    assert "may already have changed files" in exc.value.message and exc.value.hint
