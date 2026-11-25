"""Offline checks for Docker command construction and the unchanged baseline's adapter."""
import importlib
import json
import os
import subprocess
import time
from types import SimpleNamespace

import pytest

from bench.sandbox import baseline as baseline_sandbox, docker as sandbox


@pytest.fixture
def sandbox_identity(monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("HARNESS_TOOL_UID", "501")
    monkeypatch.setenv("HARNESS_TOOL_GID", "20")


@pytest.mark.parametrize("method", ["agent", "checker", "selfcheck"])
def test_docker_rejects_root_host_before_starting_any_process(monkeypatch, tmp_path, method):
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must reject before Docker starts"))
    if method == "selfcheck":
        assert sandbox.selfcheck() == 2
    else:
        with pytest.raises(ValueError, match="non-root host user"):
            if method == "agent":
                sandbox.agent_command(tmp_path, tmp_path, {}, [], "task")
            else:
                sandbox.checker_command(tmp_path, tmp_path, 7)


def test_baseline_shell_drops_identity_and_scrubs_environment(monkeypatch, sandbox_identity):
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("OTHER_TOKEN", "test-other-token")
    monkeypatch.setenv("PUBLIC_SETTING", "keep")
    seen = {}

    def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 3, "output\n", "error\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert baseline_sandbox.run_bash("a command") == "Exit code: 3\noutput\n\nSTDERR:\nerror\n"
    assert seen["command"] == "a command"
    assert (seen["user"], seen["group"], seen["extra_groups"]) == (501, 20, [])
    assert "LLM_API_KEY" not in seen["env"] and "OTHER_TOKEN" not in seen["env"]
    assert not any(name.startswith("HARNESS_") for name in seen["env"])
    assert seen["env"]["PUBLIC_SETTING"] == "keep" and seen["env"]["HOME"] == "/home/agent"
    assert seen["timeout"] == 120 and seen["shell"] and seen["capture_output"] and seen["text"]


def test_baseline_shell_keeps_original_timeout_message(monkeypatch, sandbox_identity):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    assert baseline_sandbox.run_bash("slow command") == "Error: command timed out after 120s"


@pytest.mark.parametrize("uid", [None, "0", "-1"])
def test_baseline_launcher_rejects_missing_or_root_tool_identity(monkeypatch, uid):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    if uid is None:
        monkeypatch.delenv("HARNESS_TOOL_UID", raising=False)
    else:
        monkeypatch.setenv("HARNESS_TOOL_UID", uid)
    assert baseline_sandbox.main(["task"]) == 2


def test_baseline_launcher_keeps_original_loop_and_metrics(monkeypatch, sandbox_identity, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    baseline = importlib.import_module("baseline.agent_original")
    metrics = {"status": "running", "error": "", "turns": 0, "tool_calls": 0,
               "prompt_tokens": 0, "completion_tokens": 0, "started": time.time()}
    monkeypatch.setattr(baseline, "METRICS", metrics)
    monkeypatch.setattr(baseline, "run_bash", baseline.run_bash)  # restore the adapter assignment afterwards
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    calls = iter([
        ("", [{"id": "call-1", "type": "function", "function": {"name": "bash",
                                                                  "arguments": '{"command":"echo ok"}'}}]),
        ("Done.", []),
    ])
    requests = []

    def reply(messages):
        requests.append(json.loads(json.dumps(messages)))
        return next(calls)

    monkeypatch.setattr(baseline, "call_llm", reply)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "ok\n", ""))
    assert baseline_sandbox.main(["the", "task"]) == 0
    assert requests[0] == [{"role": "system", "content": baseline.SYSTEM_PROMPT}, {"role": "user", "content": "the task"}]
    assert requests[1][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "Exit code: 0\nok\n"}
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["status"] == "finished" and saved["turns"] == 2 and saved["tool_calls"] == 1


@pytest.mark.parametrize("variant", ["original", "original-retry"])
def test_docker_baselines_use_secure_launcher(monkeypatch, tmp_path, variant):
    from bench.runners import model as run

    monkeypatch.setattr(os, "getuid", lambda: 501)
    monkeypatch.setattr(os, "getgid", lambda: 20)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")

    def workspace(path, seed):
        path.mkdir(parents=True)
        return path

    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"success": False, "passed": 0, "total": 12, "checks": {}}), "")

    monkeypatch.setattr(run, "make_workspace", workspace)
    monkeypatch.setattr(subprocess, "run", execute)
    args = SimpleNamespace(seed=7, max_turns=5, window=16000, enforce_window=False, docker=True, timeout=60)
    run.run_one(variant, 1, tmp_path, args)
    agent = commands[0]
    assert agent[-3:] == ["-m", "bench.sandbox.baseline", run.TASK]
    assert "HARNESS_TOOL_UID=501" in agent and "HARNESS_TOOL_GID=20" in agent
    assert any(agent[i:i + 2] == ["--cap-add", "KILL"] for i in range(len(agent) - 1))
    assert "LLM_API_KEY" in agent and "test-api-key" not in agent
    assert "--network" in commands[1] and "none" in commands[1]
    assert sandbox.IMAGE + "-checker" in commands[1]


def test_builds_separate_agent_and_checker_images(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    sandbox.build_image(tmp_path, image="test-sandbox")
    assert len(commands) == 2
    assert commands[0][commands[0].index("--target") + 1] == "agent"
    assert commands[1][commands[1].index("--target") + 1] == "checker"
    assert commands[0][commands[0].index("-t") + 1] == "test-sandbox"
    assert commands[1][commands[1].index("-t") + 1] == "test-sandbox-checker"


def test_selfcheck_exercises_baseline_and_redacts_failed_probes(monkeypatch, capsys):
    from harness.runtime.config import Config
    from harness.tools import registry
    from harness.context import skills

    secret = "test-selfcheck-secret"
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setenv("HARNESS_TOOL_UID", "501")
    monkeypatch.setattr(Config, "from_env", lambda **kw: SimpleNamespace(skills_dir="unused"))
    monkeypatch.setattr(skills, "SkillStore", lambda *_: None)
    monkeypatch.setattr(registry, "ToolRegistry", lambda *_: SimpleNamespace(execute=lambda *a: SimpleNamespace(text=secret)))
    commands = []

    def baseline_bash(command):
        commands.append(command)
        if command == "id -u":
            return "Exit code: 0\n501\n"
        if command.startswith("echo ok"):
            return "Exit code: 0\nok\n"
        return secret

    monkeypatch.setattr(baseline_sandbox, "run_bash", baseline_bash)
    with pytest.raises(SystemExit) as result:
        exec(sandbox.PROBE, {})
    assert result.value.code == 1
    output = capsys.readouterr().out
    assert secret not in output and "<redacted>" in output
    assert "env" in commands and "id -u" in commands and len(commands) == 4
    assert "LEAK baseline: env" in output
