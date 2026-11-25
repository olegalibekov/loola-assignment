import sys
import time

import pytest

from harness.tools.registry import ToolRegistry, cap_output
from harness.context.skills import SkillStore
from harness.tools import CATALOG_TOOLS, CORE_TOOLS, ToolError, parse_arguments, validate_arguments
from harness.tools.base import Tool, params


def registry(make_cfg, **overrides) -> ToolRegistry:
    cfg = make_cfg(**overrides)
    return ToolRegistry(cfg, SkillStore(cfg.skills_dir))


def test_catalog_is_large_and_names_are_unique():
    names = [t.name for t in CORE_TOOLS + CATALOG_TOOLS]
    assert len(names) == len(set(names)) and len(names) >= 30


# ---------------------------------------------------------------- argument parsing / validation

@pytest.mark.parametrize("raw, expected, repaired", [
    ('{"a": 1}', {"a": 1}, False),
    ({"a": 1}, {"a": 1}, False),
    ("", {}, False),
    (None, {}, False),
    ('```json\n{"a": 1}\n```', {"a": 1}, True),
    ('"{\\"a\\": 1}"', {"a": 1}, True),
])
def test_parse_arguments_repairs_unambiguous_input(raw, expected, repaired):
    assert parse_arguments(raw) == (expected, None, repaired)


@pytest.mark.parametrize("raw", ['{"a": 1', "[1, 2]", "not json"])
def test_parse_arguments_reports_real_errors(raw):
    args, error, _ = parse_arguments(raw)
    assert args is None and error


def _tool() -> Tool:
    return Tool("t", "d", params({"path": "string: p", "n?": "integer: n", "flag?": "boolean: f",
                                  "mode?": "enum[a|b]: m"}), lambda ctx, **kw: "ok")


def test_validate_coerces_lossless_types():
    clean, coerced = validate_arguments(_tool(), {"path": "x", "n": "5", "flag": "true", "mode": None})
    assert clean == {"path": "x", "n": 5, "flag": True} and coerced == 2


def test_validate_reports_every_problem_with_schema():
    with pytest.raises(ToolError) as exc:
        validate_arguments(_tool(), {"n": "five", "bogus": 1, "mode": "c"})
    text = exc.value.render()
    assert exc.value.kind == "invalid_arguments"
    for fragment in ("missing required argument 'path'", "'n' must be integer", "unexpected argument 'bogus'",
                     "one of a, b", "Schema:"):
        assert fragment in text


# ---------------------------------------------------------------- core tools

def test_bash_reports_exit_code_and_stderr(make_cfg):
    out = registry(make_cfg).execute("bash", {"command": "echo hi; echo oops >&2; exit 3"})
    assert out.ok and out.text.startswith("Exit code: 3") and "hi" in out.text and "STDERR:\noops" in out.text


def test_bash_timeout_kills_process_group(make_cfg):
    started = time.monotonic()
    out = registry(make_cfg).execute("bash", {"command": "sleep 30 & sleep 30", "timeout": 1})
    assert time.monotonic() - started < 10
    assert not out.ok and out.kind == "timeout" and "Hint:" in out.text


def test_bash_has_no_stdin(make_cfg):
    out = registry(make_cfg).execute("bash", {"command": "read line; echo got=$line", "timeout": 5})
    assert out.ok and "got=" in out.text


def test_bash_policy_denies_catastrophic_commands(make_cfg):
    from harness.tools.core import policy_violation  # checked as a pure function: nothing is executed
    for command in ("rm -rf /", "sudo rm -fr ~ ", "rm -r -f $HOME", "mkfs.ext4 /dev/sda1", "dd if=/dev/zero of=/dev/sda"):
        assert policy_violation(command), command
    for command in ("rm -rf build/", "rm -rf ./tmp /tmp/x", "echo reboot-required > f", "ls /"):
        assert policy_violation(command) is None, command
    out = registry(make_cfg).execute("bash", {"command": "echo mkfs.ext4"})  # harmless even if not blocked
    assert not out.ok and out.kind == "policy_denied"


def test_file_tools_roundtrip_and_errors(make_cfg):
    reg = registry(make_cfg)
    assert reg.execute("write_file", {"path": "a/b.txt", "content": "one\ntwo\ntwo\n"}).ok
    assert "    2\ttwo" in reg.execute("read_file", {"path": "a/b.txt"}).text
    missing = reg.execute("read_file", {"path": "a/c.txt"})
    assert missing.kind == "not_found" and "b.txt" in missing.text
    assert reg.execute("edit_file", {"path": "a/b.txt", "old_string": "two", "new_string": "2"}).kind == "ambiguous_match"
    assert reg.execute("edit_file", {"path": "a/b.txt", "old_string": "zzz", "new_string": "2"}).kind == "no_match"
    assert reg.execute("edit_file", {"path": "a/b.txt", "old_string": "one", "new_string": "1"}).ok
    assert (reg.ctx.workdir / "a/b.txt").read_text() == "1\ntwo\ntwo\n"


def test_tool_exception_becomes_error_result_not_a_crash(make_cfg):
    cfg = make_cfg()
    boom = Tool("boom", "d", params(), lambda ctx: 1 / 0)
    out = ToolRegistry(cfg, SkillStore(None), core=[boom], catalog=[]).execute("boom", {})
    assert not out.ok and out.kind == "execution_error" and "ZeroDivisionError" in out.text


# ---------------------------------------------------------------- output cap

def test_cap_output_keeps_head_and_tail_and_spills(tmp_path):
    text = "HEAD" + "x" * 50_000 + "TAIL"
    capped, was_capped = cap_output(text, 1000, tmp_path / "out" / "1.txt")
    assert was_capped and len(capped) < 1400
    assert capped.startswith("HEAD") and capped.endswith("TAIL") and "omitted by the harness" in capped
    assert (tmp_path / "out" / "1.txt").read_text() == text


def test_registry_caps_big_outputs_only_when_enabled(make_cfg):
    cmd = {"command": f"{sys.executable} -c \"print('y' * 40000)\""}
    capped = registry(make_cfg).execute("bash", cmd)
    assert "capped_output" in capped.intercepted and len(capped.text) < 9000
    raw = registry(make_cfg, cap_tool_output=False).execute("bash", cmd)
    assert not raw.intercepted and len(raw.text) > 40000


# ---------------------------------------------------------------- a few catalog tools

def test_catalog_tools_work(make_cfg):
    reg = registry(make_cfg, tool_mode="eager")
    work = reg.ctx.workdir
    (work / "d.csv").write_text("region,amount\nEU,10\nEU,5\nUS,1\n")
    (work / "app.log").write_text("t INFO a\nt ERROR b id=1\nt FATAL c id=2\nt INFO d\n")
    (work / "m.py").write_text("class A:\n    def f(self, x):\n        return x\n")
    assert "'EU': 15" in reg.execute("csv_aggregate", {"path": "d.csv", "group_by": "region", "value": "amount", "agg": "sum"}).text
    levels = reg.execute("log_levels", {"path": "app.log"}).text
    assert "FATAL: 1" in levels and "id=2" in levels
    assert "def f(self, x)" in reg.execute("py_outline", {"path": "m.py"}).text
    assert "app.log:3" in reg.execute("grep", {"pattern": "FATAL"}).text
    assert reg.execute("calc", {"expression": "(2 + 3) * 4"}).text.endswith("= 20")
    assert reg.execute("csv_aggregate", {"path": "d.csv", "group_by": "nope"}).kind == "invalid_arguments"


def test_secrets_and_harness_settings_do_not_reach_the_agent_shell(make_cfg, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok-secret-value")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    out = registry(make_cfg).execute("bash", {"command": "env"}).text
    assert "sk-secret-value" not in out and "tok-secret-value" not in out and "LLM_" not in out
    assert "HARMLESS_SETTING=visible" in out and "PAGER=cat" in out


def test_git_config_pairs_survive_secret_filtering(make_cfg, monkeypatch):
    import os
    from harness.tools.processes import child_env, run_process

    for name in list(os.environ):
        if name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "3")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.extraHeader")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Authorization: Bearer test-secret")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "Inspection Test")
    monkeypatch.setenv("GIT_CONFIG_KEY_2", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_2", "!echo password=test-secret")
    env = child_env()
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "user.name"
    assert env["GIT_CONFIG_VALUE_0"] == "Inspection Test"
    assert "GIT_CONFIG_KEY_1" not in env and "GIT_CONFIG_VALUE_1" not in env
    assert not any("test-secret" in value for value in env.values())
    out = run_process(["git", "config", "--get", "user.name"], cwd=make_cfg().workdir, timeout=5)
    assert out == "Exit code: 0\nInspection Test\n"


@pytest.mark.parametrize("count", ["invalid", "-1", "2"])
def test_incomplete_git_config_arrays_are_not_forwarded(monkeypatch, count):
    import os
    from harness.tools.processes import child_env

    for name in list(os.environ):
        if name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_COUNT", count)
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "orphan")
    env = child_env()
    assert "GIT_CONFIG_COUNT" not in env and "GIT_CONFIG_VALUE_0" not in env


def test_timeout_error_warns_about_partial_effects(make_cfg):
    out = registry(make_cfg).execute("bash", {"command": "sleep 5", "timeout": 1})
    assert out.kind == "timeout" and "may already have changed files" in out.text


def test_file_tools_can_be_confined_to_the_workspace(make_cfg, tmp_path):
    reg = registry(make_cfg, confine_file_tools=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    (reg.ctx.workdir / "inside.txt").write_text("fine")
    (reg.ctx.workdir / "link").symlink_to(outside)
    assert reg.execute("read_file", {"path": "inside.txt"}).ok
    for path in (str(outside), "../secret.txt", "link", "/proc/self/environ", "/etc/passwd"):
        out = reg.execute("read_file", {"path": path})
        assert out.kind == "policy_denied" and "top secret" not in out.text, path
    assert reg.execute("write_file", {"path": "../escape.txt", "content": "x"}).kind == "policy_denied"
    assert not (tmp_path / "escape.txt").exists()
    assert reg.execute("grep", {"pattern": "secret", "path": str(tmp_path)}).kind == "policy_denied"
    spill = reg.cfg.run_dir / "outputs" / "x.txt"  # truncated outputs live in the run dir and stay readable
    spill.parent.mkdir(parents=True)
    spill.write_text("spilled")
    assert "spilled" in reg.execute("read_file", {"path": str(spill)}).text


def test_file_tools_are_not_confined_by_default(make_cfg, tmp_path):
    (tmp_path / "note.txt").write_text("hello")
    assert "hello" in registry(make_cfg).execute("read_file", {"path": str(tmp_path / "note.txt")}).text


def test_commands_run_as_current_user_outside_the_sandbox(monkeypatch):
    from harness.tools.processes import child_identity
    monkeypatch.delenv("HARNESS_TOOL_UID", raising=False)
    assert child_identity() == {}
    monkeypatch.setenv("HARNESS_TOOL_UID", "1000")
    monkeypatch.setattr("os.geteuid", lambda: 501)  # not root: cannot switch users, so do not try
    assert child_identity() == {}
    monkeypatch.setattr("os.geteuid", lambda: 0)
    assert child_identity() == {"user": 1000, "group": 1000, "extra_groups": []}
