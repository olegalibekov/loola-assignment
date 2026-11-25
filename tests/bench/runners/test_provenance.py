import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.runners.provenance import SourceChangedError, safe_config, series_lock, source_manifest, verify_source
from harness.runtime.config import Config


def sources(root):
    for name in ("harness/runtime/loop.py", "skills/example/SKILL.md", "bench/task/check.py", "bench/task/workspace.py", "baseline/agent_original.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")


def test_manifest_is_stable_and_ignores_secrets_and_results(tmp_path):
    sources(tmp_path)
    first = source_manifest(tmp_path)
    (tmp_path / ".env").write_text("LLM_API_KEY=not-a-real-secret")
    (tmp_path / "bench/results").mkdir()
    (tmp_path / "bench/results/result.json").write_text("{}")
    assert source_manifest(tmp_path) == first
    assert ".env" not in first["files"] and len(first["files"]) == 5


@pytest.mark.parametrize("change", ["edit", "add", "delete"])
def test_source_guard_detects_all_source_changes(tmp_path, change):
    sources(tmp_path)
    frozen = source_manifest(tmp_path)
    path = tmp_path / "bench/task/check.py"
    if change == "edit":
        path.write_text("changed")
    elif change == "add":
        (tmp_path / "harness/runtime/new.py").write_text("new")
    else:
        path.unlink()
    with pytest.raises(SourceChangedError, match="source changed"):
        verify_source(tmp_path, frozen)


def test_config_metadata_never_serializes_credentials():
    cfg = Config(api_key="private-api-secret", base_url="https://user:password@example.com/v1?token=private",
                 extra_body={"authorization": "private-extension-secret"})
    metadata = safe_config(cfg)
    text = json.dumps(metadata)
    assert all(secret not in text for secret in (cfg.api_key, "password", "private", "authorization"))
    assert len(metadata["endpoint_sha256"]) == len(metadata["extra_body_sha256"]) == 64


def test_key_lock_excludes_same_key_allows_others_and_releases(tmp_path):
    with series_lock("test-key", tmp_path):
        with pytest.raises(RuntimeError, match="Another local benchmark") as error:
            with series_lock("test-key", tmp_path):
                pytest.fail("overlapping series")
        assert "test-key" not in str(error.value)
        with series_lock("different-key", tmp_path):
            pass
    with series_lock("test-key", tmp_path):
        pass
    assert all("test-key" not in path.name and path.read_bytes() == b"" for path in tmp_path.iterdir())


def test_series_writes_preflight_and_stops_after_source_drift(tmp_path, monkeypatch):
    from bench.runners import model as run

    sources(tmp_path)
    monkeypatch.setattr(run, "REPO_ROOT", tmp_path)
    calls = []

    def fake_run(variant, index, root, args):
        calls.append((variant, index))
        assert (root / "settings.json").is_file()
        assert (root / variant / str(index) / "source-before.json").is_file()
        (tmp_path / "harness/runtime/loop.py").write_text("changed during run")
        return {}

    monkeypatch.setattr(run, "run_one", fake_run)
    args = SimpleNamespace(docker=False, label="test", runs=2, seed=7, window=16000,
                           enforce_window=True, max_turns=10, timeout=60)
    with pytest.raises(SourceChangedError):
        run.run_series(args, Config(model="test"), ["full"])
    assert calls == [("full", 1)]
    assert len(list((tmp_path / "runs").glob("*/full/1/source-after.json"))) == 1
    assert not (tmp_path / "bench/results").exists()


def test_control_variants_change_exactly_one_harness_option():
    from bench.runners.model import HARNESS, VARIANTS
    from harness.cli import build_parser

    parser = build_parser()
    def options(name):
        parsed = parser.parse_args(VARIANTS[name]["cmd"][len(HARNESS):])
        return {key: getattr(parsed, key) for key in ("tool_mode", "compaction", "cap_tool_output")}
    full = options("full")
    assert full == {"tool_mode": "dispatch", "compaction": True, "cap_tool_output": True}
    assert options("eager-managed") == {**full, "tool_mode": "eager"}
    assert options("full-no-cap") == {**full, "cap_tool_output": False}
    assert options("full-nocompact") == {**full, "compaction": False}


def test_benchmark_forwards_harness_run_budgets(tmp_path, monkeypatch):
    import subprocess
    from bench.runners import model as run

    verdict = {"success": False, "passed": 0, "total": 12, "checks": {}}
    def workspace(path, seed):
        path.mkdir(parents=True)
        return path
    seen = {}
    def execute(command, **kwargs):
        seen.update(kwargs["env"])
        (Path(seen["RUN_DIR"]) / "metrics.json").write_text(json.dumps({"requests": 10, "api_attempts": 17}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(run, "make_workspace", workspace)
    monkeypatch.setattr(run, "check", lambda *a: verdict)
    monkeypatch.setattr(subprocess, "run", execute)
    args = SimpleNamespace(docker=False, seed=7, window=16000, enforce_window=True, max_turns=10,
                           timeout=60, max_api_attempts=17, max_wall_seconds=45.0)
    row = run.run_one("full", 1, tmp_path, args)
    assert seen["MAX_API_ATTEMPTS"] == "17" and seen["MAX_WALL_SECONDS"] == "45.0"
    assert row["requests"] == 10 and row["api_attempts"] == 17


@pytest.mark.parametrize("flags,retries,diagnostic", [
    (["--max-api-attempts", "-1"], "8", "max_api_attempts"),
    (["--max-wall-seconds", "nan"], "8", "max_wall_seconds"),
    ([], "-1", "max_retries"),
    (["--window", "100"], "8", "context_window"),
])
def test_benchmark_invalid_config_is_an_argparse_error(monkeypatch, capsys, flags, retries, diagnostic):
    import sys
    from bench.runners import model as run

    monkeypatch.setattr("harness.runtime.config.load_dotenv", lambda *_: None)
    monkeypatch.setenv("LLM_MODEL", "test")
    monkeypatch.setenv("LLM_MAX_RETRIES", retries)
    monkeypatch.setattr(sys, "argv", ["bench.run", *flags])
    monkeypatch.setattr(run, "run_series", lambda *a: pytest.fail("invalid config must stop before the series"))
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert diagnostic in stderr and "Traceback" not in stderr
