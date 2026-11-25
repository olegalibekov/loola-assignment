"""The benchmark must be trustworthy: solvable, checked strictly, and deterministic."""
import os
import sys
from pathlib import Path

import pytest

from bench.task.check import check, reference
from bench.runners.replay import VARIANTS, replay, reference_steps, script_for
from bench.task.workspace import build_data, make_workspace


@pytest.fixture(autouse=True)
def python_on_path(monkeypatch):
    # the reference solution shells out to `python -m inventory.cli`
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")


def test_data_is_deterministic_per_seed():
    assert build_data(7) == build_data(7)
    assert build_data(7)["logs/app.log"] != build_data(8)["logs/app.log"]
    assert reference(7)["incident"] != reference(8)["incident"]


def test_log_has_exactly_one_fatal_and_many_decoys():
    log = build_data(7)["logs/app.log"].splitlines()
    assert len(log) == 3000 and sum(" FATAL " in line for line in log) == 1
    assert sum("incident=" in line for line in log) > 50


def test_untouched_workspace_fails_every_check(tmp_path):
    verdict = check(make_workspace(tmp_path / "ws"))
    assert verdict["passed"] == 0 and not verdict["success"]


def run(name, window=1_000_000, enforce=False):
    overrides, lookup = VARIANTS[name]
    return replay(overrides, window, enforce, lookup)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_reference_solution_passes_in_every_mode(name):
    result = run(name)
    assert result["status"] == "finished" and result["checks_passed"] == result["checks_total"] == 12
    assert not result["tool_errors"]


def test_after_survives_a_window_that_kills_before():
    before = run("eager, no cap, no compaction (before)", window=12_000, enforce=True)
    after = run("dispatch + cap + compaction (after)", window=12_000, enforce=True)
    assert before["status"] == "context_overflow" and not before["success"]
    assert after["success"] and after["compactions"]
    assert after["peak_prompt_tokens"] <= 12_000


def test_dispatch_keeps_more_of_the_prefix_than_append():
    append = run("append + cap + compaction")
    dispatch = run("dispatch + cap + compaction, schema lookups")
    assert dispatch["prefix_reuse"] > append["prefix_reuse"] + 0.05
    assert dispatch["uncached_prompt_tokens_est"] < append["uncached_prompt_tokens_est"] / 2


def test_script_translation_per_mode():
    steps = reference_steps()

    def names(script):
        return [c["function"]["name"] for r in script for c in r.tool_calls]

    eager, dispatch = names(script_for("eager", steps)), names(script_for("dispatch", steps, lookup=False))
    assert "load_skill" not in eager and "use_tool" not in eager and "git_commit" in eager
    assert "git_commit" not in dispatch and "use_tool" in dispatch and "tool_search" not in dispatch


def test_skills_do_not_leak_task_specifics():
    """Skills are generic playbooks. Anything that names this benchmark's bugs, data or answers is a leak
    that would bias the comparison towards the variants that can see skills."""
    import re

    from harness.runtime.config import REPO_ROOT
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "skills").glob("*/SKILL.md"))).lower()
    banned = ["decimal", "round_half", "half-up", "floats?", "<=", "off-by-one", "cancelled", "fatal", "incident",
              "sku", "top_products", "pricing", "inventory", "parse_row", "stock", "revenue", "whitespace",
              "upper", "strip", "discount", "__pycache__", "ties?", "decoys?"]
    leaks = [word for word in banned if re.search(rf"(?<![a-z_]){word}(?![a-z_])", text)]
    assert not leaks, f"task-specific hints in skills: {leaks}"
