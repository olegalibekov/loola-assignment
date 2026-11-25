import json

from harness.context.prompt import build_system_prompt
from harness.tools.registry import LOAD_SKILL, TOOL_SEARCH, USE_TOOL, ToolRegistry
from harness.context.skills import SkillStore


def build(make_cfg, mode):
    cfg = make_cfg(tool_mode=mode)
    skills = SkillStore(cfg.skills_dir)
    return cfg, skills, ToolRegistry(cfg, skills)


def names(reg):
    return [t["function"]["name"] for t in reg.native_tools()]


def test_eager_exposes_everything_up_front(make_cfg):
    cfg, skills, reg = build(make_cfg, "eager")
    assert len(names(reg)) >= 30 and TOOL_SEARCH not in names(reg)
    assert "### Skill: log-triage" in build_system_prompt(cfg, reg, skills)


def test_dispatch_tools_array_never_changes(make_cfg):
    cfg, skills, reg = build(make_cfg, "dispatch")
    before = json.dumps(reg.native_tools())
    assert names(reg) == ["bash", "read_file", "write_file", "edit_file", TOOL_SEARCH, USE_TOOL, LOAD_SKILL]

    found = reg.execute(TOOL_SEARCH, {"query": "csv aggregate"})
    assert '"name": "csv_aggregate"' in found.text and '"parameters"' in found.text
    (reg.ctx.workdir / "d.csv").write_text("k,v\na,1\na,2\n")
    used = reg.execute(USE_TOOL, {"name": "csv_aggregate", "arguments": {"path": "d.csv", "group_by": "k"}})
    assert used.ok and used.tool == "csv_aggregate" and "'a': 2" in used.text
    reg.execute(LOAD_SKILL, {"name": "log-triage"})

    assert json.dumps(reg.native_tools()) == before  # byte-identical: the cached prefix is untouched


def test_append_mode_grows_tools_append_only(make_cfg):
    _, _, reg = build(make_cfg, "append")
    base = names(reg)
    reg.execute(TOOL_SEARCH, {"query": "grep"})
    first = names(reg)
    reg.execute(TOOL_SEARCH, {"query": "git_status git_commit"})
    second = names(reg)
    assert first == base + ["grep"] and second[:len(first)] == first and {"git_status", "git_commit"} <= set(second)


def test_system_prompt_is_static_and_lazy_prompt_is_small(make_cfg):
    cfg, skills, reg = build(make_cfg, "dispatch")
    prompt = build_system_prompt(cfg, reg, skills)
    reg.execute(TOOL_SEARCH, {"query": "grep"})
    reg.execute(LOAD_SKILL, {"name": "git-commit"})
    assert build_system_prompt(cfg, reg, skills) == prompt
    assert "- log-triage:" in prompt and "Never print a whole log" not in prompt  # index only, no bodies
    eager_cfg, eager_skills, eager_reg = build(make_cfg, "eager")
    eager = len(build_system_prompt(eager_cfg, eager_reg, eager_skills)) + len(json.dumps(eager_reg.native_tools()))
    lazy = len(prompt) + len(json.dumps(reg.native_tools()))
    assert lazy < eager / 3


def test_search_ranks_exact_names_first_and_handles_no_match(make_cfg):
    _, _, reg = build(make_cfg, "dispatch")
    assert [t.name for t in reg.search("git_commit grep")][:2] == ["git_commit", "grep"]
    assert reg.search("commit")[0].name == "git_commit"
    assert "No catalog tool matches" in reg.execute(TOOL_SEARCH, {"query": "kubernetes"}).text


def test_harness_routes_unambiguous_calls_instead_of_erroring(make_cfg):
    _, _, reg = build(make_cfg, "dispatch")
    native = reg.execute("calc", {"expression": "1+1"})  # catalog tool called natively without loading
    assert native.ok and "auto_routed" in native.intercepted
    flat = reg.execute(USE_TOOL, {"name": "calc", "expression": "1+1"})  # arguments not nested
    assert flat.ok and "flattened_arguments" in flat.intercepted
    stringly = reg.execute(USE_TOOL, {"name": "calc", "arguments": '{"expression": "1+1"}'})
    assert stringly.ok and "coerced_argument" in stringly.intercepted


def test_model_fixable_errors_come_back_with_hints(make_cfg):
    _, _, reg = build(make_cfg, "dispatch")
    unknown = reg.execute("grpe", {})
    assert unknown.kind == "unknown_tool" and "grep" in unknown.text
    bad = reg.execute(USE_TOOL, {"name": "calc", "arguments": {"expr": "1"}})
    assert bad.kind == "invalid_arguments" and "Schema:" in bad.text
    skill = reg.execute(LOAD_SKILL, {"name": "log-triag"})
    assert skill.kind == "unknown_skill" and "log-triage" in skill.text


def test_skill_is_loaded_once_until_compaction(make_cfg):
    _, skills, reg = build(make_cfg, "dispatch")
    first = reg.execute(LOAD_SKILL, {"name": "log-triage"}).text
    again = reg.execute(LOAD_SKILL, {"name": "log-triage"}).text
    assert "Never print a whole log" in first and "already loaded" in again and len(again) < 120
    assert "log-triage" in reg.pinned_state()
    skills.on_compaction()
    assert "Never print a whole log" in reg.execute(LOAD_SKILL, {"name": "log-triage"}).text


def test_unknown_tool_name_cannot_escape_output_directory(make_cfg):
    cfg = make_cfg(tool_output_chars=1024)
    reg = ToolRegistry(cfg, SkillStore(cfg.skills_dir))
    name = "nested/" + "../" * 8 + "x" * 2000
    result = reg.execute(name, {})
    assert result.kind == "unknown_tool" and "capped_output" in result.intercepted
    spills = list(cfg.run_dir.rglob("*.txt"))
    assert len(spills) == 1 and spills[0].parent == cfg.run_dir / "outputs"
    assert len(spills[0].name) < 80 and ".." not in spills[0].name
