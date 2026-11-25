"""Offline replay benchmark: the same scripted solution through every harness configuration.

No model is involved. A fixed, deliberately un-optimised reference trajectory (it reads whole
files and runs verbose tests, like a mid-sized model would) solves the benchmark task with real
tool calls. Because the trajectory is identical everywhere, differences in the numbers are caused
by the harness alone: how much prompt each request carries, how much of it is a reusable prefix,
and whether the run survives a small window.

What this does NOT measure: whether a real model succeeds. That is bench/run.py.

    python -m bench.replay                 # table for the default windows
    python -m bench.replay --window 16000  # one window size
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # started as a file (IDE "Run"), not with `-m`: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import tempfile

from harness.runtime.config import Config
from harness.llm.client import LLMResponse
from harness.runtime.loop import Agent
from harness.llm.scripted import ScriptedLLM, call, say

from bench.task.check import check, reference
from bench.task.workspace import BUGS, CLI_REFERENCE, DEFAULT_SEED, FILES, TASK, TOP_PRODUCTS_IMPL, TOP_PRODUCTS_STUB, make_workspace

CORE = {"bash", "read_file", "write_file", "edit_file"}

# name -> (Config overrides, lookup). "before" is the naive way to scale the original loop to many tools.
# lookup=True models a cautious model that fetches the full schema before the first use of each
# catalog tool; lookup=False models one that calls straight from the signatures in the prompt
# (only possible in dispatch mode - in append mode a tool is not callable until it was searched).
VARIANTS: dict[str, tuple[dict, bool]] = {
    "eager, no cap, no compaction (before)": (dict(tool_mode="eager", compaction=False, cap_tool_output=False), False),
    "eager + cap + compaction": (dict(tool_mode="eager"), False),
    "append + cap + compaction": (dict(tool_mode="append"), True),
    "dispatch + cap + compaction, schema lookups": (dict(tool_mode="dispatch"), True),
    "dispatch + cap + compaction (after)": (dict(tool_mode="dispatch"), False),
}

NEW_TESTS = '''\
from decimal import Decimal

from inventory.report import top_products


def test_orders_by_revenue_desc(order):
    orders = [order(sku="A", quantity=1), order(sku="B", quantity=3), order(sku="C", quantity=2)]
    assert top_products(orders, 2) == [("B", Decimal("30.00")), ("C", Decimal("20.00"))]


def test_ties_are_broken_by_sku(order):
    assert [sku for sku, _ in top_products([order(sku="Z"), order(sku="M")])] == ["M", "Z"]


def test_ignores_non_completed_and_handles_empty(order):
    assert top_products([order(sku="A", status="cancelled")]) == []
    assert top_products([]) == []
'''

CHANGELOG_ENTRY = '''\
## [Unreleased]
### Added
- `report`: `top_products(orders, n)` returns the best-selling SKUs by revenue.
- `cli`: `summary` command printing a JSON summary of an order export.

### Fixed
- `parser`: region and status were not normalised and a blank discount crashed the loader.
- `pricing`: line totals were rounded with binary floats; they now use ROUND_HALF_UP on decimals.
- `stock`: the reorder threshold was exclusive and non-completed orders reduced stock.
'''

STOCK_SCRIPT = '''\
import json
from inventory.parser import load_orders
from inventory.stock import apply_orders
stock = {p["sku"]: p["initial_stock"] for p in json.load(open("data/products.json"))}
print(apply_orders(stock, load_orders("data/orders.csv"))["%s"])
'''


def reference_steps(seed: int = DEFAULT_SEED) -> list[tuple]:
    """The solution as logical steps: ("skill", name) or ("tool", name, args)."""
    ref = reference(seed)
    summary = ref["summary"]

    def t(name: str, **args) -> tuple:
        return ("tool", name, args)

    def fix(index: int) -> tuple:
        path, correct, broken = BUGS[index]
        return t("edit_file", path=path, old_string=broken, new_string=correct)

    stock_loop = "    for order in orders:\n        result[order.sku]"
    return [
        t("list_dir", path=".", depth=2),
        t("read_file", path="docs/SPEC.md"),
        # 1. incident triage
        ("skill", "log-triage"),
        t("bash", command="wc -l logs/app.log && ls -la logs"),
        t("log_levels", path="logs/app.log"),
        t("bash", command="head -150 logs/app.log"),
        t("grep", pattern="FATAL", path="logs/app.log"),
        t("write_file", path="notes/incident.txt", content=f"incident={ref['incident']}\nsku={ref['incident_sku']}\n"),
        # 2. bug fixing
        ("skill", "python-bugfix"),
        t("run_pytest", verbose=True),
        t("read_file", path="inventory/parser.py"),
        t("read_file", path="tests/test_parser.py"),
        fix(0), fix(1), fix(2),
        t("run_pytest", target="tests/test_parser.py"),
        t("read_file", path="inventory/pricing.py"),
        t("read_file", path="tests/test_pricing.py"),
        t("write_file", path="inventory/pricing.py", content=FILES["inventory/pricing.py"]),
        t("run_pytest", target="tests/test_pricing.py"),
        t("read_file", path="inventory/stock.py"),
        t("read_file", path="tests/test_stock.py"),
        fix(4),
        t("edit_file", path="inventory/stock.py", old_string=stock_loop,
          new_string=stock_loop.replace("\n        result", "\n        if not order.is_completed:\n            continue\n        result")),
        t("run_pytest", verbose=True),
        # 3. feature
        t("read_file", path="inventory/report.py"),
        t("read_file", path="tests/test_report.py"),
        t("edit_file", path="inventory/report.py", old_string=TOP_PRODUCTS_STUB, new_string=TOP_PRODUCTS_IMPL),
        # 4. cli
        ("skill", "cli-design"),
        t("py_outline", path="inventory/cli.py"),
        t("read_file", path="inventory/cli.py"),
        t("write_file", path="inventory/cli.py", content=CLI_REFERENCE),
        # 5. results
        ("skill", "csv-data-analysis"),
        t("csv_summary", path="data/orders.csv"),
        t("read_file", path="data/orders.csv", limit=120),
        t("bash", command="mkdir -p out && python -m inventory.cli summary data/orders.csv > out/summary.json && cat out/summary.json"),
        t("json_query", path="out/summary.json", pointer="revenue_by_region"),
        # 6. tests
        ("skill", "test-writing"),
        t("read_file", path="tests/conftest.py"),
        t("write_file", path="tests/test_top_products.py", content=NEW_TESTS),
        t("run_pytest", verbose=True),
        # 7. changelog
        ("skill", "changelog-writing"),
        t("read_file", path="CHANGELOG.md"),
        t("edit_file", path="CHANGELOG.md", old_string="## [Unreleased]\n", new_string=CHANGELOG_ENTRY),
        # 8. report
        t("json_query", path="data/products.json", pointer="0"),
        t("run_python", code=STOCK_SCRIPT % ref["incident_sku"]),
        t("read_file", path="notes/incident.txt"),
        t("write_file", path="REPORT.md",
          content=f"# Report\n\nIncident: {ref['incident']}\nTotal revenue: {summary['total_revenue']}\n"
                  f"Top SKU: {summary['top_sku']}\nIncident SKU stock: {ref['incident_sku_stock']}\n"),
        # 9. commit
        ("skill", "git-commit"),
        t("git_status"),
        t("write_file", path=".gitignore", content="__pycache__/\n.pytest_cache/\n"),
        t("git_diff"),
        t("git_commit", message="fix: parser, pricing and stock bugs; add top_products and summary command"),
        t("git_log", limit=3),
        t("git_status"),
    ]


def script_for(mode: str, steps: list[tuple], lookup: bool = True) -> list[LLMResponse]:
    """Translate logical steps into what a model would emit in the given tool mode."""
    out: list[LLMResponse] = []
    known: set[str] = set()
    for step in steps:
        if step[0] == "skill":
            if mode != "eager":  # eager already has every skill body in the system prompt
                out.append(call("load_skill", name=step[1]))
            continue
        _, name, args = step
        if name in CORE or mode == "eager":
            out.append(call(name, **args))
            continue
        if lookup and name not in known:  # look the schema up once; git tools are fetched as one batch
            batch = [n for n in ("git_status", "git_diff", "git_commit", "git_log")] if name.startswith("git_") else [name]
            known.update(batch)
            out.append(call("tool_search", query=" ".join(batch)))
        out.append(call(name, **args) if mode == "append" else call("use_tool", name=name, arguments=args))
    return out + [say("All nine steps are done.")]


def replay(overrides: dict, window: int, enforce: bool, lookup: bool = True, seed: int = DEFAULT_SEED,
           root: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(root or tmp)
        workspace = make_workspace(base / "workspace", seed)
        cfg = Config(workdir=workspace, run_dir=base / "run", model="scripted-replay", context_window=window,
                     enforce_window=enforce, max_output_tokens=2000, **overrides)
        llm = ScriptedLLM(script_for(cfg.tool_mode, reference_steps(seed), lookup))
        result = Agent(cfg, llm=llm, echo=None).run(TASK)
        verdict = check(workspace, seed)
        return {**result.metrics, "checks_passed": verdict["passed"], "checks_total": verdict["total"],
                "success": result.ok and verdict["success"]}


def table(rows: list[tuple[str, dict]]) -> str:
    head = ("| variant | result | checks | requests | prompt tokens | reusable prefix | uncached prompt | peak prompt | compactions |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for name, m in rows:
        body += (f"| {name} | {m['status']} | {m['checks_passed']}/{m['checks_total']} | {m['requests']} | "
                 f"{m['prompt_tokens']:,} | {m['prefix_reuse']:.0%} | {m['uncached_prompt_tokens_est']:,} | "
                 f"{m['peak_prompt_tokens']:,} | {len(m['compactions'])} |\n")
    return head + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, action="append", help="Enforced window in tokens (repeatable).")
    parser.add_argument("--json", type=Path, help="Also write raw results to this file.")
    args = parser.parse_args()
    results = {}
    print("Tokens are estimated as chars/4 (no tokenizer); the trajectory is identical in every row.\n")
    for window, enforce in [(1_000_000, False)] + [(w, True) for w in (args.window or [24_000, 12_000])]:
        label = f"window {window:,} tokens (enforced)" if enforce else "unlimited window"
        rows = [(name, replay(overrides, window, enforce, lookup)) for name, (overrides, lookup) in VARIANTS.items()]
        results[label] = dict(rows)
        print(f"### {label}\n\n{table(rows)}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
