"""Deterministic generator for the benchmark workspace.

The workspace is a small Python project with seeded bugs, a missing feature, a noisy 3000-line
log and a 1500-row CSV. Everything derives from `seed`, so the checker can regenerate the
pristine data and compute the expected answers independently of what the agent did.

    python -m bench.task.workspace /tmp/ws --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

DEFAULT_SEED = 7

TASK = """\
You are working in a small Python project (`inventory/`) that analyses order exports. Work through ALL \
nine steps below. docs/SPEC.md is the source of truth for how the code must behave - read it first.

1. Incident triage. logs/app.log contains exactly one FATAL entry. Write its incident id and SKU to \
notes/incident.txt as two lines: `incident=<id>` and `sku=<sku>`.
2. Bug fixing. `python -m pytest tests` currently fails. Fix the bugs in the `inventory` package (do not \
change the existing tests) so the code follows docs/SPEC.md and all tests pass.
3. Feature. Implement `inventory.report.top_products(orders, n=5)` as specified in SPEC section 4.
4. CLI. Add a `summary` subcommand: `python -m inventory.cli summary data/orders.csv` must print the JSON \
document described in SPEC section 5.
5. Results. Save the output of that command for data/orders.csv to out/summary.json.
6. Tests. Add tests/test_top_products.py with at least three test functions for `top_products`. The whole \
suite must pass.
7. Changelog. Under `## [Unreleased]` in CHANGELOG.md add a `### Fixed` section with one bullet per fixed \
module (name `parser`, `pricing` and `stock`) and a `### Added` section mentioning `top_products` and the \
`summary` command.
8. Report. Write REPORT.md containing these four lines with the real values:
   Incident: <incident id from step 1>
   Total revenue: <total_revenue from out/summary.json>
   Top SKU: <top_sku from out/summary.json>
   Incident SKU stock: <stock level of the incident SKU after applying all orders from data/orders.csv to \
the initial stock in data/products.json with inventory.stock.apply_orders>
9. Commit. Commit all of your work to the existing git repository with a message that starts with `fix:`. \
Do not commit `__pycache__` or `.pytest_cache` directories.

When all nine steps are done, reply with a short summary of what you did."""

# --------------------------------------------------------------------------- project sources

FILES: dict[str, str] = {}

FILES["inventory/__init__.py"] = '"""Inventory analytics for order exports."""\n'

FILES["inventory/models.py"] = '''\
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Order:
    order_id: str
    date: str
    region: str
    sku: str
    quantity: int
    unit_price: Decimal
    discount_pct: int
    status: str

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"
'''

FILES["inventory/parser.py"] = '''\
"""CSV loading for order exports (docs/SPEC.md section 1)."""
import csv
from decimal import Decimal

from .models import Order


def parse_row(row: dict) -> Order:
    """Build an Order from one CSV row, normalising the fields."""
    return Order(
        order_id=row["order_id"].strip(),
        date=row["date"].strip(),
        region=row["region"].strip().upper(),
        sku=row["sku"].strip().upper(),
        quantity=int(row["quantity"]),
        unit_price=Decimal(row["unit_price"].strip()),
        discount_pct=int(row["discount_pct"].strip() or 0),
        status=row["status"].strip().lower(),
    )


def load_orders(path) -> list[Order]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [parse_row(row) for row in csv.DictReader(handle)]
'''

FILES["inventory/pricing.py"] = '''\
"""Money arithmetic (docs/SPEC.md section 2)."""
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def line_total(order) -> Decimal:
    """Net amount of one order line, rounded half-up to cents."""
    gross = order.unit_price * order.quantity
    net = gross * (Decimal(100) - Decimal(order.discount_pct)) / Decimal(100)
    return net.quantize(CENT, rounding=ROUND_HALF_UP)
'''

FILES["inventory/stock.py"] = '''\
"""Stock levels (docs/SPEC.md section 3)."""


def needs_reorder(level: int, threshold: int) -> bool:
    return level <= threshold


def apply_orders(stock: dict, orders) -> dict:
    """Return a new stock mapping after shipping the given orders."""
    result = dict(stock)
    for order in orders:
        if not order.is_completed:
            continue
        result[order.sku] = result.get(order.sku, 0) - order.quantity
    return result


def reorder_list(stock: dict, thresholds: dict) -> list:
    return sorted(sku for sku, level in stock.items() if needs_reorder(level, thresholds.get(sku, 0)))
'''

FILES["inventory/report.py"] = '''\
"""Revenue reports (docs/SPEC.md section 4)."""
from collections import defaultdict
from decimal import Decimal

from .pricing import line_total


def completed(orders) -> list:
    return [order for order in orders if order.is_completed]


def total_revenue(orders) -> Decimal:
    return sum((line_total(order) for order in completed(orders)), Decimal("0.00"))


def revenue_by_region(orders) -> dict:
    totals = defaultdict(lambda: Decimal("0.00"))
    for order in completed(orders):
        totals[order.region] += line_total(order)
    return dict(sorted(totals.items()))


def top_products(orders, n=5):
    """Return the n best-selling SKUs as (sku, revenue) pairs. See docs/SPEC.md section 4."""
    raise NotImplementedError("top_products is not implemented yet")
'''

FILES["inventory/cli.py"] = '''\
"""Command line interface: python -m inventory.cli <command> ..."""
import argparse

from .parser import load_orders
from .report import revenue_by_region


def cmd_regions(args) -> int:
    for region, revenue in revenue_by_region(load_orders(args.csv)).items():
        print(f"{region}\\t{revenue}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inventory")
    sub = parser.add_subparsers(dest="command", required=True)
    regions = sub.add_parser("regions", help="revenue per region")
    regions.add_argument("csv")
    regions.set_defaults(func=cmd_regions)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''

# (file, correct snippet, buggy snippet). The workspace ships the buggy text; the reference
# solution (bench/runners/replay.py) applies the reverse edits.
BUGS: list[tuple[str, str, str]] = [
    ("inventory/parser.py", 'region=row["region"].strip().upper(),', 'region=row["region"],'),
    ("inventory/parser.py", 'discount_pct=int(row["discount_pct"].strip() or 0),', 'discount_pct=int(row["discount_pct"]),'),
    ("inventory/parser.py", 'status=row["status"].strip().lower(),', 'status=row["status"].strip(),'),
    ("inventory/pricing.py",
     "    gross = order.unit_price * order.quantity\n"
     "    net = gross * (Decimal(100) - Decimal(order.discount_pct)) / Decimal(100)\n"
     "    return net.quantize(CENT, rounding=ROUND_HALF_UP)\n",
     "    gross = float(order.unit_price) * order.quantity\n"
     "    net = gross * (1 - order.discount_pct / 100)\n"
     "    return Decimal(str(round(net, 2)))\n"),
    ("inventory/stock.py", "    return level <= threshold\n", "    return level < threshold\n"),
    ("inventory/stock.py", "        if not order.is_completed:\n            continue\n", ""),
]

TOP_PRODUCTS_STUB = '    raise NotImplementedError("top_products is not implemented yet")\n'
TOP_PRODUCTS_IMPL = '''\
    totals = defaultdict(lambda: Decimal("0.00"))
    for order in completed(orders):
        totals[order.sku] += line_total(order)
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:n]
'''

CLI_REFERENCE = '''\
"""Command line interface: python -m inventory.cli <command> ..."""
import argparse
import json

from .parser import load_orders
from .report import completed, revenue_by_region, top_products, total_revenue


def cmd_regions(args) -> int:
    for region, revenue in revenue_by_region(load_orders(args.csv)).items():
        print(f"{region}\\t{revenue}")
    return 0


def build_summary(orders) -> dict:
    top = top_products(orders, 1)
    return {
        "orders": len(completed(orders)),
        "total_revenue": str(total_revenue(orders)),
        "top_sku": top[0][0] if top else None,
        "revenue_by_region": {region: str(value) for region, value in revenue_by_region(orders).items()},
    }


def cmd_summary(args) -> int:
    print(json.dumps(build_summary(load_orders(args.csv)), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inventory")
    sub = parser.add_subparsers(dest="command", required=True)
    regions = sub.add_parser("regions", help="revenue per region")
    regions.add_argument("csv")
    regions.set_defaults(func=cmd_regions)
    summary = sub.add_parser("summary", help="JSON summary of an order export")
    summary.add_argument("csv")
    summary.set_defaults(func=cmd_summary)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''

FILES["docs/SPEC.md"] = '''\
# Inventory analytics - specification

## 1. Parsing (`inventory.parser`)
`parse_row` turns one CSV row into an `Order`. Exports come from several systems, so fields are messy:
- every text field is stripped of surrounding whitespace;
- `region` and `sku` are upper-cased, `status` is lower-cased;
- an empty `discount_pct` means 0.

## 2. Pricing (`inventory.pricing`)
`line_total(order)` = quantity x unit_price x (100 - discount_pct) / 100, computed with `decimal.Decimal`
and rounded to cents with ROUND_HALF_UP (7.575 -> 7.58). Binary floats must not be used for money.

## 3. Stock (`inventory.stock`)
- `needs_reorder(level, threshold)` is true when the level is **at or below** the threshold.
- `apply_orders(stock, orders)` subtracts the quantity of every **completed** order from its SKU and
  returns a new mapping. Cancelled and pending orders do not move stock. Unknown SKUs start at 0.

## 4. Reports (`inventory.report`)
Only completed orders count as revenue.
- `total_revenue(orders)` - sum of line totals.
- `revenue_by_region(orders)` - mapping region -> revenue, sorted by region.
- `top_products(orders, n=5)` - list of `(sku, revenue)` tuples for the `n` SKUs with the highest
  revenue, highest first; ties are broken by SKU in ascending order. Revenue is a `Decimal`.
  Fewer than `n` SKUs -> return them all. No completed orders -> empty list.

## 5. CLI (`python -m inventory.cli`)
`summary <csv>` prints one JSON object to stdout and nothing else:

    {
      "orders": <number of completed orders, integer>,
      "revenue_by_region": {"<REGION>": "<revenue with two decimals, as a string>", ...},
      "top_sku": "<SKU with the highest revenue>",
      "total_revenue": "<total revenue with two decimals, as a string>"
    }
'''

FILES["CHANGELOG.md"] = '''\
# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## [0.1.0] - 2026-01-12
### Added
- `regions` command printing revenue per region.
'''

FILES["README.md"] = "# inventory\n\nOrder export analytics. See docs/SPEC.md.\n"

FILES["tests/conftest.py"] = '''\
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inventory.models import Order  # noqa: E402


@pytest.fixture
def order():
    def make(sku="SKU-1", quantity=1, unit_price="10.00", discount_pct=0, status="completed", region="EU"):
        return Order("O-1", "2026-01-01", region, sku, quantity, Decimal(unit_price), discount_pct, status)

    return make
'''

FILES["tests/test_parser.py"] = '''\
from decimal import Decimal

from inventory.parser import load_orders, parse_row

ROW = {"order_id": "O-1", "date": "2026-01-01", "region": "EU", "sku": "SKU-1", "quantity": "2",
       "unit_price": "9.99", "discount_pct": "10", "status": "completed"}


def test_parse_row_basic():
    order = parse_row(ROW)
    assert (order.sku, order.quantity, order.unit_price, order.discount_pct) == ("SKU-1", 2, Decimal("9.99"), 10)


def test_region_is_stripped_and_uppercased():
    assert parse_row({**ROW, "region": " eu "}).region == "EU"


def test_status_is_stripped_and_lowercased():
    assert parse_row({**ROW, "status": "Completed "}).is_completed


def test_blank_discount_means_zero():
    assert parse_row({**ROW, "discount_pct": ""}).discount_pct == 0


def test_load_orders(tmp_path):
    path = tmp_path / "o.csv"
    path.write_text("order_id,date,region,sku,quantity,unit_price,discount_pct,status\\n"
                    "O-1,2026-01-01,us ,sku-2,3,1.50,,COMPLETED\\n")
    (order,) = load_orders(path)
    assert (order.region, order.sku, order.discount_pct, order.status) == ("US", "SKU-2", 0, "completed")
'''

FILES["tests/test_pricing.py"] = '''\
from decimal import Decimal

from inventory.pricing import line_total


def test_plain_total(order):
    assert line_total(order(quantity=3, unit_price="2.50")) == Decimal("7.50")


def test_discount(order):
    assert line_total(order(quantity=2, unit_price="10.00", discount_pct=15)) == Decimal("17.00")


def test_half_cent_rounds_up(order):
    assert line_total(order(quantity=1, unit_price="10.10", discount_pct=25)) == Decimal("7.58")


def test_result_has_two_decimals(order):
    assert str(line_total(order(quantity=1, unit_price="3", discount_pct=0))) == "3.00"
'''

FILES["tests/test_stock.py"] = '''\
from inventory.stock import apply_orders, needs_reorder, reorder_list


def test_reorder_at_threshold():
    assert needs_reorder(5, 5)
    assert needs_reorder(4, 5)
    assert not needs_reorder(6, 5)


def test_apply_orders_ignores_non_completed(order):
    orders = [order(sku="A", quantity=3), order(sku="A", quantity=4, status="cancelled"),
              order(sku="B", quantity=1, status="pending")]
    assert apply_orders({"A": 10, "B": 2}, orders) == {"A": 7, "B": 2}


def test_apply_orders_does_not_mutate_input(order):
    stock = {"A": 1}
    apply_orders(stock, [order(sku="A")])
    assert stock == {"A": 1}


def test_reorder_list():
    assert reorder_list({"A": 1, "B": 9, "C": 3}, {"A": 2, "B": 2, "C": 3}) == ["A", "C"]
'''

FILES["tests/test_report.py"] = '''\
from decimal import Decimal

from inventory.report import revenue_by_region, total_revenue


def test_revenue_by_region_counts_completed_only(order):
    orders = [order(region="EU", quantity=2), order(region="US"), order(region="EU", status="cancelled")]
    assert revenue_by_region(orders) == {"EU": Decimal("20.00"), "US": Decimal("10.00")}


def test_total_revenue(order):
    assert total_revenue([order(quantity=2), order(status="pending")]) == Decimal("20.00")
'''

# --------------------------------------------------------------------------- data

REGIONS = ["EU", "US", "APAC", "LATAM"]
COMPONENTS = ["api", "stock-sync", "billing", "scheduler", "auth", "export", "cache"]
INFO_MESSAGES = [
    'request completed method=GET path=/v1/orders status=200 duration_ms={ms} request_id={rid}',
    'request completed method=POST path=/v1/orders status=201 duration_ms={ms} request_id={rid}',
    'cache refresh finished keys={n} duration_ms={ms}',
    'export batch written rows={n} target=s3://exports/orders/{rid}.csv',
    'stock sync tick skus={n} drift=0',
    'token refreshed subject=svc-{n} ttl_s=3600',
    'job scheduled name=nightly-report run_id={rid}',
]
DEBUG_MESSAGES = ['pool stats active={n} idle={ms}', 'sql SELECT * FROM orders WHERE id={n} took {ms}ms',
                  'feature flag evaluated name=new-pricing value=false request_id={rid}']
WARN_MESSAGES = ['slow query duration_ms={ms} table=orders request_id={rid}',
                 'retrying upstream call attempt={n} service=billing', 'stock below threshold sku={sku} level={n}']
ERROR_MESSAGES = ['incident={inc} sku={sku} msg="price mismatch between catalog and order"',
                  'incident={inc} sku={sku} msg="upstream timeout while reserving stock"',
                  'incident={inc} sku={sku} msg="export row rejected"']


def build_data(seed: int) -> dict[str, str]:
    """Return {relative path: content} for products, orders and the log."""
    rng = random.Random(seed)
    skus = [f"SKU-{rng.randrange(100, 1000):04d}" for _ in range(60)]
    skus = sorted(set(skus))[:40]
    products = [{"sku": sku, "name": f"Item {sku[-4:]}", "unit_price": f"{rng.randrange(199, 9999) / 100:.2f}",
                 "initial_stock": rng.randrange(150, 400), "reorder_threshold": rng.randrange(10, 60)} for sku in skus]
    price = {p["sku"]: p["unit_price"] for p in products}

    rows = ["order_id,date,region,sku,quantity,unit_price,discount_pct,status"]
    for number in range(1, 1501):
        sku = rng.choice(skus)
        region = rng.choice(REGIONS)
        status = rng.choices(["completed", "cancelled", "pending"], weights=[78, 12, 10])[0]
        discount = str(rng.choice([0, 0, 0, 5, 10, 15, 25]))
        roll = rng.random()
        if roll < 0.06:
            region = rng.choice([f" {region.lower()}", f"{region.title()} ", f"{region.lower()}"])
        elif roll < 0.10:
            status = rng.choice([status.title(), f"{status.upper()} ", f" {status}"])
        elif roll < 0.13:
            discount = ""
        elif roll < 0.15:
            sku = sku.lower()
        date = f"2026-{rng.randrange(1, 4):02d}-{rng.randrange(1, 29):02d}"
        rows.append(f"O-{number:05d},{date},{region},{sku},{rng.randrange(1, 13)},{price[sku.upper()]},{discount},{status}")

    incident_sku = rng.choice(skus)
    incident = f"INC-{rng.randrange(10000, 99999)}"
    fatal_at = rng.randrange(1700, 2700)
    lines = []
    second = 0
    for number in range(3000):
        second += rng.randrange(1, 9)
        stamp = f"2026-03-14T{second // 3600 % 24:02d}:{second // 60 % 60:02d}:{second % 60:02d}Z"
        fill = dict(ms=rng.randrange(2, 900), n=rng.randrange(1, 500), rid=f"{rng.getrandbits(48):012x}",
                    sku=rng.choice(skus), inc=f"INC-{rng.randrange(10000, 99999)}")
        if number == fatal_at:
            lines.append(f'{stamp} FATAL [stock-sync] incident={incident} sku={incident_sku} '
                         'msg="negative stock level after sync" action=halted')
            continue
        level, pool = rng.choices([("INFO", INFO_MESSAGES), ("DEBUG", DEBUG_MESSAGES), ("WARN", WARN_MESSAGES),
                                   ("ERROR", ERROR_MESSAGES)], weights=[78, 12, 7, 3])[0]
        lines.append(f"{stamp} {level} [{rng.choice(COMPONENTS)}] {rng.choice(pool).format(**fill)}")

    return {"data/products.json": json.dumps(products, indent=1) + "\n",
            "data/orders.csv": "\n".join(rows) + "\n",
            "logs/app.log": "\n".join(lines) + "\n"}


def project_files(buggy: bool = True) -> dict[str, str]:
    files = dict(FILES)
    if buggy:
        for path, correct, broken in BUGS:
            assert correct in files[path], (path, correct)
            files[path] = files[path].replace(correct, broken)
    return files


def make_workspace(dest: Path, seed: int = DEFAULT_SEED, git: bool = True) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in {**project_files(buggy=True), **build_data(seed)}.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (dest / "TASK.md").write_text(TASK + "\n", encoding="utf-8")
    if git:
        def run(*args: str) -> None:
            subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True)

        run("init", "-q", "-b", "main")
        run("config", "user.name", "bench")
        run("config", "user.email", "bench@example.com")
        run("config", "commit.gpgsign", "false")
        run("add", "-A")
        run("commit", "-q", "-m", "chore: import inventory project")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(make_workspace(args.dest, args.seed))
