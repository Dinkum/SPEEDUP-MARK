"""Plan and execute a compact query representation over in-memory integer tables.

The workload is a fixed grammar of relational plans -- scans, filters, equality
joins, grouped aggregates, top-k selections and projections -- handed to the
candidate as a parsed tree per query. There is no SQL text, no server, no disk
and no networking: what is timed is planning, index construction and execution
of a small engine, all inside the candidate call.

Family anchors place filters above joins; randomized transformations vary join
orientation, predicate nesting and output operators. Pushing predicates, choosing
which side to hash, and reordering a three-way join are real decisions rather
than decoration. Families are generated in regimes that disagree: a filter that
is highly selective in one family is nearly useless in another, and the table
sizes that make one side the cheap build side are reversed elsewhere.
"""

from __future__ import annotations

import random
from itertools import groupby

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


# Handing the whole job to an embedded relational engine is a contract
# violation, not an optimization; a vector or dictionary primitive is fine.
ENGINE_ROOTS = ("sqlite3", "_sqlite3", "duckdb", "_duckdb", "sqlalchemy", "apsw",
                "polars", "pandas", "datafusion", "pyarrow", "sqlglot", "pandasql")
AGGREGATES = ("sum", "count", "min", "max")


def _verify_node(node, tables):
    """Independent relational interpreter: sort/merge joins and sorted groups.

    Keep positional schemas: a join can contain duplicate column names, and
    the language resolves those names to the first occurrence.
    """
    kind = node[0]
    if kind == "scan":
        table = tables[node[1]]
        return tuple(table["columns"]), list(table["rows"])
    if kind == "join":
        left_schema, left = _verify_node(node[1], tables)
        right_schema, right = _verify_node(node[2], tables)
        li, ri = left_schema.index(node[3]), right_schema.index(node[4])
        left_groups = iter(groupby(sorted(left, key=lambda row: row[li]), lambda row: row[li]))
        right_groups = iter(groupby(sorted(right, key=lambda row: row[ri]), lambda row: row[ri]))
        a, b = next(left_groups, None), next(right_groups, None)
        rows = []
        while a is not None and b is not None:
            if a[0] < b[0]:
                a = next(left_groups, None)
            elif a[0] > b[0]:
                b = next(right_groups, None)
            else:
                right_rows = list(b[1])
                rows.extend(tuple(l) + tuple(r) for l in a[1] for r in right_rows)
                a, b = next(left_groups, None), next(right_groups, None)
        return left_schema + right_schema, rows
    schema, rows = _verify_node(node[1], tables)
    if kind == "filter":
        index, op, value = schema.index(node[2]), node[3], node[4]
        def keep(row):
            x = row[index]
            return {"eq": x == value, "ne": x != value, "lt": x < value,
                    "le": x <= value, "gt": x > value, "ge": x >= value}[op]
        return schema, list(filter(keep, rows))
    if kind == "project":
        indices = tuple(schema.index(column) for column in node[2])
        return tuple(node[2]), [tuple(row[i] for i in indices) for row in rows]
    if kind == "topk":
        if node[3] < 1:
            raise ValueError("topk requires a positive limit")
        index = schema.index(node[2])
        return schema, sorted(rows, key=lambda row: (-row[index], row))[:node[3]]
    if kind == "group":
        keys, aggregates = node[2:]
        key_indices = tuple(schema.index(column) for column in keys)
        key_of = lambda row: tuple(row[i] for i in key_indices)
        result = []
        for key, members in groupby(sorted(rows, key=key_of), key_of):
            members = list(members)
            values = []
            for op, column in aggregates:
                if op == "count":
                    values.append(len(members))
                else:
                    index = schema.index(column)
                    reducer = {"sum": sum, "min": min, "max": max}[op]
                    values.append(reducer(row[index] for row in members))
            result.append(key + tuple(values))
        names = tuple(keys) + tuple("count" if col is None else f"{op}:{col}"
                                    for op, col in aggregates)
        return names, result
    raise ValueError(f"unsupported plan node {kind!r}")


def _verified_answer(problem):
    return tuple(tuple(tuple(sorted(_verify_node(query, family["tables"])[1]))
                       for query in family["queries"]) for family in problem["families"])


def _vary_plans(family, rng):
    """Sample the grammar, not just the data behind a published template.

    Memoization preserves shared subtrees. Join reversal and column renaming
    prevent positional/template dispatch while retaining each workload regime.
    """
    memo = {}
    def vary(node):
        if node in memo:
            return memo[node]
        kind = node[0]
        if kind == "scan":
            result = node
        elif kind == "join":
            left, right = vary(node[1]), vary(node[2])
            if rng.randrange(2):
                result = (kind, right, left, node[4], node[3])
            else:
                result = (kind, left, right, *node[3:])
        elif kind == "filter":
            # Preserve selective/permissive thresholds while varying equality,
            # rank boundaries, and nested predicate depth in every family.
            column, op, value = node[2:]
            value += rng.choice((-1, 0, 1))
            result = (kind, vary(node[1]), column, op, value)
            if rng.randrange(3) == 0:
                result = (kind, result, column, "ne", value + rng.randrange(1, 4))
        elif kind == "group":
            keys, aggregates = list(node[2]), list(node[3])
            rng.shuffle(keys)
            aggregates = [(rng.choice(AGGREGATES) if col is not None else "count", col)
                          for _, col in aggregates]
            rng.shuffle(aggregates)
            result = (kind, vary(node[1]), tuple(keys), tuple(aggregates))
        elif kind == "project":
            columns = list(node[2])
            rng.shuffle(columns)
            result = (kind, vary(node[1]), tuple(columns))
        elif kind == "topk":
            result = (kind, vary(node[1]), node[2], rng.randrange(1, 17))
        else:
            raise ValueError(kind)
        memo[node] = result
        return result
    queries = [vary(query) for query in family["queries"]]
    # Extra compositions force support for operator nesting beyond the anchors.
    for _ in range(rng.randrange(2, 5)):
        source = rng.choice(queries)
        schema = _verify_node(source, {name: table | {"rows": ()}
                                     for name, table in family["tables"].items()})[0]
        column = rng.choice(schema)
        source = ("filter", source, column, rng.choice(("ne", "ge", "lt")), rng.randrange(12))
        if rng.randrange(2):
            source = ("group", source, (column,), (("count", None),))
        else:
            source = ("topk", source, column, rng.randrange(1, 12))
        queries.append(source)
    rng.shuffle(queries)
    table_names = {name: f"t{rng.getrandbits(40):010x}" for name in family["tables"]}
    columns = {column: f"c{rng.getrandbits(40):010x}"
               for table in family["tables"].values() for column in table["columns"]}
    def renamed(column):
        if column in columns:
            return columns[column]
        if column == "count":
            return column
        op, original = column.split(":", 1)
        return f"{op}:{renamed(original)}"
    def rename(node):
        kind = node[0]
        if kind == "scan":
            return kind, table_names[node[1]]
        if kind == "join":
            return kind, rename(node[1]), rename(node[2]), renamed(node[3]), renamed(node[4])
        if kind == "group":
            return (kind, rename(node[1]), tuple(map(renamed, node[2])),
                    tuple((op, None if col is None else renamed(col)) for op, col in node[3]))
        if kind == "project":
            return kind, rename(node[1]), tuple(map(renamed, node[2]))
        return kind, rename(node[1]), renamed(node[2]), *node[3:]
    tables = {table_names[name]: table | {"columns": tuple(map(renamed, table["columns"]))}
              for name, table in family["tables"].items()}
    return family | {"tables": tables, "queries": tuple(map(rename, queries))}


def _materialized(value):
    """Plain containers only: reject subclasses that defer work until verification."""
    if type(value) is int:
        return True
    if type(value) in (tuple, list):
        return all(_materialized(item) for item in value)
    return False


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _orders(rng, rows, customers):
    return {
        "columns": ("o_id", "cust", "region", "amount"),
        "rows": tuple(
            (o_id, rng.randrange(customers), rng.randrange(8), rng.randrange(1, 1000))
            for o_id in range(rows)
        ),
    }


def _customers(rng, count):
    return {
        "columns": ("cust", "tier", "signup"),
        "rows": tuple((cust, rng.randrange(4), rng.randrange(1, 400))
                      for cust in range(count)),
    }


def _items(rng, count, cost_ceiling=500):
    return {
        "columns": ("item", "category", "unit_cost"),
        "rows": tuple((item, rng.randrange(32), rng.randrange(1, cost_ceiling))
                      for item in range(count)),
    }


def _lineitems(rng, rows, orders, items, price_ceiling=500):
    return {
        "columns": ("li_o_id", "item", "quantity", "unit_price"),
        "rows": tuple(
            (rng.randrange(orders), rng.randrange(items), rng.randrange(1, 13),
             rng.randrange(1, price_ceiling))
            for _ in range(rows)
        ),
    }


def _events(rng, rows, customers, regions):
    """A skewed event log: few distinct keys, many rows per key."""
    return {
        "columns": ("ev_cust", "ev_region", "ev_amount", "ev_day"),
        "rows": tuple(
            (rng.randrange(customers), rng.randrange(regions), rng.randrange(1, 500),
             rng.randrange(1, 8))
            for _ in range(rows)
        ),
    }


class AdaptiveQueryEngineTask:
    forbidden_import_roots = ENGINE_ROOTS
    name = "adaptive_query_engine"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 2400
    grading_cases = (2400, 3600)

    def generate_problem(self, n=800, random_seed=0):
        if n < 32:
            raise ValueError("n must be at least 32")
        rng = random.Random(random_seed)
        families = (
            self._selective_filter(n, rng),
            self._permissive_filter(n, rng),
            self._skew(n, rng),
            self._topk(n, rng),
            self._shared_subplan(n, rng),
        )
        return {"families": tuple(_vary_plans(family, rng) for family in families)}

    def _selective_filter(self, n, rng):
        customers = max(4, n // 16)
        items = max(4, n // 8)
        tables = {
            "orders": _orders(rng, n, customers),
            "customers": _customers(rng, customers),
            "items": _items(rng, items),
            "lineitems": _lineitems(rng, 2 * n, n, items),
        }
        queries = (
            # The amount filter sits above the join; pushing it into the orders
            # scan shrinks the build side before any row is concatenated.
            ("group", ("filter", ("join", ("scan", "orders"), ("scan", "customers"),
                                  "cust", "cust"), "amount", "gt", 900),
             ("tier",), (("sum", "amount"), ("count", None))),
            ("topk", ("filter", ("join", ("scan", "lineitems"), ("scan", "items"),
                                 "item", "item"), "category", "eq", 3), "unit_price", 6),
            ("group", ("filter", ("join", ("join", ("scan", "orders"), ("scan", "lineitems"),
                                           "o_id", "li_o_id"), ("scan", "items"), "item", "item"),
                       "region", "lt", 2), ("item",), (("count", None), ("sum", "quantity"))),
            ("project", ("filter", ("scan", "orders"), "region", "eq", 5), ("o_id", "amount")),
            ("topk", ("filter", ("join", ("scan", "orders"), ("scan", "lineitems"),
                                 "o_id", "li_o_id"), "quantity", "ge", 11), "amount", 8),
        )
        return {"family": "selective_filter", "tables": tables, "queries": queries}

    def _permissive_filter(self, n, rng):
        customers = max(4, n // 16)
        items = max(4, n // 8)
        tables = {
            "orders": _orders(rng, n, customers),
            "customers": _customers(rng, customers),
            "items": _items(rng, items),
            "lineitems": _lineitems(rng, 2 * n, n, items),
        }
        queries = (
            # Filters matching ~99% of rows: pushing them into the scan is an
            # extra pass that buys almost nothing, so a candidate that pushes
            # every predicate unconditionally pays for it here while the
            # selective family demands exactly that pushdown.
            ("group", ("filter", ("join", ("scan", "orders"), ("scan", "customers"),
                                  "cust", "cust"), "amount", "gt", 8),
             ("tier",), (("sum", "amount"), ("count", None))),
            ("topk", ("filter", ("join", ("scan", "lineitems"), ("scan", "items"),
                                 "item", "item"), "unit_price", "gt", 3), "quantity", 5),
            ("group", ("filter", ("join", ("join", ("scan", "orders"), ("scan", "lineitems"),
                                           "o_id", "li_o_id"), ("scan", "items"), "item", "item"),
                       "amount", "gt", 5),
             ("region",), (("count", None), ("sum", "quantity"), ("min", "unit_price"))),
            # Written large side first: hashing the written left input is a loss.
            ("group", ("join", ("scan", "lineitems"), ("scan", "items"), "item", "item"),
             ("category",), (("sum", "quantity"), ("max", "unit_price"))),
            ("project", ("join", ("scan", "orders"), ("scan", "customers"), "cust", "cust"),
             ("o_id", "tier")),
        )
        return {"family": "permissive_filter", "tables": tables, "queries": queries}

    def _skew(self, n, rng):
        customers = max(4, n // 32)
        tables = {
            "customers": _customers(rng, customers),
            "eventlog": _events(rng, 4 * n, max(2, customers // 4), 3),
        }
        queries = (
            # Aggregating the event log by its own join key before the join keeps
            # the join's fan-out proportional to the key count, not the row count.
            ("group", ("filter", ("join", ("scan", "eventlog"), ("scan", "customers"),
                                  "ev_cust", "cust"), "ev_day", "gt", 2),
             ("tier", "ev_region"), (("sum", "ev_amount"), ("count", None))),
            ("group", ("join", ("scan", "eventlog"), ("scan", "customers"), "ev_cust", "cust"),
             ("ev_region",), (("min", "ev_day"), ("max", "ev_day"), ("count", None))),
            ("topk", ("join", ("scan", "eventlog"), ("scan", "customers"), "ev_cust", "cust"),
             "ev_amount", 7),
            ("group", ("filter", ("scan", "eventlog"), "ev_region", "eq", 1), ("ev_cust",),
             (("sum", "ev_amount"),)),
            ("group", ("join", ("scan", "customers"), ("scan", "eventlog"), "cust", "ev_cust"),
             ("tier",), (("count", None), ("max", "ev_amount"))),
        )
        return {"family": "skew", "tables": tables, "queries": queries}

    def _topk(self, n, rng):
        customers = max(4, n // 16)
        items = max(4, n // 4)
        tables = {
            "orders": _orders(rng, max(16, n // 2), customers),
            "customers": _customers(rng, customers),
            # A deliberately coarse price column: the rank-k boundary is a tie by
            # construction, so top-k selection must respect the stated tie-break.
            "items": _items(rng, items, cost_ceiling=9),
            "lineitems": _lineitems(rng, 3 * n, max(16, n // 2), items, price_ceiling=9),
        }
        queries = (
            ("topk", ("join", ("scan", "lineitems"), ("scan", "items"), "item", "item"),
             "unit_price", 5),
            ("topk", ("filter", ("join", ("scan", "lineitems"), ("scan", "orders"),
                                 "li_o_id", "o_id"), "region", "eq", 4), "quantity", 3),
            ("topk", ("join", ("scan", "items"), ("scan", "lineitems"), "item", "item"),
             "unit_cost", 4),
            ("topk", ("filter", ("join", ("scan", "orders"), ("scan", "customers"),
                                 "cust", "cust"), "tier", "ge", 2), "amount", 6),
        )
        return {"family": "topk", "tables": tables, "queries": queries}

    def _shared_subplan(self, n, rng):
        customers = max(4, n // 16)
        items = max(4, n // 8)
        tables = {
            "orders": _orders(rng, 2 * n, customers),
            "customers": _customers(rng, customers),
            "items": _items(rng, items),
            "lineitems": _lineitems(rng, 2 * n, 2 * n, items),
        }
        # One filtered join, written identically into five queries. Evaluating it
        # once and reusing the rows is the intended improvement; each query is
        # still independently well-defined, so recomputation stays correct.
        base = ("filter", ("join", ("scan", "orders"), ("scan", "customers"),
                           "cust", "cust"), "amount", "gt", 700)
        queries = (
            ("group", base, ("region",), (("sum", "amount"), ("count", None))),
            ("topk", base, "amount", 4),
            ("group", base, ("tier",), (("count", None),)),
            ("project", ("filter", base, "region", "eq", 2), ("o_id", "amount", "tier")),
            ("group", ("filter", base, "tier", "ge", 2), ("region", "tier"),
             (("max", "amount"), ("count", None))),
            ("group", ("join", base, ("scan", "lineitems"), "o_id", "li_o_id"),
             ("region",), (("sum", "quantity"),)),
        )
        return {"family": "shared_subplan", "tables": tables, "queries": queries}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            if not _materialized(proposed):
                return False
            return _same_materialized(proposed, _verified_answer(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = AdaptiveQueryEngineTask()
