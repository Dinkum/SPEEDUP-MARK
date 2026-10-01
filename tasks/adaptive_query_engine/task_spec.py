"""Plan and execute a compact query representation over in-memory integer tables.

The workload is a fixed grammar of relational plans -- scans, filters, equality
joins, grouped aggregates, top-k selections and projections -- handed to the
candidate as a parsed tree per query. There is no SQL text, no server, no disk
and no networking: what is timed is planning, index construction and execution
of a small engine, all inside the candidate call.

Every generated family places its filters *above* the joins and writes its
joins in a fixed left-deep order, so pushing a predicate into a scan, choosing
which side to hash, and reordering a three-way join are real decisions rather
than decoration. Families are generated in regimes that disagree: a filter that
is highly selective in one family is nearly useless in another, and the table
sizes that make one side the cheap build side are reversed elsewhere.
"""

from __future__ import annotations

import random
import sys

from speedupmark.task import forbidden_imports, load_candidate, watch_imports


_candidate = load_candidate(__file__)

# Handing the whole job to an embedded relational engine is a contract
# violation, not an optimization; a vector or dictionary primitive is fine.
ENGINE_ROOTS = ("sqlite3", "_sqlite3", "duckdb", "_duckdb", "sqlalchemy", "apsw",
                "polars", "pandas", "datafusion", "pyarrow", "sqlglot", "pandasql")
AGGREGATES = ("sum", "count", "min", "max")


def _comparison(op):
    if op == "eq":
        return lambda left, right: left == right
    if op == "ne":
        return lambda left, right: left != right
    if op == "lt":
        return lambda left, right: left < right
    if op == "le":
        return lambda left, right: left <= right
    if op == "gt":
        return lambda left, right: left > right
    if op == "ge":
        return lambda left, right: left >= right
    raise ValueError(f"unsupported comparison {op!r}")


def _evaluate_node(node, tables):
    """Return ``(schema, rows)`` for one plan node, evaluated left to right.

    The reference never reorders anything: it hashes the left input of a join,
    probes with the right input, and applies each filter exactly where the tree
    puts it.
    """
    kind = node[0]
    if kind == "scan":
        table = tables[node[1]]
        return tuple(table["columns"]), [tuple(row) for row in table["rows"]]
    if kind == "filter":
        _, source, column, op, value = node
        schema, rows = _evaluate_node(source, tables)
        index = schema.index(column)
        keep = _comparison(op)
        return schema, [row for row in rows if keep(row[index], value)]
    if kind == "join":
        _, left_node, right_node, left_column, right_column = node
        left_schema, left_rows = _evaluate_node(left_node, tables)
        right_schema, right_rows = _evaluate_node(right_node, tables)
        left_index = left_schema.index(left_column)
        right_index = right_schema.index(right_column)
        buckets = {}
        for row in left_rows:
            buckets.setdefault(row[left_index], []).append(row)
        joined = []
        for row in right_rows:
            for match in buckets.get(row[right_index], ()):
                joined.append(match + row)
        return left_schema + right_schema, joined
    if kind == "group":
        _, source, keys, aggregates = node
        schema, rows = _evaluate_node(source, tables)
        key_indices = [schema.index(key) for key in keys]
        aggregate_indices = [
            None if column is None else schema.index(column)
            for _, column in aggregates
        ]
        groups = {}
        for row in rows:
            key = tuple(row[index] for index in key_indices)
            accumulator = groups.get(key)
            if accumulator is None:
                accumulator = [None] * len(aggregates)
                groups[key] = accumulator
            for position, (function, column) in enumerate(aggregates):
                if function == "count":
                    accumulator[position] = (accumulator[position] or 0) + 1
                    continue
                value = row[aggregate_indices[position]]
                current = accumulator[position]
                if current is None:
                    # sum, min and max all start from the first value seen.
                    accumulator[position] = value
                elif function == "sum":
                    accumulator[position] = current + value
                elif function == "min":
                    accumulator[position] = min(current, value)
                else:
                    accumulator[position] = max(current, value)
        output_schema = tuple(keys) + tuple(
            "count" if column is None else f"{function}:{column}"
            for function, column in aggregates
        )
        return output_schema, [key + tuple(value) for key, value in sorted(groups.items())]
    if kind == "topk":
        _, source, column, limit = node
        schema, rows = _evaluate_node(source, tables)
        if limit < 1:
            raise ValueError("topk requires a positive limit")
        index = schema.index(column)
        # Total order: the named column descending, then the whole row ascending,
        # so the boundary between rank k and rank k+1 is decided exactly.
        rows.sort(key=lambda row: (-row[index], row))
        selected = rows[:limit]
        selected.sort()
        return schema, selected
    if kind == "project":
        _, source, columns = node
        schema, rows = _evaluate_node(source, tables)
        indices = [schema.index(column) for column in columns]
        return tuple(columns), [tuple(row[index] for index in indices) for row in rows]
    raise ValueError(f"unsupported plan node {kind!r}")


def _answer_query(node, tables):
    schema, rows = _evaluate_node(node, tables)
    for row in rows:
        if any(type(value) is not int or isinstance(value, bool) for value in row):
            raise ValueError("query results must be exact integers")
    return tuple(sorted(rows))


def _answer(problem):
    """Canonical answers: one tuple of rows per query, ascending, per family."""
    return tuple(
        tuple(_answer_query(query, family["tables"]) for query in family["queries"])
        for family in problem["families"]
    )


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
    name = "adaptive_query_engine"
    task_version = "1.1.0"
    display_name = "Adaptive Query Engine"
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
        return {"families": families}

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

    def solve(self, problem):
        return _answer(problem)

    def candidate_solve(self, problem):
        self.policy_violations = ()
        loaded = set(sys.modules)
        with watch_imports(ENGINE_ROOTS) as imported_during:
            result = _candidate.solve(problem, self.solve)
        violations = forbidden_imports(_candidate, ENGINE_ROOTS, loaded, imported_during)
        if violations:
            # An invalid submission must not be scored as a fast one. Returning a
            # value that cannot be a plan answer fails verification for the
            # candidate alone; the reference still verifies independently.
            self.policy_violations = violations
            print(f"candidate uses forbidden engine imports: {', '.join(violations)}")
            return None
        return result

    def is_solution(self, problem, proposed):
        try:
            if not _materialized(proposed):
                return False
            return _same_materialized(proposed, _answer(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = AdaptiveQueryEngineTask()
