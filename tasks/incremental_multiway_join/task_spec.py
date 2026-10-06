"""Exact incremental weighted triangle joins across changing relations."""

from __future__ import annotations

import math
import random
import sys

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import forbidden_imports, load_candidate, plain_containers, watch_imports


_candidate = load_candidate(__file__)
ENGINE_ROOTS = ("sqlite3", "_sqlite3", "duckdb", "_duckdb", "sqlalchemy", "apsw",
                "polars", "pandas", "datafusion", "pyarrow", "sqlglot", "pandasql")


def _dot(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(key, 0) for key, weight in left.items())


def _indexed_join(problem):
    results = []
    for scenario in problem["scenarios"]:
        outgoing = {name: {} for name in ("R", "S", "T")}
        incoming = {name: {} for name in ("R", "S", "T")}

        def store(name, source, target, weight):
            forward = outgoing[name].setdefault(source, {})
            reverse = incoming[name].setdefault(target, {})
            if weight == 0:
                forward.pop(target, None)
                reverse.pop(source, None)
            else:
                forward[target] = weight
                reverse[source] = weight

        for name, rows in scenario["relations"].items():
            for source, target, weight in rows:
                store(name, source, target, weight)
        total = sum(
            weight * _dot(outgoing["S"].get(b, {}), incoming["T"].get(a, {}))
            for a, row in outgoing["R"].items()
            for b, weight in row.items()
        )
        answers = []
        for operation in scenario["operations"]:
            if operation[0] == "query":
                answers.append(total)
                continue
            kind, name, source, target, *value = operation
            weight = value[0] if kind == "set" else 0
            previous = outgoing[name].get(source, {}).get(target, 0)
            if name == "R":
                coefficient = _dot(outgoing["S"].get(target, {}), incoming["T"].get(source, {}))
            elif name == "S":
                coefficient = _dot(incoming["R"].get(source, {}), outgoing["T"].get(target, {}))
            else:
                coefficient = _dot(incoming["S"].get(source, {}), outgoing["R"].get(target, {}))
            # The changed relation occurs once in each triangle, so this delta
            # remains exact for replacement, signed weights, and deletion.
            total += (weight - previous) * coefficient
            store(name, source, target, weight)
        results.append(tuple(answers))
    return tuple(results)


def _recompute_join(problem):
    """Independent oracle: enumerate two-edge paths at each checkpoint."""
    results = []
    for scenario in problem["scenarios"]:
        relations = {
            name: {(source, target): weight for source, target, weight in rows}
            for name, rows in scenario["relations"].items()
        }
        answers = []
        for operation in scenario["operations"]:
            if operation[0] != "query":
                kind, name, source, target, *value = operation
                if kind == "set" and value[0] != 0:
                    relations[name][source, target] = value[0]
                else:
                    relations[name].pop((source, target), None)
                continue
            by_b = {}
            for (a, b), weight in relations["R"].items():
                by_b.setdefault(b, []).append((a, weight))
            total = 0
            for (b, c), weight_s in relations["S"].items():
                for a, weight_r in by_b.get(b, ()):
                    total += weight_r * weight_s * relations["T"].get((c, a), 0)
            answers.append(total)
        results.append(tuple(answers))
    return tuple(results)


class IncrementalMultiwayJoinTask:
    name = "incremental_multiway_join"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 1200
    grading_cases = (1200, 2400)

    def generate_problem(self, n=1200, random_seed=0):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a non-negative integer")
        rng = random.Random(random_seed)
        side = max(8, math.isqrt(n) * 2)
        scenarios = []
        for family in ("uniform", "hub_skew", "dense_bursts"):
            def endpoints():
                if family == "hub_skew" and rng.random() < 0.8:
                    return rng.randrange(max(2, side // 12)), rng.randrange(side)
                return rng.randrange(side), rng.randrange(side)

            edge_count = side * 7 if family != "dense_bursts" else side * side // 3
            initial = {}
            live = {}
            for name in ("R", "S", "T"):
                rows = {}
                for _ in range(edge_count):
                    pair = endpoints()
                    rows[pair] = rng.choice((-9, -3, -1, 1, 2, 5, 11))
                initial[name] = tuple((a, b, weight) for (a, b), weight in sorted(rows.items()))
                live[name] = dict(rows)
            operations = [("query",)]
            for index in range(n):
                # Family-specific phases put different pressure on eager delta
                # maintenance, deferred batches, and materialized intermediates.
                phase = (index // max(1, n // 4)) % 4
                interval = 9 if family == "uniform" else 23
                if family == "dense_bursts":
                    interval = 3 if phase in (1, 3) else 97
                if index % interval == interval - 1:
                    operations.append(("query",))
                    continue
                name = ("R", "S", "T")[rng.randrange(3)]
                if family == "dense_bursts" and phase == 2:
                    name = "R"
                if live[name] and rng.random() < 0.65:
                    source, target = rng.choice(tuple(live[name]))
                else:
                    source, target = endpoints()
                if rng.random() < 0.23:
                    operations.append(("delete", name, source, target))
                    live[name].pop((source, target), None)
                else:
                    weight = rng.randrange(-15, 16)
                    operations.append(("set", name, source, target, weight))
                    if weight:
                        live[name][source, target] = weight
                    else:
                        live[name].pop((source, target), None)
            operations.append(("query",))
            scenarios.append({"family": family, "relations": initial, "operations": tuple(operations)})
        return {"scenarios": tuple(scenarios)}

    def solve(self, problem):
        return _indexed_join(problem)

    def candidate_solve(self, problem):
        self.policy_violations = ()
        loaded = set(sys.modules)
        with watch_imports(ENGINE_ROOTS) as imported_during:
            result = _candidate.solve(problem, self.solve)
        violations = forbidden_imports(_candidate, ENGINE_ROOTS, loaded, imported_during)
        if violations:
            self.policy_violations = violations
            print(f"candidate uses forbidden relational engine imports: {', '.join(violations)}")
            return None
        return result

    def is_solution(self, problem, proposed):
        if not plain_containers(proposed) or type(proposed) not in (tuple, list) or len(proposed) != len(problem["scenarios"]):
            return False
        if any(
            not isinstance(answers, (tuple, list)) or any(type(value) is not int for value in answers)
            for answers in proposed
        ):
            return False
        return tuple(tuple(answers) for answers in proposed) == _recompute_join(problem)


TASK = IncrementalMultiwayJoinTask()
