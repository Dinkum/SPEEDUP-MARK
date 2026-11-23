"""Exact incremental weighted triangle joins across changing relations."""

from __future__ import annotations

import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)


ENGINE_ROOTS = ("sqlite3", "_sqlite3", "duckdb", "_duckdb", "sqlalchemy", "apsw",
                "polars", "pandas", "datafusion", "pyarrow", "sqlglot", "pandasql")


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
    forbidden_import_roots = ENGINE_ROOTS
    name = "incremental_multiway_join"
    task_version = "2.2.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 4000
    grading_cases = (4000, 8000)
    uses_reference_output = True

    def correctness_problem(self, seed):
        return self.generate_problem(32, seed)

    def generate_problem(self, n=4000, random_seed=0):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a non-negative integer")
        rng = random.Random(random_seed)
        side = max(8, math.isqrt(n) * 8)
        scenarios = []
        names = ("R", "S", "T")
        targets = dict(zip(names, ("S", "T", "R")))
        for family in ("uniform", "hub_skew", "dense_bursts"):
            updated = rng.choice(names)
            forward = targets[updated]
            domains = {name: rng.sample(range(side), side) for name in names}
            hubs = max(2, math.isqrt(side) // 2)
            hot_edges = tuple(
                (a, b) for a in domains[updated][:hubs] for b in domains[forward][:hubs]
            )

            def endpoints(name):
                return rng.choice(domains[name]), rng.choice(domains[targets[name]])

            initial, starting_pairs = {}, {}
            for name in names:
                pairs = {endpoints(name) for _ in range(side * 5)}
                if family != "uniform" and name != updated:
                    broad = domains[targets[name] if name == forward else name]
                    for vertex in domains[forward if name == forward else updated][:hubs]:
                        neighbors = rng.sample(broad, side * (4 if family == "dense_bursts" else 3) // 5)
                        pairs.update((vertex, neighbor) if name == forward else (neighbor, vertex)
                                     for neighbor in neighbors)
                rows = {pair: rng.choice((-9, -3, -1, 1, 2, 5, 11)) for pair in sorted(pairs)}
                initial[name] = tuple((a, b, weight) for (a, b), weight in rows.items())
                starting_pairs[name] = tuple(rows)
            operations = [("query",)]
            for index in range(n):
                block_offset = index % 113
                if family == "dense_bursts" and block_offset == 0:
                    burst_edges = rng.sample(hot_edges, min(6, len(hot_edges)))
                phase = index * 4 // n
                interval = (9 if family == "uniform" else 11 if family == "hub_skew"
                            else 3 if phase % 2 else 113)
                if (index + 1) % interval == 0:
                    operations.append(("query",))
                    continue
                if (family == "hub_skew" and rng.random() < 0.95
                        or family == "dense_bursts" and block_offset >= 2):
                    name = updated
                    source, target = rng.choice(hot_edges if family == "hub_skew" else burst_edges)
                else:
                    name = rng.choice((forward, targets[forward]) if family == "dense_bursts" else names)
                    source, target = (rng.choice(starting_pairs[name]) if rng.random() < 0.65
                                      else endpoints(name))
                if rng.random() < 0.20:
                    operations.append(("delete", name, source, target))
                else:
                    operations.append(("set", name, source, target, rng.randrange(-15, 16)))
            operations.append(("query",))
            scenarios.append({"family": family, "relations": initial, "operations": tuple(operations)})
        return {"scenarios": tuple(scenarios)}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed, *, reference_output=None):
        if not plain_containers(proposed) or type(proposed) not in (tuple, list) or len(proposed) != len(problem["scenarios"]):
            return False
        if any(
            not isinstance(answers, (tuple, list)) or any(type(value) is not int for value in answers)
            for answers in proposed
        ):
            return False
        expected = _recompute_join(problem) if reference_output is None else reference_output
        return tuple(tuple(answers) for answers in proposed) == expected


TASK = IncrementalMultiwayJoinTask()
