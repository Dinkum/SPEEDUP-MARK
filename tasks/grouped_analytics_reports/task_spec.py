"""Build an in-memory SQLite database and return exact analytics reports."""

from __future__ import annotations

import random
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            type(actual) in (tuple, list)
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


class SQLiteAnalyticsReportsTask:
    name = "grouped_analytics_reports"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    uses_reference_output = True
    default_n = 5000
    grading_cases = (5000, 10000)

    def generate_problem(self, n=5000, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        categories = (None, "alpha", "beta", "gamma", "delta", "epsilon")
        notes = (None, None, "ok", "retry", "manual", "imported")
        rows = []
        for index in range(n):
            row = (
                rng.randrange(max(1, n // 25 + 1)),
                rng.choice(categories),
                None if index % 13 == 0 else rng.randrange(-20, 101),
                rng.choice(notes),
            )
            rows.append(row)
            if index % 41 == 0:
                rows.append(row)
        return {"rows": tuple(rows), "minimum_total": 450}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed, *, reference_output=None):
        try:
            if reference_output is None:
                reference_output = self.solve(problem)
            return _same_materialized(proposed, reference_output)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = SQLiteAnalyticsReportsTask()
