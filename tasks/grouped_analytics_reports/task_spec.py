"""Build an in-memory SQLite database and return exact analytics reports."""

from __future__ import annotations

import random
import sqlite3
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            type(actual) in (tuple, list)
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _reports_sqlite(problem):
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE events (user_id INTEGER NOT NULL, category TEXT, value INTEGER, note TEXT)"
        )
        connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", problem["rows"])
        connection.execute("CREATE INDEX events_category ON events(category)")
        connection.execute("CREATE INDEX events_user_category ON events(user_id, category)")
        categories = tuple(
            connection.execute(
                """SELECT category, COUNT(*), COUNT(value), SUM(value), COUNT(DISTINCT note)
                   FROM events GROUP BY category
                   ORDER BY SUM(value) DESC, category IS NOT NULL, category"""
            )
        )
        duplicates = tuple(
            connection.execute(
                """SELECT user_id, category, COUNT(*) AS copies
                   FROM events GROUP BY user_id, category HAVING copies > 1
                   ORDER BY copies DESC, user_id, category IS NOT NULL, category"""
            )
        )
        totals = tuple(
            connection.execute(
                """SELECT user_id, SUM(value) AS total
                   FROM events GROUP BY user_id HAVING total >= ?
                   ORDER BY total DESC, user_id""",
                (problem["minimum_total"],),
            )
        )
        return categories, duplicates, totals
    finally:
        connection.close()


class SQLiteAnalyticsReportsTask:
    name = "grouped_analytics_reports"
    task_version = "1.1.2"
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

    def solve(self, problem):
        return _reports_sqlite(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed, *, reference_output=None):
        try:
            if reference_output is None:
                reference_output = self.solve(problem)
            return _same_materialized(proposed, reference_output)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = SQLiteAnalyticsReportsTask()
