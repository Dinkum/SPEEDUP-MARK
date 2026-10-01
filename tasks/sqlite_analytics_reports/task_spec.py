"""Build an in-memory SQLite database and return exact analytics reports."""

from __future__ import annotations

import random
import sqlite3
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


def _reports_python(problem):
    category_groups = {}
    duplicate_groups = {}
    user_values = {}
    for user_id, category, value, note in problem["rows"]:
        category_groups.setdefault(category, []).append((value, note))
        duplicate_groups[(user_id, category)] = duplicate_groups.get((user_id, category), 0) + 1
        if value is not None:
            user_values[user_id] = user_values.get(user_id, 0) + value

    categories = []
    for category, rows in category_groups.items():
        values = [value for value, _ in rows if value is not None]
        distinct_notes = {note for _, note in rows if note is not None}
        categories.append((category, len(rows), len(values), sum(values) if values else None, len(distinct_notes)))
    categories.sort(
        key=lambda row: (row[3] is None, -(row[3] or 0), row[0] is not None, row[0] or "")
    )

    duplicates = [
        (user_id, category, copies)
        for (user_id, category), copies in duplicate_groups.items()
        if copies > 1
    ]
    duplicates.sort(key=lambda row: (-row[2], row[0], row[1] is not None, row[1] or ""))
    totals = [
        (user_id, total)
        for user_id, total in user_values.items()
        if total >= problem["minimum_total"]
    ]
    totals.sort(key=lambda row: (-row[1], row[0]))
    return tuple(categories), tuple(duplicates), tuple(totals)


class SQLiteAnalyticsReportsTask:
    name = "sqlite_analytics_reports"
    task_version = "1.1.1"
    display_name = "SQLite Analytics Report Workload"
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

    def is_solution(self, problem, proposed):
        try:
            return _same_materialized(proposed, _reports_python(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = SQLiteAnalyticsReportsTask()
