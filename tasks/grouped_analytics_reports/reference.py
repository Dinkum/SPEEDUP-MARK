"""Reference implementation for grouped_analytics_reports; copied into fresh run candidates."""

from __future__ import annotations

import sqlite3


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


def solve(problem):
    return _reports_sqlite(problem)
