"""Join events to the latest eligible per-entity dimension version."""

from __future__ import annotations

import random
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


def _join(problem):
    output = []
    dimensions = problem["dimensions"]
    for event_id, entity, event_time in problem["events"]:
        best_key = None
        best_value = None
        for input_position, row in enumerate(dimensions):
            row_entity, timestamp, sequence, value = row
            if row_entity != entity or timestamp > event_time:
                continue
            key = (timestamp, sequence, input_position)
            if best_key is None or key > best_key:
                best_key = key
                best_value = value
        output.append((event_id, best_value if best_key is not None else None))
    return tuple(output)


def _verify_join(problem):
    """Sorted per-entity histories and binary search instead of full scans."""
    from bisect import bisect_right
    histories = {}
    for position, (entity, timestamp, sequence, value) in enumerate(problem["dimensions"]):
        histories.setdefault(entity, []).append((timestamp, sequence, position, value))
    for history in histories.values():
        history.sort()
    result = []
    for identifier, entity, timestamp in problem["events"]:
        history = histories.get(entity, ())
        # Timestamp-only insertion locates the final version at that time,
        # including the greatest sequence and physical input position.
        position = bisect_right(history, timestamp, key=lambda row: row[0])
        result.append((identifier, history[position - 1][3] if position else None))
    return tuple(result)

class TemporalAsOfJoinTask:
    name = "temporal_asof_join"
    task_version = "1.1.2"
    display_name = TASK_CATALOG[name].display_name
    default_n = 700
    grading_cases = (700, 1400)

    def generate_problem(self, n=700, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        entity_count = max(1, min(120, n // 35 + 1))
        dimensions = []
        for index in range(n):
            entity = rng.randrange(entity_count)
            timestamp = rng.randrange(max(1, n // 3 + 1))
            sequence = rng.randrange(4)
            dimensions.append((entity, timestamp, sequence, f"v:{index}"))
            if index % 67 == 0:
                dimensions.append((entity, timestamp, sequence, f"tie:{index}"))
        rng.shuffle(dimensions)
        events = tuple(
            (event_id, rng.randrange(entity_count), rng.randrange(-10, max(1, n // 3 + 30)))
            for event_id in range(n)
        )
        return {"dimensions": tuple(dimensions), "events": events}

    def solve(self, problem):
        return _join(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        try:
            return _same_materialized(proposed, _verify_join(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = TemporalAsOfJoinTask()
