"""Join events to the latest eligible per-entity dimension version."""

from __future__ import annotations

import random
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


class TemporalAsOfJoinTask:
    name = "temporal_asof_join"
    task_version = "1.1.1"
    display_name = "Temporal Per-Entity As-Of Join"
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
            return _same_materialized(proposed, _join(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = TemporalAsOfJoinTask()
