"""Reference implementation for temporal_asof_join; copied into fresh run candidates."""

from __future__ import annotations

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


def solve(problem):
    return _join(problem)
