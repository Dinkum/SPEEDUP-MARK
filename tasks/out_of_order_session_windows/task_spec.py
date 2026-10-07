"""Sessionize out-of-order events under an explicit watermark policy."""

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


def _verify_sessions(problem):
    """Recluster retained timestamps, independently of interval mutation."""
    retained, trace, maximum = {}, [], None
    gap, lateness = problem["gap"], problem["allowed_lateness"]
    def clusters(timestamps):
        groups = []
        for timestamp in sorted(timestamps):
            if groups and timestamp <= groups[-1][-1] + gap:
                groups[-1].append(timestamp)
            else:
                groups.append([timestamp])
        return groups
    for identifier, key, timestamp in problem["events"]:
        maximum = timestamp if maximum is None else max(maximum, timestamp)
        watermark = maximum - lateness
        if timestamp < watermark:
            trace.append(("drop", identifier, key, timestamp, watermark))
        else:
            current = retained.setdefault(key, [])
            touched = sum(group[0] - gap <= timestamp <= group[-1] + gap
                          for group in clusters(current))
            current.append(timestamp)
            merged = next(group for group in clusters(current) if timestamp in group)
            trace.append(("upsert", identifier, key, merged[0], merged[-1], len(merged), touched, watermark))
        for entity in sorted(retained):
            live = []
            for group in clusters(retained[entity]):
                if group[-1] + gap < watermark:
                    trace.append(("emit", entity, group[0], group[-1], len(group), watermark))
                else:
                    live.extend(group)
            retained[entity] = live
    for entity in sorted(retained):
        for group in clusters(retained[entity]):
            trace.append(("emit", entity, group[0], group[-1], len(group), None))
    return tuple(trace)


class OutOfOrderSessionWindowsTask:
    name = "out_of_order_session_windows"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 2400
    grading_cases = (2400, 4800)

    def generate_problem(self, n=2400, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        events = []
        for event_id in range(n):
            key = f"user-{rng.randrange(max(1, min(24, n // 20 + 1)))}"
            timestamp = event_id * 3 + rng.randrange(-18, 19)
            events.append((event_id, key, timestamp))
        # Local swaps create normal disorder; injected old records exercise drops.
        for index in range(0, n - 1, 7):
            swap = min(n - 1, index + rng.randrange(1, min(7, n - index)))
            events[index], events[swap] = events[swap], events[index]
        for index in range(97, n, 211):
            event_id, key, timestamp = events[index]
            events[index] = (event_id, key, timestamp - 240)
        return {"events": tuple(events), "gap": 20, "allowed_lateness": 45}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            return _same_materialized(proposed, _verify_sessions(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = OutOfOrderSessionWindowsTask()
