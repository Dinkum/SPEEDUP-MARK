"""Exact shortest-distance queries on changing directed weighted graphs."""

from __future__ import annotations

import heapq
import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)


def _bidirectional(forward, reverse, source, target):
    """Independent verifier search, meeting between opposite graph directions."""
    if source == target:
        return 0
    distances = ({source: 0}, {target: 0})
    queues = ([(0, source)], [(0, target)])
    settled = (set(), set())
    graphs = (forward, reverse)
    best = math.inf
    while queues[0] and queues[1]:
        for direction in (0, 1):
            queue = queues[direction]
            while queue and (
                queue[0][1] in settled[direction]
                or queue[0][0] != distances[direction][queue[0][1]]
            ):
                heapq.heappop(queue)
        if not queues[0] or not queues[1]:
            break
        # Any undiscovered improvement must extend beyond both live frontiers.
        if queues[0][0][0] + queues[1][0][0] >= best:
            break
        direction = 0 if queues[0][0][0] <= queues[1][0][0] else 1
        distance, vertex = heapq.heappop(queues[direction])
        settled[direction].add(vertex)
        opposite = distances[1 - direction]
        if vertex in opposite:
            best = min(best, distance + opposite[vertex])
        for neighbor, weight in graphs[direction][vertex].items():
            candidate = distance + weight
            if candidate < distances[direction].get(neighbor, math.inf):
                distances[direction][neighbor] = candidate
                heapq.heappush(queues[direction], (candidate, neighbor))
            if neighbor in opposite:
                best = min(best, candidate + opposite[neighbor])
    return None if best == math.inf else best


def _checked_queries(problem):
    results = []
    for scenario in problem["scenarios"]:
        forward = [{} for _ in range(scenario["node_count"])]
        reverse = [{} for _ in range(scenario["node_count"])]
        for source, target, weight in scenario["edges"]:
            forward[source][target] = weight
            reverse[target][source] = weight
        answers = []
        for operation in scenario["operations"]:
            kind, source, target, *value = operation
            if kind == "query":
                answers.append(_bidirectional(forward, reverse, source, target))
            elif kind == "set":
                forward[source][target] = value[0]
                reverse[target][source] = value[0]
            else:
                forward[source].pop(target, None)
                reverse[target].pop(source, None)
        results.append(tuple(answers))
    return tuple(results)


class DynamicShortestPathsTask:
    name = "dynamic_shortest_paths"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 400
    grading_cases = (400, 800)

    def generate_problem(self, n=400, random_seed=0):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a non-negative integer")
        rng = random.Random(random_seed)
        size = max(8, n)
        width = math.isqrt(size)
        scenarios = []
        for family in ("grid", "hubs", "bottleneck"):
            edges = {}

            def add(source, target):
                if source != target:
                    edges[source, target] = rng.randrange(1, 31)

            if family == "grid":
                for vertex in range(size):
                    if vertex + 1 < size and vertex // width == (vertex + 1) // width:
                        add(vertex, vertex + 1)
                        add(vertex + 1, vertex)
                    if vertex + width < size:
                        add(vertex, vertex + width)
                        add(vertex + width, vertex)
            elif family == "hubs":
                for vertex in range(size):
                    add(vertex, (vertex + 1) % size)
                    add((vertex + 1) % size, vertex)
                    for _ in range(2):
                        hub = rng.randrange(max(2, size // 40))
                        add(vertex, hub)
                        add(hub, vertex)
                    add(vertex, rng.randrange(size))
            else:
                half = size // 2
                for start, stop in ((0, half), (half, size)):
                    for vertex in range(start, stop):
                        neighbor = start + (vertex - start + 1) % (stop - start)
                        add(vertex, neighbor)
                        add(neighbor, vertex)
                        for _ in range(3):
                            add(vertex, rng.randrange(start, stop))
                add(half - 1, half)
                add(half, half - 1)
            initial = tuple((source, target, weight) for (source, target), weight in sorted(edges.items()))
            update_pairs = tuple(sorted(edges))
            hot_sources = tuple(rng.randrange(size) for _ in range(3))
            hot_targets = tuple(rng.randrange(size) for _ in range(3))
            operations = [("query", 0, size - 1), ("query", size - 1, 0)]
            for index in range(n * 2):
                phase = (index // max(1, n // 2)) % 4
                # The same stream includes query reuse, mostly cold queries,
                # and concentrated update bursts; preprocessing is not free.
                query_rate = (0.95, 0.70, 0.15, 0.90)[phase]
                if rng.random() < query_rate:
                    source = rng.choice(hot_sources) if phase == 0 else rng.randrange(size)
                    target = rng.choice(hot_targets) if phase == 3 else rng.randrange(size)
                    operations.append(("query", source, target))
                else:
                    source, target = rng.choice(update_pairs)
                    if rng.random() < 0.3:
                        operations.append(("delete", source, target))
                    else:
                        operations.append(("set", source, target, rng.randrange(1, 81)))
                if family == "bottleneck" and index % max(1, n // 3) == 0:
                    # Close the only crossing in both directions, verify the
                    # disconnected state, then reopen with different weights.
                    half = size // 2
                    operations.extend((
                        ("delete", half - 1, half),
                        ("delete", half, half - 1),
                        ("query", 0, size - 1),
                        ("query", size - 1, 0),
                        ("set", half - 1, half, rng.randrange(1, 81)),
                        ("set", half, half - 1, rng.randrange(1, 81)),
                        ("query", 0, size - 1),
                    ))
            scenarios.append({
                "family": family,
                "node_count": size,
                "edges": initial,
                "operations": tuple(operations),
            })
        return {"scenarios": tuple(scenarios)}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        if not plain_containers(proposed) or type(proposed) not in (tuple, list) or len(proposed) != len(problem["scenarios"]):
            return False
        if any(
            not isinstance(answers, (tuple, list))
            or any(value is not None and type(value) is not int for value in answers)
            for answers in proposed
        ):
            return False
        return tuple(tuple(answers) for answers in proposed) == _checked_queries(problem)


TASK = DynamicShortestPathsTask()
