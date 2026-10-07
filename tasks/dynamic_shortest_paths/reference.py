"""Reference implementation for dynamic_shortest_paths; copied into fresh run candidates."""

from __future__ import annotations

import heapq


def _distance(adjacency, source, target):
    distances = {source: 0}
    queue = [(0, source)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        if vertex == target:
            return distance
        for neighbor, weight in adjacency[vertex].items():
            proposal = distance + weight
            if neighbor not in distances or proposal < distances[neighbor]:
                distances[neighbor] = proposal
                heapq.heappush(queue, (proposal, neighbor))
    return None


def _forward_queries(problem):
    results = []
    for scenario in problem["scenarios"]:
        adjacency = [{} for _ in range(scenario["node_count"])]
        for source, target, weight in scenario["edges"]:
            adjacency[source][target] = weight
        answers = []
        for operation in scenario["operations"]:
            kind, source, target, *value = operation
            if kind == "set":
                adjacency[source][target] = value[0]
            elif kind == "delete":
                adjacency[source].pop(target, None)
            else:
                answers.append(_distance(adjacency, source, target))
        results.append(tuple(answers))
    return tuple(results)


def solve(problem):
    return _forward_queries(problem)
