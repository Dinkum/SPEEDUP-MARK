"""Reference implementation for labeled_graph_isomorphism; copied into fresh run candidates."""

from __future__ import annotations

def _adjacency(graph):
    size = len(graph["labels"])
    adjacency = [set() for _ in range(size)]
    for left, right in graph["edges"]:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _find_mapping(problem):
    left, right = problem["left"], problem["right"]
    if len(left["labels"]) != len(right["labels"]):
        return None
    size = len(left["labels"])
    left_adj, right_adj = _adjacency(left), _adjacency(right)
    candidates = []
    for node in range(size):
        matches = [
            other
            for other in range(size)
            if left["labels"][node] == right["labels"][other]
            and len(left_adj[node]) == len(right_adj[other])
        ]
        if not matches:
            return None
        candidates.append(matches)
    order = []
    remaining = set(range(size))
    while remaining:
        node = min(remaining, key=lambda v: (-len(left_adj[v].intersection(order)),
                                            len(candidates[v]), -len(left_adj[v]), v))
        order.append(node)
        remaining.remove(node)
    mapping = [-1] * size
    used = set()

    def visit(depth):
        if depth == size:
            return True
        node = order[depth]
        for other in candidates[node]:
            if other in used:
                continue
            if any(
                ((prior in left_adj[node]) != (mapping[prior] in right_adj[other]))
                for prior in range(size)
                if mapping[prior] >= 0
            ):
                continue
            mapping[node] = other
            used.add(other)
            if visit(depth + 1):
                return True
            used.remove(other)
            mapping[node] = -1
        return False

    return tuple(mapping) if visit(0) else None


def solve(problem):
    return _find_mapping(problem)
