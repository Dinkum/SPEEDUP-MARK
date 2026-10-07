"""Reference implementation for minimum_spanning_tree; copied into fresh run candidates."""

import heapq


def solve(problem):
    n, edges = problem["num_nodes"], problem["edges"]
    adjacency = [[] for _ in range(n)]
    for i, (u, v, weight) in enumerate(edges):
        adjacency[u].append((weight, v, i))
        adjacency[v].append((weight, u, i))
    seen, heap, selected = {0}, list(adjacency[0]), []
    heapq.heapify(heap)
    while heap and len(seen) < n:
        _, v, i = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        selected.append(i)
        for item in adjacency[v]:
            if item[1] not in seen:
                heapq.heappush(heap, item)
    return {"edge_indices": sorted(selected)}
