"""Reference implementation for articulation_points; copied into fresh run candidates."""

def components(n, adjacency, removed=-1):
    seen = {removed}
    count = 0
    for start in range(n):
        if start in seen:
            continue
        count += 1
        seen.add(start)
        stack = [start]
        while stack:
            for v in adjacency[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return count


def adjacency_of(problem):
    adjacency = [[] for _ in range(problem["num_nodes"])]
    for u, v in problem["edges"]:
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency


def solve(problem):
    n = problem["num_nodes"]
    adjacency = adjacency_of(problem)
    base = components(n, adjacency)
    return {"articulation_points": [v for v in range(n)
            if components(n, adjacency, v) > base]}
