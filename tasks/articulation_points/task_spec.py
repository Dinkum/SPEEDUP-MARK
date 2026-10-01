"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

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


class Task:
    name = "articulation_points"
    task_version = "1.1.0"
    display_name = "Articulation Points Across Graph Components"
    default_n = 300
    grading_cases = (300, 600)

    def generate_problem(self, n=300, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        edges = set()
        # Chains of locally dense blocks retain cut vertices; some seeds
        # disconnect blocks so component-count semantics are exercised.
        for v in range(1, n):
            if v % 19 != 0 or random_seed % 2:
                edges.add((v - 1, v))
        for start in range(0, n, 19):
            end = min(n, start + 19)
            for _ in range((end - start) * (1 + random_seed % 4)):
                u, v = sorted((rng.randrange(start, end), rng.randrange(start, end)))
                if u != v:
                    edges.add((u, v))
        return {"num_nodes": n, "edges": [list(e) for e in sorted(edges)]}

    def solve(self, problem):
        n = problem["num_nodes"]
        adjacency = adjacency_of(problem)
        base = components(n, adjacency)
        return {"articulation_points": [v for v in range(n)
                if components(n, adjacency, v) > base]}

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"articulation_points"}:
            return False
        points = proposed["articulation_points"]
        if type(points) is not list or any(type(v) is not int for v in points):
            return False
        # Iterative low-link DFS avoids recursion limits and provides a
        # different correctness algorithm from remove-and-recount.
        n, adjacency = problem["num_nodes"], adjacency_of(problem)
        discovery, low, parent, children = [-1] * n, [0] * n, [-1] * n, [0] * n
        found, clock = set(), 0
        for root in range(n):
            if discovery[root] != -1:
                continue
            discovery[root] = low[root] = clock
            clock += 1
            stack = [(root, iter(adjacency[root]))]
            while stack:
                u, neighbors = stack[-1]
                v = next(neighbors, None)
                if v is None:
                    stack.pop()
                    p = parent[u]
                    if p == -1:
                        if children[u] > 1:
                            found.add(u)
                    else:
                        low[p] = min(low[p], low[u])
                        if parent[p] != -1 and low[u] >= discovery[p]:
                            found.add(p)
                    continue
                if discovery[v] == -1:
                    parent[v] = u
                    children[u] += 1
                    discovery[v] = low[v] = clock
                    clock += 1
                    stack.append((v, iter(adjacency[v])))
                elif v != parent[u]:
                    low[u] = min(low[u], discovery[v])
        return points == sorted(found)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
