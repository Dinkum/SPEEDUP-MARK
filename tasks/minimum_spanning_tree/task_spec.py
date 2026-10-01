"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

import heapq


class Task:
    name = "minimum_spanning_tree"
    task_version = "1.1.0"
    display_name = "Weighted Minimum Spanning Tree"
    default_n = 500
    grading_cases = (500, 1000)

    def generate_problem(self, n=500, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        pairs = {(rng.randrange(v), v) for v in range(1, n)}
        for _ in range(n * (4, 12, 25)[random_seed % 3]):
            u, v = sorted((rng.randrange(n), rng.randrange(n)))
            if u != v:
                pairs.add((u, v))
        edges = [[u, v, rng.randrange(-100, 1001)] for u, v in sorted(pairs)]
        rng.shuffle(edges)
        return {"num_nodes": n, "edges": edges}

    def solve(self, problem):
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

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"edge_indices"}:
            return False
        selected = proposed["edge_indices"]
        n, edges = problem["num_nodes"], problem["edges"]
        if (type(selected) is not list or len(selected) != n - 1
                or any(type(i) is not int or not 0 <= i < len(edges) for i in selected)
                or selected != sorted(set(selected))):
            return False
        # Kruskal verifies the optimum independently of the Prim baseline.
        parent = list(range(n))
        def find(x):
            while x != parent[x]:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        total = 0
        for i in selected:
            u, v, w = edges[i]
            a, b = find(u), find(v)
            if a == b:
                return False
            parent[a] = b
            total += w
        parent = list(range(n))
        optimum = 0
        for u, v, w in sorted(edges, key=lambda e: e[2]):
            a, b = find(u), find(v)
            if a != b:
                parent[a] = b
                optimum += w
        return total == optimum

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
