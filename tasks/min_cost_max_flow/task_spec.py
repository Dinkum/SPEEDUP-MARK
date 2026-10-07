"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
residual = _reference.residual


class Task:
    name = "min_cost_max_flow"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 100
    grading_cases = (100, 180)

    def generate_problem(self, n=100, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        n = max(2, n)
        rng = random.Random(random_seed)
        pairs = {(u, u + 1) for u in range(n - 1)}
        for _ in range(n * (3 + random_seed % 5)):
            u, v = sorted((rng.randrange(n), rng.randrange(n)))
            if u != v:
                pairs.add((u, v))
        edges = [[u, v, rng.randrange(1, 15), rng.randrange(0, 3 if random_seed % 2 else 40)]
                 for u, v in sorted(pairs)]
        rng.shuffle(edges)
        return {"num_nodes": n, "source": 0, "sink": n - 1, "edges": edges}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"flow"}:
            return False
        flow, edges = proposed["flow"], problem["edges"]
        if (type(flow) is not list or len(flow) != len(edges)
                or any(type(f) is not int or not 0 <= f <= edge[2] for f, edge in zip(flow, edges))):
            return False
        n, source, sink = problem["num_nodes"], problem["source"], problem["sink"]
        balance = [0] * n
        for f, (u, v, _, _) in zip(flow, edges):
            balance[u] -= f
            balance[v] += f
        if (balance[source] > 0 or balance[sink] != -balance[source]
                or any(balance[v] for v in range(n) if v not in (source, sink))):
            return False
        arcs = residual(problem, flow)
        adjacency = [[] for _ in range(n)]
        for u, v, *_ in arcs:
            adjacency[u].append(v)
        seen, stack = {source}, [source]
        while stack:
            for v in adjacency[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if sink in seen:
            return False
        # A feasible max flow is minimum cost iff its residual network has
        # no negative cycle, including cycles disconnected from the source.
        distance = [0] * n
        for _ in range(n):
            changed = False
            for u, v, _, cost, _, _ in arcs:
                if distance[u] + cost < distance[v]:
                    distance[v] = distance[u] + cost
                    changed = True
            if not changed:
                return True
        return False


TASK = Task()
