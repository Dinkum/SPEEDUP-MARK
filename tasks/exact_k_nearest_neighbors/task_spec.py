"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

class Task:
    name = "exact_k_nearest_neighbors"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 700
    grading_cases = (700, 1400)

    def generate_problem(self, n=700, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        dim = (2, 4, 8)[random_seed % 3]
        points = []
        for i in range(n):
            center = (i % 7) * 1000 if random_seed % 2 else 0
            points.append([center + rng.randrange(-100, 101) for _ in range(dim)])
        queries = [points[rng.randrange(n)][:] if i % 9 == 0
                   else [rng.randrange(-200, 6201 if random_seed % 2 else 201) for _ in range(dim)]
                   for i in range(max(1, n // 7))]
        return {"points": points, "queries": queries, "k": min(n, (1, 5, 13)[random_seed % 3])}

    def solve(self, problem):
        points, k = problem["points"], problem["k"]
        result = []
        for query in problem["queries"]:
            distances = [(sum((a - b) ** 2 for a, b in zip(point, query)), i)
                         for i, point in enumerate(points)]
            result.append([i for _, i in sorted(distances)[:k]])
        return {"indices": result}

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"indices"}:
            return False
        rows, points, k = proposed["indices"], problem["points"], problem["k"]
        if type(rows) is not list or len(rows) != len(problem["queries"]):
            return False
        for row, query in zip(rows, problem["queries"]):
            if (type(row) is not list or len(row) != k
                    or any(type(i) is not int or not 0 <= i < len(points) for i in row)
                    or len(set(row)) != k):
                return False
            def key(i):
                return (sum((points[i][j] - query[j]) ** 2 for j in range(len(query))), i)
            ranked = [key(i) for i in row]
            if ranked != sorted(ranked):
                return False
            chosen = set(row)
            if any(key(i) < ranked[-1] for i in range(len(points)) if i not in chosen):
                return False
        return True

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
