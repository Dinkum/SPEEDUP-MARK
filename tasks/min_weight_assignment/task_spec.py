"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


class Task:
    name = "min_weight_assignment"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 70
    grading_cases = (70, 110)

    def generate_problem(self, n=70, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        rows, cols = [rng.randrange(-100, 100) for _ in range(n)], [rng.randrange(-100, 100) for _ in range(n)]
        costs = [[rng.randrange(-500, 501) if random_seed % 3 == 0
                  else rows[i] + cols[j] + rng.randrange(3 if random_seed % 3 == 1 else 100)
                  for j in range(n)] for i in range(n)]
        return {"costs": costs}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"assignment"}:
            return False
        assignment, costs = proposed["assignment"], problem["costs"]
        n = len(costs)
        if (type(assignment) is not list or len(assignment) != n
                or any(type(j) is not int for j in assignment)
                or sorted(assignment) != list(range(n))):
            return False
        # Any better permutation decomposes into an improving exchange cycle.
        # Bellman-Ford detects such a negative cycle independently of Hungarian.
        distance = [0] * n
        for _ in range(n):
            changed = False
            for i in range(n):
                for j in range(n):
                    alt = distance[i] + costs[i][assignment[j]] - costs[i][assignment[i]]
                    if alt < distance[j]:
                        distance[j] = alt
                        changed = True
            if not changed:
                return True
        return False


TASK = Task()
