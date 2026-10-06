"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

class Task:
    name = "matrix_multiplication"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 45
    grading_cases = (45, 75)

    def generate_problem(self, n=45, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        shape = [(n, n, n), (max(1, n // 3), 2 * n, n), (n, max(1, n // 2), 2 * n)][random_seed % 3]
        rows, inner, columns = shape
        def value():
            return 0 if random_seed % 4 == 0 and rng.random() < 0.9 else rng.randrange(-128, 129)
        return {"A": [[value() for _ in range(inner)] for _ in range(rows)],
                "B": [[value() for _ in range(columns)] for _ in range(inner)]}

    def solve(self, problem):
        a, b = problem["A"], problem["B"]
        return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
                 for j in range(len(b[0]))] for i in range(len(a))]

    def is_solution(self, problem, proposed):
        a, b = problem["A"], problem["B"]
        if (type(proposed) is not list or len(proposed) != len(a)
                or any(type(row) is not list or len(row) != len(b[0])
                       or any(type(x) is not int for x in row) for row in proposed)):
            return False
        # Accumulate scaled rows instead of reproducing the baseline dot loop.
        for i, row in enumerate(a):
            expected = [0] * len(b[0])
            for coefficient, other in zip(row, b):
                for j, value in enumerate(other):
                    expected[j] += coefficient * value
            if proposed[i] != expected:
                return False
        return True

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
