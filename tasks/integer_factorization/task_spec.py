"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

import math


def prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    return all(x % d for d in range(3, math.isqrt(x) + 1, 2))


class Task:
    name = "integer_factorization"
    task_version = "1.1.0"
    display_name = "Exact Semiprime Factorization"
    default_n = 400000
    grading_cases = (400000, 800000)

    def generate_problem(self, n=400000, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        scale = max(20, n)
        p = rng.randrange(scale // 4, scale)
        while not prime(p):
            p += 1
        q = rng.randrange(3 if random_seed % 3 == 0 else scale // 2, 2 * scale)
        while not prime(q) or q == p:
            q += 1
        return {"composite": p * q}

    def solve(self, problem):
        value = problem["composite"]
        if value % 2 == 0:
            return {"p": 2, "q": value // 2}
        for p in range(3, math.isqrt(value) + 1, 2):
            if value % p == 0:
                return {"p": p, "q": value // p}
        raise ValueError("expected semiprime")

    def is_solution(self, problem, proposed):
        return (type(proposed) is dict and set(proposed) == {"p", "q"}
                and type(proposed["p"]) is int and type(proposed["q"]) is int
                and proposed["p"] < proposed["q"]
                and proposed["p"] * proposed["q"] == problem["composite"]
                and prime(proposed["p"]) and prime(proposed["q"]))

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
