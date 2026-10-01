"""A dependency-free discrete logarithm challenge inspired by AlgoTune."""

import math
import random

from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _prime(value):
    return value >= 2 and all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


def _factors(value):
    factors = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    return factors


class DiscreteLogTask:
    name = "discrete_log"
    task_version = "1.1.0"
    display_name = "Prime-Field Discrete Logarithm"
    default_n = 300000
    grading_cases = (300000, 600000)

    def generate_problem(self, n=300000, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        p = max(7, n) + rng.randrange(max(1, n // 4))
        while not _prime(p):
            p += 1
        divisors = _factors(p - 1)
        while True:
            g = rng.randrange(2, p)
            if all(pow(g, (p - 1) // factor, p) != 1 for factor in divisors):
                break
        return {"p": p, "g": g, "h": pow(g, rng.randrange(p - 1), p)}

    def solve(self, problem):
        value = 1
        for exponent in range(problem["p"] - 1):
            if value == problem["h"]:
                return {"x": exponent}
            value = value * problem["g"] % problem["p"]
        raise ValueError("target is outside the generated group")

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        return (
            type(proposed) is dict and set(proposed) == {"x"}
            and type(proposed["x"]) is int
            and 0 <= proposed["x"] < problem["p"] - 1
            and pow(problem["g"], proposed["x"], problem["p"]) == problem["h"]
        )


TASK = DiscreteLogTask()
