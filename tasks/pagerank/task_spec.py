"""Sparse PageRank with a contraction-based, solver-independent verifier."""

import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


def _step(problem, scores):
    matrix = problem["weights"]
    damping = problem["damping"]
    n = len(matrix)
    out_weight = [sum(row) for row in matrix]
    dangling = math.fsum(scores[u] for u, total in enumerate(out_weight) if total == 0)
    base = (1.0 - damping) / n + damping * dangling / n
    result = [base] * n
    for source, row in enumerate(matrix):
        total = out_weight[source]
        if total:
            scale = damping * scores[source] / total
            for target, weight in enumerate(row):
                if weight:
                    result[target] += scale * weight
    return result


class Task:
    name = "pagerank"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 256
    grading_cases = (256, 384)

    def generate_problem(self, n=256, random_seed=0):
        if type(n) is not int or not 4 <= n <= 600:
            raise ValueError("n must be an integer from 4 through 600")
        rng = random.Random(random_seed)
        weights = [[0] * n for _ in range(n)]
        block = max(2, n // 8)
        for source in range(n):
            if source and source % max(7, n // 12) == 0:
                continue  # Dangling nodes occur in every workload size.
            community = source // block
            targets = {
                (source + 1) % n,
                (source + 2) % n,
                ((community + 1) * block + source % block) % n,
                0,
            }
            while len(targets) < min(7, n - 1):
                targets.add(rng.randrange(n))
            targets.discard(source)
            for target in sorted(targets):
                weights[source][target] += rng.randint(1, 9)
        return {
            "weights": tuple(tuple(row) for row in weights),
            "damping": 0.85,
            "tolerance": 1e-9,
        }

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"scores"}:
            return False
        scores = proposed["scores"]
        n = len(problem["weights"])
        if type(scores) is not list or len(scores) != n:
            return False
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in scores):
            return False
        if abs(math.fsum(scores) - 1.0) > max(problem["tolerance"] * 0.1, 1e-12):
            return False
        residual = math.fsum(abs(a - b) for a, b in zip(_step(problem, scores), scores))
        # PageRank's update is a damping contraction in L1, so this residual
        # certifies distance from its unique fixed point without calling solve.
        return residual <= problem["tolerance"] * (1.0 - problem["damping"]) * (1.0 + 1e-6)


TASK = Task()
