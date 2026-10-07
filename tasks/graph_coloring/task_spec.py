"""Exact minimum graph coloring.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 colors the
adjacency matrix with CP-SAT, a clique seed, and dominator reduction. This
adaptation keeps that input and output contract and proves optimality with
Bron-Kerbosch plus DSATUR branch and bound. Sparse, dense, symmetric, and
planted families are an intentional addition. See README.md.
"""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_minimum_coloring = _reference._minimum_coloring


def _family(seed):
    return ("sparse", "dense", "symmetric", "planted")[seed % 4]


class GraphColoringAssign:
    name = "graph_coloring"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 14
    grading_cases = (14, 20)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=12, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        family = _family(random_seed)
        matrix = [[0] * n for _ in range(n)]
        if family == "symmetric":
            # Blow up a varied base graph into interchangeable twin classes.
            # This preserves symmetry without fixing the chromatic number to 3.
            parts = min(n, max(3, n // 2))
            base = [[0] * parts for _ in range(parts)]
            density = rng.uniform(0.25, 0.7)
            for i in range(parts):
                for j in range(i + 1, parts):
                    base[i][j] = base[j][i] = int(rng.random() < density)
            group = list(range(parts)) + [rng.randrange(parts) for _ in range(n - parts)]
            rng.shuffle(group)
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i][j] = matrix[j][i] = base[group[i]][group[j]]
        elif family == "planted":
            colors = 3 if n >= 6 else 2
            group = [vertex % colors for vertex in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if group[i] != group[j] and rng.random() < 0.75:
                        matrix[i][j] = matrix[j][i] = 1
        else:
            probability = 0.18 if family == "sparse" else 0.45
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < probability:
                        matrix[i][j] = matrix[j][i] = 1
        return matrix

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        n = len(problem)
        if type(proposed) is not list or len(proposed) != n:
            return False
        if any(type(color) is not int or color < 1 for color in proposed):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] and proposed[i] == proposed[j]:
                    return False
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j]:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        optimal = max(_minimum_coloring(adj)) if n else 0
        return len(set(proposed)) == optimal


TASK = GraphColoringAssign()
