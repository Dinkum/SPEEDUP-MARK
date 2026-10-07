"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


class Task:
    name = "articulation_points"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    uses_reference_output = True
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

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed, *, reference_output=None):
        if type(proposed) is not dict or set(proposed) != {"articulation_points"}:
            return False
        points = proposed["articulation_points"]
        if (type(points) is not list
                or any(type(v) is not int or not 0 <= v < problem["num_nodes"] for v in points)
                or points != sorted(set(points))):
            return False
        # Direct checks can compute the existing reference; grading reuses its
        # measured output and never reruns a solver for verification.
        if reference_output is None:
            reference_output = self.solve(problem)
        return proposed == reference_output


TASK = Task()
