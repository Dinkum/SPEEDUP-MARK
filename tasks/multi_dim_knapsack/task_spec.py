"""Exact three-resource 0/1 knapsack with a bounded Pareto state space."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_DIMENSIONS = _reference._DIMENSIONS


def _dynamic_programming_optimum(problem):
    """Independent exact oracle: enumerate attainable resource vectors."""
    states = {(0, 0, 0): 0}
    capacities = problem["capacities"]
    for item, profit in enumerate(problem["profits"]):
        item_weights = problem["weights"][item]
        additions = []
        for used, value in tuple(states.items()):
            candidate = tuple(used[d] + item_weights[d] for d in range(_DIMENSIONS))
            if all(candidate[d] <= capacities[d] for d in range(_DIMENSIONS)):
                additions.append((candidate, value + profit))
        for used, value in additions:
            if value > states.get(used, -1):
                states[used] = value
    return max(states.values(), default=0)


class Task:
    name = "multi_dim_knapsack"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 24
    grading_cases = (24, 30)

    def generate_problem(self, n=24, random_seed=0):
        if type(n) is not int or not 1 <= n <= 40:
            raise ValueError("n must be an integer from 1 through 40")
        rng = random.Random(random_seed)
        weights = []
        profits = []
        for item in range(n):
            favored = item % _DIMENSIONS
            resource = [rng.randint(1, 6) for _ in range(_DIMENSIONS)]
            resource[favored] += rng.randint(1, 5)
            coefficients = [rng.randint(2, 7) for _ in range(_DIMENSIONS)]
            # Rotate the premium across resource classes so one ratio cannot
            # order every item well in all three capacity dimensions.
            coefficients[favored] += 3
            profit = 8 + sum(resource[d] * coefficients[d] for d in range(_DIMENSIONS))
            profit += rng.randint(0, 12)
            weights.append(tuple(resource))
            profits.append(profit)

        ratios = (0.25, 0.27, 0.26)
        capacities = tuple(
            max(1, int(sum(row[d] for row in weights) * ratios[(d + random_seed) % 3]))
            for d in range(_DIMENSIONS)
        )
        return {
            "profits": tuple(profits),
            "weights": tuple(weights),
            "capacities": capacities,
        }

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"profit"}:
            return False
        reported = proposed["profit"]
        return type(reported) is int and reported == _dynamic_programming_optimum(problem)


TASK = Task()
