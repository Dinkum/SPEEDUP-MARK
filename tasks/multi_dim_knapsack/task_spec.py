"""Exact three-resource 0/1 knapsack with a bounded Pareto state space."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)
_DIMENSIONS = 3


def _fractional_bound(problem, density_orders, remaining_mask, remaining_capacity, value):
    """Bound a branch by three independent one-resource relaxations."""
    profits = problem["profits"]
    weights = problem["weights"]
    bounds = []
    for dimension, capacity in enumerate(remaining_capacity):
        bound = float(value)
        for item in density_orders[dimension]:
            if not (remaining_mask >> item) & 1:
                continue
            weight = weights[item][dimension]
            if weight <= capacity:
                capacity -= weight
                bound += profits[item]
            else:
                bound += profits[item] * capacity / weight
                break
        bounds.append(bound)
    return min(bounds)


def _branch_and_bound(problem):
    """Reference search using fractional resource bounds and memoized states."""
    profits = problem["profits"]
    weights = problem["weights"]
    capacities = problem["capacities"]
    count = len(profits)
    density_orders = tuple(
        tuple(sorted(range(count), key=lambda i, d=d: (-profits[i] / weights[i][d], i)))
        for d in range(_DIMENSIONS)
    )
    order = tuple(
        sorted(
            range(count),
            key=lambda i: (
                -profits[i] / sum(weights[i][d] / capacities[d] for d in range(_DIMENSIONS)),
                i,
            ),
        )
    )

    best_profit = 0
    greedy_orders = (order,) + density_orders
    for greedy_order in greedy_orders:
        remaining = list(capacities)
        value = 0
        for item in greedy_order:
            item_weights = weights[item]
            if all(item_weights[d] <= remaining[d] for d in range(_DIMENSIONS)):
                for d in range(_DIMENSIONS):
                    remaining[d] -= item_weights[d]
                value += profits[item]
        if value > best_profit:
            best_profit = value

    suffix_profit = [0] * (count + 1)
    for position in range(count - 1, -1, -1):
        suffix_profit[position] = suffix_profit[position + 1] + profits[order[position]]

    seen = {}

    def visit(position, remaining_mask, remaining, value):
        nonlocal best_profit
        if value + suffix_profit[position] <= best_profit:
            return
        if _fractional_bound(problem, density_orders, remaining_mask, remaining, value) <= best_profit:
            return
        state = (position, remaining[0], remaining[1], remaining[2])
        prior = seen.get(state)
        if prior is not None and prior >= value:
            return
        seen[state] = value
        if position == count:
            best_profit = value
            return

        item = order[position]
        item_weights = weights[item]
        next_mask = remaining_mask & ~(1 << item)
        if all(item_weights[d] <= remaining[d] for d in range(_DIMENSIONS)):
            next_remaining = tuple(remaining[d] - item_weights[d] for d in range(_DIMENSIONS))
            visit(position + 1, next_mask, next_remaining, value + profits[item])
        visit(position + 1, next_mask, remaining, value)

    visit(0, (1 << count) - 1, capacities, 0)
    return {"profit": best_profit}


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
    task_version = "1.1.0"
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

    def solve(self, problem):
        return _branch_and_bound(problem)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"profit"}:
            return False
        reported = proposed["profit"]
        return type(reported) is int and reported == _dynamic_programming_optimum(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
