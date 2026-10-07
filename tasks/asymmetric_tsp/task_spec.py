"""Exact directed tours: small general matrices and large assignment-tight cases."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_assignment_dual = _reference._assignment_dual


def _held_karp_cost(distances):
    """Verifier oracle using bottom-up subset dynamic programming."""
    n = len(distances)
    if n == 2:
        return distances[0][1] + distances[1][0]
    state_count = 1 << (n - 1)
    full = state_count - 1
    table = [[float("inf")] * n for _ in range(state_count)]
    for city in range(1, n):
        table[1 << (city - 1)][city] = distances[0][city]
    for mask in range(1, state_count):
        last_bits = mask
        while last_bits:
            bit = last_bits & -last_bits
            last_bits ^= bit
            last = bit.bit_length()
            previous_mask = mask ^ bit
            if previous_mask:
                prior_bits = previous_mask
                best = float("inf")
                while prior_bits:
                    prior_bit = prior_bits & -prior_bits
                    prior_bits ^= prior_bit
                    prior = prior_bit.bit_length()
                    candidate = table[previous_mask][prior] + distances[prior][last]
                    if candidate < best:
                        best = candidate
                table[mask][last] = best
    return min(table[full][last] + distances[last][0] for last in range(1, n))


def _dual_certifies(distances, cost):
    """Dual feasibility plus objective equality proves optimality itself.

    The Hungarian routine is only a way to propose a bound. Its answer cannot
    make a wrong tour pass: every inequality is checked against the input.
    """
    _, rows, columns = _assignment_dual(distances)
    n = len(distances)
    return (sum(rows) + sum(columns) == cost
            and all(rows[i] + columns[j] <= distances[i][j]
                    for i in range(n) for j in range(n) if i != j))


class Task:
    name = "asymmetric_tsp"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 48
    grading_cases = (12, 48, 96)

    def generate_problem(self, n=48, random_seed=0):
        if type(n) is not int or not 2 <= n <= 256:
            raise ValueError("n must be an integer from 2 through 256")
        rng = random.Random(random_seed)
        if n > 16:
            # A planted tight cycle attains a feasible assignment dual. Neither
            # the cycle nor the potentials are handed to the candidate.
            rows = [rng.randrange(1, 60) for _ in range(n)]
            columns = [rng.randrange(1, 60) for _ in range(n)]
            family = random_seed % 3
            distances = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    chance = (0.18, 0.28, 0.42)[family]
                    if family == 1:
                        chance = 0.50 if i % 4 == j % 4 else 0.16
                    slack = 0 if rng.random() < chance else rng.randrange(1, 100)
                    distances[i][j] = rows[i] + columns[j] + slack
            tour = list(range(n))
            rng.shuffle(tour)
            for i, j in zip(tour, tour[1:] + tour[:1]):
                distances[i][j] = rows[i] + columns[j]
            return {"distances": tuple(tuple(row) for row in distances)}
        distances = [[0] * n for _ in range(n)]
        for source in range(n):
            for target in range(n):
                if source == target:
                    continue
                same_region = (source % 3) == (target % 3)
                low, high = (3, 42) if same_region else (24, 115)
                # Directed costs deliberately need not satisfy symmetry or the
                # triangle inequality; source and destination regions differ.
                value = rng.randint(low, high)
                if source == 0:
                    value += rng.randint(0, 18)
                if target == 0:
                    value += rng.randint(0, 18)
                distances[source][target] = value
        return {"distances": tuple(tuple(row) for row in distances)}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"tour", "cost"}:
            return False
        tour, reported = proposed["tour"], proposed["cost"]
        distances = problem["distances"]
        n = len(distances)
        if type(tour) is not list or type(reported) is not int:
            return False
        if any(type(city) is not int for city in tour):
            return False
        if len(tour) != n + 1 or tour[0] != 0 or tour[-1] != 0:
            return False
        if sorted(tour[:-1]) != list(range(n)):
            return False
        cost = sum(distances[tour[i]][tour[i + 1]] for i in range(n))
        if cost != reported:
            return False
        return cost == _held_karp_cost(distances) if n <= 16 else _dual_certifies(distances, cost)


TASK = Task()
