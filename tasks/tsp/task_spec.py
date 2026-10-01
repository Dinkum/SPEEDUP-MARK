"""Exact closed-tour optimization on small, asymmetric city matrices."""

import random

from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _lower_bound(distances, current, remaining_mask):
    n = len(distances)
    remaining = [city for city in range(1, n) if remaining_mask & (1 << city)]
    if not remaining:
        return distances[current][0]
    nodes = [current, *remaining]
    targets = [*remaining, 0]
    outgoing = 0
    for source in nodes:
        outgoing += min(distances[source][target] for target in targets if target != source)
    incoming = 0
    for target in targets:
        incoming += min(distances[source][target] for source in nodes if source != target)
    return max(outgoing, incoming)


def _branch_and_bound(problem):
    """Reference exact search using relaxed in- and out-degree bounds."""
    distances = problem["distances"]
    n = len(distances)
    if n == 2:
        return {"tour": [0, 1, 0], "cost": distances[0][1] + distances[1][0]}

    def greedy(first):
        route = [0]
        remaining = set(range(1, n))
        current = 0
        while remaining:
            next_city = min(remaining, key=lambda city: (distances[current][city], city))
            route.append(next_city)
            remaining.remove(next_city)
            current = next_city
        route.append(0)
        return route, sum(distances[route[i]][route[i + 1]] for i in range(n))

    best_route, best_cost = greedy(0)
    # Trying each possible first arc gives the search a useful incumbent even
    # when one cluster has an attractive nearest-neighbor trap.
    for first in range(1, n):
        route = [0, first]
        remaining = set(range(1, n)) - {first}
        current = first
        while remaining:
            nxt = min(remaining, key=lambda city: (distances[current][city], city))
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt
        route.append(0)
        cost = sum(distances[route[i]][route[i + 1]] for i in range(n))
        if cost < best_cost:
            best_route, best_cost = route, cost

    full_mask = ((1 << n) - 1) ^ 1
    path = [0]

    def visit(current, remaining_mask, cost):
        nonlocal best_route, best_cost
        if not remaining_mask:
            total = cost + distances[current][0]
            if total < best_cost:
                best_cost = total
                best_route = path + [0]
            return
        if cost + _lower_bound(distances, current, remaining_mask) >= best_cost:
            return
        choices = [city for city in range(1, n) if remaining_mask & (1 << city)]
        choices.sort(key=lambda city: (distances[current][city], city))
        for city in choices:
            next_cost = cost + distances[current][city]
            if next_cost >= best_cost:
                continue
            path.append(city)
            visit(city, remaining_mask ^ (1 << city), next_cost)
            path.pop()

    visit(0, full_mask, 0)
    return {"tour": best_route, "cost": best_cost}


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


class Task:
    name = "tsp"
    task_version = "1.1.0"
    display_name = "Asymmetric Traveling Salesperson"
    default_n = 10
    grading_cases = (10, 12)

    def generate_problem(self, n=10, random_seed=0):
        if type(n) is not int or not 2 <= n <= 16:
            raise ValueError("n must be an integer from 2 through 16")
        rng = random.Random(random_seed)
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

    def solve(self, problem):
        return _branch_and_bound(problem)

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
        return cost == _held_karp_cost(distances)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
