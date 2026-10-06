"""Exact directed tours: small general matrices and large assignment-tight cases."""

import random

from speedupmark.catalog import TASK_CATALOG
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


def _assignment_dual(distances):
    """Hungarian relaxation: exclude self-arcs, return matching and duals."""
    n = len(distances)
    forbidden = (max(max(row) for row in distances) + 1) * (n + 1)
    u, v, matched, previous = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for source in range(1, n + 1):
        matched[0] = source
        minimum, used, column = [float("inf")] * (n + 1), [False] * (n + 1), 0
        while True:
            used[column] = True
            row, delta, following = matched[column], float("inf"), 0
            for target in range(1, n + 1):
                if used[target]:
                    continue
                cost = forbidden if row == target else distances[row - 1][target - 1]
                reduced = cost - u[row] - v[target]
                if reduced < minimum[target]:
                    minimum[target], previous[target] = reduced, column
                if minimum[target] < delta:
                    delta, following = minimum[target], target
            for target in range(n + 1):
                if used[target]:
                    u[matched[target]] += delta
                    v[target] -= delta
                else:
                    minimum[target] -= delta
            column = following
            if matched[column] == 0:
                break
        while column:
            parent = previous[column]
            matched[column] = matched[parent]
            column = parent
    successors = [0] * n
    for target in range(1, n + 1):
        successors[matched[target] - 1] = target - 1
    return successors, u[1:], v[1:]


def _certified_tour(problem):
    """Cycle-cover relaxation and zero-reduced-cost cycle merging."""
    distances = problem["distances"]
    n = len(distances)
    successor, rows, columns = _assignment_dual(distances)
    tight = [set(j for j in range(n) if i != j and distances[i][j] == rows[i] + columns[j])
             for i in range(n)]
    def cycles():
        groups, visited = [], set()
        for start in range(n):
            if start in visited:
                continue
            group, current = [], start
            while current not in visited:
                visited.add(current)
                group.append(current)
                current = successor[current]
            groups.append(group)
        return groups
    while len(groups := cycles()) > 1:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for left in groups[a]:
                    for right in groups[b]:
                        if successor[right] in tight[left] and successor[left] in tight[right]:
                            successor[left], successor[right] = successor[right], successor[left]
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            # A tight Hamiltonian tour exists in every large generated input;
            # cycle splicing is a heuristic, so retain exact search as fallback.
            path, visited = [0], {0}
            def search(current):
                if len(path) == n:
                    return 0 in tight[current]
                choices = sorted(tight[current] - visited,
                                 key=lambda city: (len(tight[city] - visited), city))
                for city in choices:
                    visited.add(city)
                    path.append(city)
                    if all(tight[node] - visited or 0 in tight[node]
                           for node in range(n) if node not in visited) and search(city):
                        return True
                    path.pop()
                    visited.remove(city)
                return False
            if not search(0):
                raise ValueError("large inputs must admit an assignment-tight tour")
            tour = path + [0]
            return {"tour": tour, "cost": sum(distances[a][b] for a, b in zip(tour, tour[1:]))}
    tour, current = [0], successor[0]
    while current != 0:
        tour.append(current)
        current = successor[current]
    tour.append(0)
    return {"tour": tour, "cost": sum(distances[a][b] for a, b in zip(tour, tour[1:]))}


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
    task_version = "1.2.0"
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

    def solve(self, problem):
        return _branch_and_bound(problem) if len(problem["distances"]) <= 16 else _certified_tour(problem)

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

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
