"""Distance-only vehicle routing for exactly K nonempty depot routes.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 states a
symmetric distance matrix and solves an MTZ integer program with CP-SAT. Its
generator draws asymmetric integers. This adaptation keeps the stated contract
(no capacities), emits symmetric positive integer distances, and uses
Held-Karp subset routing plus an exact K-partition. See README.md.
"""

import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)
_INF = 10**18


def _family(seed):
    return ("clustered", "spread", "lattice")[seed % 3]


def _distance(p, q):
    return max(1, int(round(math.hypot(p[0] - q[0], p[1] - q[1]) * 10)))


def _routes_for_mask(distance, nodes, depot, mask, end_dp, end_parent):
    """One minimum depot-returning route through the customers in mask."""
    best_end = None
    best_cost = _INF
    bit = 1
    index = 0
    while bit <= mask:
        if mask & bit:
            # The subset cost returns to the depot; the cheapest open path can
            # end at a different customer than the cheapest closed route.
            value = end_dp[mask][index] + distance[nodes[index]][depot]
            if value < best_cost:
                best_cost = value
                best_end = index
        bit <<= 1
        index += 1
    route = [depot]
    current_mask = mask
    current = best_end
    steps = []
    while current is not None:
        steps.append(nodes[current])
        link = end_parent[current_mask][current]
        if link is None:
            break
        current, current_mask = link
    route.extend(reversed(steps))
    route.append(depot)
    return route


def _solve_vrp(distance, vehicles, depot):
    size = len(distance)
    nodes = [node for node in range(size) if node != depot]
    customers = len(nodes)
    if vehicles < 1 or vehicles > customers:
        raise ValueError("K must be between 1 and the number of customers")
    full = (1 << customers) - 1
    dp = [[_INF] * customers for _ in range(full + 1)]
    parent = [[None] * customers for _ in range(full + 1)]
    for index, node in enumerate(nodes):
        dp[1 << index][index] = distance[depot][node]
    for mask in range(full + 1):
        for index in range(customers):
            base = dp[mask][index]
            if not (mask & (1 << index)) or base >= _INF:
                continue
            node = nodes[index]
            for nxt in range(customers):
                if mask & (1 << nxt):
                    continue
                value = base + distance[node][nodes[nxt]]
                covered = mask | (1 << nxt)
                if value < dp[covered][nxt]:
                    dp[covered][nxt] = value
                    parent[covered][nxt] = (index, mask)
    route_cost = [_INF] * (full + 1)
    route_cost[0] = 0
    for mask in range(1, full + 1):
        best = _INF
        for index in range(customers):
            if mask & (1 << index):
                best = min(best, dp[mask][index] + distance[nodes[index]][depot])
        route_cost[mask] = best
    split = [[_INF] * (full + 1) for _ in range(vehicles + 1)]
    choice = [[0] * (full + 1) for _ in range(vehicles + 1)]
    split[0][0] = 0
    for count in range(1, vehicles + 1):
        for mask in range(full + 1):
            sub = mask
            while sub:
                previous = mask ^ sub
                if route_cost[sub] < _INF and split[count - 1][previous] < _INF:
                    value = split[count - 1][previous] + route_cost[sub]
                    if value < split[count][mask]:
                        split[count][mask] = value
                        choice[count][mask] = sub
                sub = (sub - 1) & mask
    if split[vehicles][full] >= _INF:
        raise ValueError("vehicle routing instance is infeasible")
    routes = []
    mask = full
    for count in range(vehicles, 0, -1):
        sub = choice[count][mask]
        routes.append(_routes_for_mask(distance, nodes, depot, sub, dp, parent))
        mask ^= sub
    return split[vehicles][full], routes


def _route_trace_cost(distance, vehicles, depot):
    """Independent customer/route-boundary DP, with no subset-tour tables."""
    from functools import lru_cache
    nodes = [i for i in range(len(distance)) if i != depot]
    full = (1 << len(nodes)) - 1
    @lru_cache(None)
    def finish(mask, last, routes):
        if mask == full:
            return distance[last][depot] if routes == 1 and last != depot else float("inf")
        remaining = len(nodes) - mask.bit_count()
        if last == depot and remaining < routes:
            return float("inf")
        best = float("inf")
        for index, node in enumerate(nodes):
            bit = 1 << index
            if not mask & bit:
                best = min(best, distance[last][node] + finish(mask | bit, node, routes))
        if last != depot and routes > 1 and remaining >= routes - 1:
            best = min(best, distance[last][depot] + finish(mask, depot, routes - 1))
        return best
    return finish(0, depot, vehicles)


class VehicleRouting:
    name = "vehicle_routing"
    task_version = "1.2.1"
    display_name = TASK_CATALOG[name].display_name
    default_n = 8
    grading_cases = (8, 11)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=8, random_seed=0):
        if n < 2:
            raise ValueError("n must be at least 2")
        rng = random.Random(random_seed)
        family = _family(random_seed)
        if family == "clustered":
            vehicles = 2 if n < 9 else 3
            centers = [(rng.randint(-40, 40), rng.randint(-40, 40)) for _ in range(vehicles)]
            points = [(0, 0)]
            for customer in range(n):
                center = centers[customer % vehicles]
                points.append((center[0] + rng.randint(-3, 3), center[1] + rng.randint(-3, 3)))
        elif family == "spread":
            vehicles = 2
            points = [(0, 0)] + [(rng.randint(-50, 50), rng.randint(-50, 50)) for _ in range(n)]
        else:
            vehicles = 3 if n >= 6 else 2
            width = max(3, math.isqrt(n) + 1)
            # Occupancy and unequal row/column spacing change route choices;
            # translation or a uniform scale alone would preserve the tour.
            xs = [0]
            ys = [0]
            for _ in range(width):
                xs.append(xs[-1] + rng.randint(3, 9))
                ys.append(ys[-1] + rng.randint(3, 9))
            cells = rng.sample([(x, y) for x in xs for y in ys], n + 1)
            points = cells
        vehicles = min(vehicles, n)
        size = n + 1
        distance = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                value = _distance(points[i], points[j])
                distance[i][j] = distance[j][i] = value
        return {"D": distance, "K": vehicles, "depot": 0}

    def solve(self, problem):
        _cost, routes = _solve_vrp(problem["D"], problem["K"], problem["depot"])
        return routes

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        distance = problem["D"]
        vehicles = problem["K"]
        depot = problem["depot"]
        size = len(distance)
        if type(proposed) is not list or len(proposed) != vehicles:
            return False
        seen = []
        total = 0
        for route in proposed:
            if type(route) is not list or len(route) < 3 or route[0] != depot or route[-1] != depot:
                return False
            if any(type(node) is not int or not 0 <= node < size for node in route):
                return False
            if any(node == depot for node in route[1:-1]):
                return False
            for left, right in zip(route, route[1:]):
                leg = distance[left][right]
                if type(leg) is not int or leg <= 0:
                    return False
                total += leg
            seen.extend(route[1:-1])
        if sorted(seen) != [node for node in range(size) if node != depot]:
            return False
        return total == _route_trace_cost(distance, vehicles, depot)


TASK = VehicleRouting()
