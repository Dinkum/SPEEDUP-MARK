"""Single-source capacitated facility location with an exact subset DP.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 solves the
same binary-assignment model with CVXPY and HiGHS. This adaptation keeps that
model, uses integer data so optimality is exact, and replaces the MIP solver
with the standard facility-indexed subset DP. Workload families are an
intentional addition. See README.md.
"""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


def _family(seed):
    return ("tight", "uneven", "competing")[seed % 3]


def _no_cheaper_assignment(problem, target):
    """Independent customer-first branch search, rather than facility subsets."""
    demands, fixed, capacity = problem["demands"], problem["fixed_costs"], problem["capacities"]
    costs = problem["transportation_costs"]
    facilities = len(fixed)
    order = sorted(range(len(demands)), key=lambda j: (-demands[j], j))
    minimum = [min(costs[i][j] for i in range(facilities)) for j in order]
    suffix = [0] * (len(order) + 1)
    for j in range(len(order) - 1, -1, -1):
        suffix[j] = suffix[j + 1] + minimum[j]
    seen = {}
    def cheaper(position, loads, opened, cost):
        if cost + suffix[position] >= target:
            return False
        if position == len(order):
            return True
        state = position, loads, opened
        if seen.get(state, float("inf")) <= cost:
            return False
        seen[state] = cost
        customer = order[position]
        for facility in sorted(range(facilities), key=lambda i: costs[i][customer]):
            if loads[facility] + demands[customer] > capacity[facility]:
                continue
            next_loads = list(loads)
            next_loads[facility] += demands[customer]
            opening = 0 if opened & (1 << facility) else fixed[facility]
            if cheaper(position + 1, tuple(next_loads), opened | (1 << facility),
                       cost + costs[facility][customer] + opening):
                return True
        return False
    return not cheaper(0, (0,) * facilities, 0, 0)


class CapacitatedFacilityLocation:
    name = "capacitated_facility_location"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 8
    grading_cases = (8, 11)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=8, random_seed=0):
        if n < 3:
            raise ValueError("n must be at least 3")
        customers = n
        facilities = 4 if customers <= 9 else 5
        rng = random.Random(random_seed)
        family = _family(random_seed)
        if family == "uneven":
            demands = [30] + [rng.randint(1, 3) for _ in range(customers - 1)]
            fixed = [40] + [8, 9, 10, 11][: facilities - 1]
            capacities = [40] + [8] * (facilities - 1)
            sites = [(0, 0)] + [(rng.randint(8, 24), rng.randint(8, 24)) for _ in range(facilities - 1)]
        else:
            demands = [rng.randint(2, 8) for _ in range(customers)]
            total = sum(demands)
            share = (total + facilities - 2) // (facilities - 1)
            while max(demands) > share or (facilities - 2) * share >= total:
                demands[demands.index(max(demands))] -= 1
                total = sum(demands)
                share = (total + facilities - 2) // (facilities - 1)
            capacities = [share] * facilities
            if family == "tight":
                fixed = [rng.randint(15, 60) for _ in range(facilities)]
                sites = [(rng.randint(0, 30), rng.randint(0, 30)) for _ in range(facilities)]
            else:
                fixed = [30 + rng.randint(0, 4) for _ in range(facilities)]
                anchor = (rng.randint(0, 10), rng.randint(0, 10))
                sites = [(anchor[0] + rng.randint(0, 2), anchor[1] + rng.randint(0, 2)) for _ in range(facilities)]
        clients = [(rng.randint(0, 30), rng.randint(0, 30)) for _ in range(customers)]
        transport = [
            [abs(site[0] - client[0]) + abs(site[1] - client[1]) + 1 for client in clients]
            for site in sites
        ]
        return {
            "fixed_costs": fixed,
            "capacities": capacities,
            "demands": demands,
            "transportation_costs": transport,
        }

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {
            "objective_value", "facility_status", "assignments"
        }:
            return False
        fixed = problem["fixed_costs"]
        capacities = problem["capacities"]
        demands = problem["demands"]
        transport = problem["transportation_costs"]
        status = proposed["facility_status"]
        assignment = proposed["assignments"]
        facilities, customers = len(fixed), len(demands)
        objective = proposed["objective_value"]
        if type(objective) not in (int, float) or isinstance(objective, bool):
            return False
        if type(status) is not list or len(status) != facilities:
            return False
        if any(type(flag) is not bool for flag in status):
            return False
        if type(assignment) is not list or len(assignment) != facilities:
            return False
        load = [0] * facilities
        cost = 0
        for facility, row in enumerate(assignment):
            if type(row) is not list or len(row) != customers:
                return False
            opened = False
            for customer, bit in enumerate(row):
                if type(bit) is not int or bit not in (0, 1):
                    return False
                if bit:
                    opened = True
                    load[facility] += demands[customer]
                    cost += transport[facility][customer]
            if opened:
                cost += fixed[facility]
            if opened != status[facility]:
                return False
            if load[facility] > capacities[facility]:
                return False
        served = [sum(assignment[facility][customer] for facility in range(facilities)) for customer in range(customers)]
        if served != [1] * customers:
            return False
        if cost != objective:
            return False
        return _no_cheaper_assignment(problem, cost)


TASK = CapacitatedFacilityLocation()
