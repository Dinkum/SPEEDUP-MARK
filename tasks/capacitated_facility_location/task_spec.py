"""Single-source capacitated facility location with an exact subset DP.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 solves the
same binary-assignment model with CVXPY and HiGHS. This adaptation keeps that
model, uses integer data so optimality is exact, and replaces the MIP solver
with the standard facility-indexed subset DP. Workload families are an
intentional addition. See README.md.
"""

import random

from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)
_INF = 10**18


def _family(seed):
    return ("tight", "uneven", "competing")[seed % 3]


def _subset_sums(costs):
    size = 1 << len(costs)
    sums = [0] * size
    for mask in range(1, size):
        bit = mask & -mask
        sums[mask] = sums[mask ^ bit] + costs[bit.bit_length() - 1]
    return sums


def _solve_dp(fixed, capacities, demands, transport):
    """Exact optimum and one optimal 0/1 assignment matrix."""
    facilities = len(fixed)
    customers = len(demands)
    full = (1 << customers) - 1
    demand_of = _subset_sums(demands)
    leg = [_subset_sums(row) for row in transport]
    dp = [_INF] * (full + 1)
    dp[0] = 0
    choices = []
    for facility in range(facilities):
        nxt = [_INF] * (full + 1)
        choice = [None] * (full + 1)
        cap = capacities[facility]
        opening = fixed[facility]
        leg_cost = leg[facility]
        for prev, base in enumerate(dp):
            if base >= _INF:
                continue
            if base < nxt[prev]:
                nxt[prev] = base
                choice[prev] = (prev, 0)
            remaining = full ^ prev
            sub = remaining
            while sub:
                if demand_of[sub] <= cap:
                    value = base + opening + leg_cost[sub]
                    covered = prev | sub
                    if value < nxt[covered]:
                        nxt[covered] = value
                        choice[covered] = (prev, sub)
                sub = (sub - 1) & remaining
        dp = nxt
        choices.append(choice)
    if dp[full] >= _INF:
        raise ValueError("capacitated facility instance is infeasible")
    assignment = [[0] * customers for _ in range(facilities)]
    mask = full
    for facility in range(facilities - 1, -1, -1):
        prev, sub = choices[facility][mask]
        while sub:
            bit = sub & -sub
            assignment[facility][bit.bit_length() - 1] = 1
            sub ^= bit
        mask = prev
    status = [any(row) for row in assignment]
    return dp[full], status, assignment


class CapacitatedFacilityLocation:
    name = "capacitated_facility_location"
    task_version = "1.1.1"
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

    def solve(self, problem):
        cost, status, assignment = _solve_dp(
            problem["fixed_costs"], problem["capacities"], problem["demands"], problem["transportation_costs"]
        )
        return {
            "objective_value": cost,
            "facility_status": status,
            "assignments": assignment,
        }

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

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
        optimal, _, _ = _solve_dp(fixed, capacities, demands, transport)
        return cost == optimal


TASK = CapacitatedFacilityLocation()
