"""Reference implementation for capacitated_facility_location; copied into fresh run candidates."""

_INF = 10**18


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


def solve(problem):
    cost, status, assignment = _solve_dp(
        problem["fixed_costs"], problem["capacities"], problem["demands"], problem["transportation_costs"]
    )
    return {
        "objective_value": cost,
        "facility_status": status,
        "assignments": assignment,
    }
