"""Reference implementation for multi_dim_knapsack; copied into fresh run candidates."""

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


def solve(problem):
    return _branch_and_bound(problem)
