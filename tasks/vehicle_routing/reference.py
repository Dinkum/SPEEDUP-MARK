"""Reference implementation for vehicle_routing; copied into fresh run candidates."""

_INF = 10**18


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


def solve(problem):
    _cost, routes = _solve_vrp(problem["D"], problem["K"], problem["depot"])
    return routes
