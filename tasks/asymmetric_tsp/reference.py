"""Reference implementation for asymmetric_tsp; copied into fresh run candidates."""

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


def solve(problem):
    return _branch_and_bound(problem) if len(problem["distances"]) <= 16 else _certified_tour(problem)
