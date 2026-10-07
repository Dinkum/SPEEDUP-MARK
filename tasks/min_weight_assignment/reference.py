"""Reference implementation for min_weight_assignment; copied into fresh run candidates."""

def hungarian(cost):
    n = len(cost)
    u, v, match, previous = [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for row in range(1, n + 1):
        match[0] = row
        col, minimum, used = 0, [float("inf")] * (n + 1), [False] * (n + 1)
        while True:
            used[col] = True
            active, delta, nxt = match[col], float("inf"), 0
            for j in range(1, n + 1):
                if not used[j]:
                    reduced = cost[active - 1][j - 1] - u[active] - v[j]
                    if reduced < minimum[j]:
                        minimum[j], previous[j] = reduced, col
                    if minimum[j] < delta:
                        delta, nxt = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            col = nxt
            if match[col] == 0:
                break
        while col:
            prior = previous[col]
            match[col] = match[prior]
            col = prior
    assignment = [0] * n
    for col in range(1, n + 1):
        assignment[match[col] - 1] = col - 1
    return assignment


def solve(problem):
    return {"assignment": hungarian(problem["costs"])}
