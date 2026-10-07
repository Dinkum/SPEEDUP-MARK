"""Reference implementation for min_cost_max_flow; copied into fresh run candidates."""

def residual(problem, flow):
    arcs = []
    for i, (u, v, capacity, cost) in enumerate(problem["edges"]):
        if flow[i] < capacity:
            arcs.append((u, v, capacity - flow[i], cost, i, 1))
        if flow[i]:
            arcs.append((v, u, flow[i], -cost, i, -1))
    return arcs


def solve(problem):
    n, source, sink = problem["num_nodes"], problem["source"], problem["sink"]
    flow = [0] * len(problem["edges"])
    while True:
        arcs = residual(problem, flow)
        distance, previous = [float("inf")] * n, [None] * n
        distance[source] = 0
        for _ in range(n - 1):
            changed = False
            for arc in arcs:
                u, v, capacity, cost, i, sign = arc
                if distance[u] + cost < distance[v]:
                    distance[v], previous[v] = distance[u] + cost, arc
                    changed = True
            if not changed:
                break
        if previous[sink] is None:
            return {"flow": flow}
        path, at = [], sink
        while at != source:
            arc = previous[at]
            path.append(arc)
            at = arc[0]
        amount = min(arc[2] for arc in path)
        for _, _, _, _, i, sign in path:
            flow[i] += amount * sign
