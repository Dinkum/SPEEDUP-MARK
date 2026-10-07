"""Reference implementation for exact_k_nearest_neighbors; copied into fresh run candidates."""

def solve(problem):
    points, k = problem["points"], problem["k"]
    result = []
    for query in problem["queries"]:
        distances = [(sum((a - b) ** 2 for a, b in zip(point, query)), i)
                     for i, point in enumerate(points)]
        result.append([i for _, i in sorted(distances)[:k]])
    return {"indices": result}
