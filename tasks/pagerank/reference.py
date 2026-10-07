"""Reference implementation for pagerank; copied into fresh run candidates."""

import math


def _dense_reference(problem):
    """Straight dense power iteration; the input matrix is intentionally sparse."""
    matrix = problem["weights"]
    n = len(matrix)
    damping, tolerance = problem["damping"], problem["tolerance"]
    out_weight = [sum(row) for row in matrix]
    scores = [1.0 / n] * n
    for _ in range(500):
        dangling = math.fsum(scores[u] for u, total in enumerate(out_weight) if total == 0)
        base = (1.0 - damping) / n + damping * dangling / n
        next_scores = [base] * n
        for source, row in enumerate(matrix):
            total = out_weight[source]
            if total:
                scale = damping * scores[source] / total
                for target in range(n):
                    weight = row[target]
                    if weight:
                        next_scores[target] += scale * weight
        change = math.fsum(abs(a - b) for a, b in zip(next_scores, scores))
        scores = next_scores
        if change <= tolerance * (1.0 - damping) * 0.5:
            break
    return {"scores": scores}


def solve(problem):
    return _dense_reference(problem)
