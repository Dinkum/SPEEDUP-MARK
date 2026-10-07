"""Reference implementation for dynamic_exact_ray_queries; copied into fresh run candidates."""

from __future__ import annotations

def _sub(left, right):
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _hit_parameter(ray, triangle):
    """Return ``(numerator, denominator)`` for the hit distance, or ``None``.

    Möller–Trumbore in integer arithmetic: the barycentric coordinates and the
    ray parameter are exact rationals with a shared denominator, so every sign
    test is a cross-multiplication and never a division. Boundary contacts are
    inclusive, which makes hits on a shared vertex or edge exactly equal for
    every triangle that touches them.
    """
    origin, direction = ray
    vertex0, vertex1, vertex2 = triangle
    edge1 = _sub(vertex1, vertex0)
    edge2 = _sub(vertex2, vertex0)
    pvec = _cross(direction, edge2)
    determinant = _dot(edge1, pvec)
    if determinant == 0:
        # Parallel to the triangle plane: no division, no hit.
        return None
    tvec = _sub(origin, vertex0)
    u_numerator = _dot(tvec, pvec)
    qvec = _cross(tvec, edge1)
    v_numerator = _dot(direction, qvec)
    t_numerator = _dot(edge2, qvec)
    if determinant > 0:
        if t_numerator <= 0 or u_numerator < 0 or v_numerator < 0:
            return None
        if u_numerator + v_numerator > determinant:
            return None
        return t_numerator, determinant
    if t_numerator >= 0 or u_numerator > 0 or v_numerator > 0:
        return None
    if u_numerator + v_numerator < determinant:
        return None
    # Flip both signs so every returned distance has a positive denominator.
    return -t_numerator, -determinant


def _nearest(triangles, ray):
    """Smallest id among the closest hits; ``-1`` when nothing is hit."""
    best_numerator = 0
    best_denominator = 1
    best_id = -1
    for identifier, triangle in enumerate(triangles):
        if triangle is None:
            continue
        hit = _hit_parameter(ray, triangle)
        if hit is None:
            continue
        numerator, denominator = hit
        if best_id < 0 or numerator * best_denominator < best_numerator * denominator:
            best_numerator, best_denominator, best_id = numerator, denominator, identifier
    return best_id


def _query(problem):
    """Brute-force reference: every live triangle, every ray, exact arithmetic."""
    answers = []
    for scene in problem["scenes"]:
        triangles = list(scene["triangles"])
        phases = []
        for phase in scene["phases"]:
            if "rays" in phase:
                phases.append(tuple(_nearest(triangles, ray) for ray in phase["rays"]))
                continue
            for update in phase["updates"]:
                kind = update[0]
                if kind == "remove":
                    triangles[update[1]] = None
                elif kind == "move":
                    triangles[update[1]] = update[2]
                elif kind == "add":
                    triangles.append(update[1])
                else:
                    raise ValueError(f"unsupported scene update {kind!r}")
        answers.append(tuple(phases))
    return tuple(answers)


def solve(problem):
    return _query(problem)
