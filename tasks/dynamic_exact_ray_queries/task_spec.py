"""Exact nearest-triangle ray queries through scene updates.

Semantics are integer-exact on purpose. Vertex coordinates reach ~2**32 and ray
components ~2**39, so the products and cross products inside an intersection
test exceed 2**60 — far past the 53-bit mantissa of float64. A submission that
ranks hits in floating point therefore disagrees with the exact rational
comparison, and the rays aimed at shared mesh vertices make that disagreement
reachable on ordinary work rather than only on pathological input. Every
intermediate is an exact Python integer.
"""

from __future__ import annotations

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)

# Bound on |coordinate| for generated scenes, and on ray origins/directions.
# Wide enough that difference products are inexact in float64 (a single
# cross-product term already exceeds 2**60), narrow enough to stay readable.
COORD_LIMIT = 1 << 34
RAY_LIMIT = 1 << 42
# Mesh spacing per family. Deliberately odd: power-of-two coordinates would be
# exactly representable in float64 no matter how large they get, so the
# differences and products inside an intersection test would stay exact by
# accident. Odd spacing makes the 60-bit products genuinely inexact in float64.
_COARSE = (1 << 29) + 11
_MEDIUM = (1 << 28) + 3
_FINE = (1 << 16) + 1


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


def _materialized(value):
    """Plain containers only: reject subclasses that defer work until verification.

    A sequence subclass can compute its elements lazily, so a submission could
    return almost instantly and let the untimed verifier do the work. Requiring
    exact ``tuple``/``list`` types and exact ``int`` leaves closes that hole.
    """
    if type(value) is int:
        return True
    if type(value) in (tuple, list):
        return all(_materialized(item) for item in value)
    return False


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _mesh(rng, cells_x, cells_y, spacing, origin, tilt):
    """Grid of quads split into unit triangles, so neighbours share edges.

    Shared edges and vertices are what make exact ties reachable: a ray aimed
    at a grid vertex lands on two to six triangles at the same exact distance.
    """
    triangles = []
    base_x, base_y = origin
    for column in range(cells_x):
        for row in range(cells_y):
            x0 = base_x + column * spacing
            y0 = base_y + row * spacing
            x1 = x0 + spacing
            y1 = y0 + spacing
            height = tilt[column % len(tilt)] if tilt else 0
            a = (x0, y0, height)
            b = (x1, y0, height)
            c = (x0, y1, height)
            d = (x1, y1, height)
            triangles.append((a, b, c))
            triangles.append((b, d, c))
    return triangles


def _ray(origin_xy, height, target):
    origin = (origin_xy[0], origin_xy[1], -height)
    return (origin, _sub((target[0], target[1], 0), origin))


def _vertex_targets(cells_x, cells_y, spacing, origin, rng):
    """Grid vertices, the exact-tie positions, in a deterministic shuffled order."""
    targets = [
        (origin[0] + column * spacing, origin[1] + row * spacing)
        for column in range(cells_x + 1)
        for row in range(cells_y + 1)
    ]
    rng.shuffle(targets)
    return targets


def _coherent_rays(rng, cells_x, cells_y, spacing, origin, count, tilt_targets):
    rays = []
    origin_xy = (origin[0] - spacing * (cells_x + 1), origin[1] + spacing // 2)
    height = spacing * (cells_x + 2)
    targets = _vertex_targets(cells_x, cells_y, spacing, origin, rng)
    for index in range(count):
        if index % 4 == 3 and tilt_targets:
            target = tilt_targets.pop()
        else:
            base = targets[index % len(targets)]
            target = (base[0] + rng.randrange(-2, 3), base[1] + rng.randrange(-2, 3))
        rays.append(_ray(origin_xy, height + index % 3, target))
    return rays


def _incoherent_rays(rng, cells_x, cells_y, spacing, origin, count):
    rays = []
    span = max(cells_x, cells_y) * spacing + spacing
    for index in range(count):
        origin_xy = (
            origin[0] + rng.randrange(-2 * span, 2 * span),
            origin[1] + rng.randrange(-2 * span, 2 * span),
        )
        height = rng.randrange(spacing // 2 + 1, 4 * span)
        if index % 3 == 0:
            # Deliberate misses: the ray meets the plane well outside the mesh.
            target = (
                origin[0] - span * 3 - rng.randrange(1, span),
                origin[1] + span * (index + 1),
            )
        else:
            target = (
                origin[0] + rng.randrange(-span, span),
                origin[1] + rng.randrange(-span, span),
            )
        rays.append(_ray(origin_xy, height, target))
    return rays


def _jitter(rng, triangle, delta):
    return tuple(
        (vertex[0] + rng.randrange(-delta, delta + 1),
         vertex[1] + rng.randrange(-delta, delta + 1),
         vertex[2]) for vertex in triangle
    )


def _check_scenes(scenes):
    """Fail loudly if generation ever leaves the documented coordinate contract."""
    limit = RAY_LIMIT
    for scene in scenes:
        for triangle in scene["triangles"]:
            for vertex in triangle:
                if any(type(value) is not int or abs(value) > COORD_LIMIT for value in vertex):
                    raise ValueError("generated vertex outside the coordinate bound")
        for phase in scene["phases"]:
            for ray in phase.get("rays", ()):
                origin, direction = ray
                if direction == (0, 0, 0):
                    raise ValueError("generated ray with a zero direction")
                for value in origin + direction:
                    if type(value) is not int or abs(value) > limit:
                        raise ValueError("generated ray outside the coordinate bound")


def _verify_rays(problem):
    """Plane intersections plus oriented edge tests; no barycentric solver."""
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    def cross(a, b):
        return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])
    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))
    def hit(ray, triangle):
        origin, direction = ray
        a, b, c = triangle
        normal = cross(sub(b, a), sub(c, a))
        denominator = dot(normal, direction)
        if denominator == 0:
            return None
        numerator = dot(normal, sub(a, origin))
        if denominator < 0:
            numerator, denominator = -numerator, -denominator
        if numerator <= 0:
            return None
        point = tuple(o * denominator + d * numerator for o, d in zip(origin, direction))
        for start, end in ((a, b), (b, c), (c, a)):
            offset = tuple(p - v * denominator for p, v in zip(point, start))
            if dot(normal, cross(sub(end, start), offset)) < 0:
                return None
        return numerator, denominator
    answers = []
    for scene in problem["scenes"]:
        triangles = dict(enumerate(scene["triangles"]))
        next_id, phases = len(triangles), []
        for phase in scene["phases"]:
            if "updates" in phase:
                for update in phase["updates"]:
                    if update[0] == "remove":
                        triangles.pop(update[1], None)
                    elif update[0] == "move":
                        triangles[update[1]] = update[2]
                    elif update[0] == "add":
                        triangles[next_id] = update[1]
                        next_id += 1
                    else:
                        raise ValueError(update[0])
                continue
            results = []
            for ray in phase["rays"]:
                best, best_id = None, -1
                for identifier, triangle in triangles.items():
                    if triangle is None:
                        continue
                    distance = hit(ray, triangle)
                    if distance is None:
                        continue
                    if (best is None or distance[0] * best[1] < best[0] * distance[1]
                            or distance[0] * best[1] == best[0] * distance[1] and identifier < best_id):
                        best, best_id = distance, identifier
                results.append(best_id)
            phases.append(tuple(results))
        answers.append(tuple(phases))
    return tuple(answers)

class DynamicExactRayQueriesTask:
    name = "dynamic_exact_ray_queries"
    task_version = "1.1.1"
    display_name = TASK_CATALOG[name].display_name
    default_n = 192
    grading_cases = (192, 288)

    def generate_problem(self, n=192, random_seed=0):
        if n < 8:
            raise ValueError("n must be at least 8")
        rng = random.Random(random_seed)
        scenes = (
            self._static_coherent(n, rng),
            self._moving_bursts(n, rng),
            self._mixed_sizes(n, rng),
            self._incoherent(n, rng),
        )
        _check_scenes(scenes)
        return {"scenes": scenes}

    def _grid(self, cells):
        """Cell counts per axis, chosen so the count matches ``cells`` closely."""
        side = max(1, int(cells ** 0.5))
        while side > 1 and side * side > cells:
            side -= 1
        return side, max(1, cells // side)

    def _static_coherent(self, n, rng):
        cells_x, cells_y = self._grid(max(2, n // 2))
        spacing = _MEDIUM
        origin = (spacing, spacing)
        triangles = _mesh(rng, cells_x, cells_y, spacing, origin, None)
        ties = [(origin[0] + c * spacing, origin[1] + r * spacing)
                for c in (cells_x // 2, cells_x // 2 + 1) for r in (2, cells_y - 2)]
        first = _coherent_rays(rng, cells_x, cells_y, spacing, origin, max(4, n // 4), list(ties))
        second = _coherent_rays(rng, cells_x, cells_y, spacing, origin, max(4, n // 8), list(ties))
        return {
            "family": "static_coherent",
            "triangles": tuple(triangles),
            "phases": ({"rays": tuple(first)}, {"rays": tuple(second)}),
        }

    def _moving_bursts(self, n, rng):
        cells_x, cells_y = self._grid(max(2, n // 4))
        spacing = _MEDIUM
        origin = (spacing, spacing)
        triangles = _mesh(rng, cells_x, cells_y, spacing, origin, None)
        delta = max(1, spacing // 8)
        phases = []
        for burst in range(4):
            rays = _incoherent_rays(rng, cells_x, cells_y, spacing, origin, max(3, n // 8))
            # Moving every third triangle keeps movement broad but the scene
            # recognizable, so a stale hierarchy stays plausible and wrong.
            updates = [("move", index, _jitter(rng, triangles[index], delta))
                       for index in range(0, len(triangles), 3)]
            if burst == 1:
                updates.append(("remove", len(triangles) - 1))
            if burst == 2:
                updates.append(("add", ((origin[0], origin[1], 0),
                                        (origin[0] + spacing, origin[1], spacing),
                                        (origin[0], origin[1] + spacing, 0))))
            phases.append({"rays": tuple(rays)})
            phases.append({"updates": tuple(updates)})
        return {
            "family": "moving_bursts",
            "triangles": tuple(triangles),
            "phases": tuple(phases),
        }

    def _mixed_sizes(self, n, rng):
        fine_x, fine_y = self._grid(max(2, int(n * 0.4)))
        coarse_x, coarse_y = self._grid(max(2, int(n * 0.15)))
        fine_origin = (0, 0)
        coarse_origin = (fine_x * _FINE + _COARSE, -_COARSE)
        fine = _mesh(rng, fine_x, fine_y, _FINE, fine_origin, None)
        coarse = _mesh(rng, coarse_x, coarse_y, _COARSE, coarse_origin, (0, _COARSE // 16))
        triangles = fine + coarse
        ties = [(fine_origin[0] + _FINE * (fine_x // 2), fine_origin[1] + _FINE * (fine_y // 2)),
                (coarse_origin[0] + _COARSE * (coarse_x // 2),
                 coarse_origin[1] + _COARSE * (coarse_y // 2))]
        target = max(2, n // 10)
        phases = (
            {"rays": tuple(_coherent_rays(rng, fine_x, fine_y, _FINE, fine_origin,
                                          target, list(ties)))},
            {"updates": (("move", 0, _jitter(rng, triangles[0], _FINE // 4)),
                         ("remove", 1),
                         ("add", ((fine_origin[0] + (fine_x + 2) * _FINE, fine_origin[1], 0),
                                  (fine_origin[0] + (fine_x + 6) * _FINE, fine_origin[1], _FINE),
                                  (fine_origin[0] + (fine_x + 2) * _FINE,
                                   fine_origin[1] + 4 * _FINE, 0))))},
            {"rays": tuple(_incoherent_rays(rng, coarse_x, coarse_y, _COARSE, coarse_origin,
                                            target))},
            {"updates": (("move", len(triangles) - 1,
                          _jitter(rng, triangles[-1], _COARSE // 16)),)},
            {"rays": tuple(_coherent_rays(rng, fine_x, fine_y, _FINE, fine_origin,
                                          target, list(ties)))},
        )
        return {
            "family": "mixed_sizes",
            "triangles": tuple(triangles),
            "phases": phases,
        }

    def _incoherent(self, n, rng):
        cells_x, cells_y = self._grid(max(2, n // 2))
        spacing = _MEDIUM
        origin = (-spacing, spacing)
        triangles = _mesh(rng, cells_x, cells_y, spacing, origin, None)
        phases = (
            {"rays": tuple(_incoherent_rays(rng, cells_x, cells_y, spacing, origin,
                                            max(8, n // 3)))},
            {"rays": tuple(_incoherent_rays(rng, cells_x, cells_y, spacing, origin,
                                            max(8, n // 6)))},
        )
        return {
            "family": "incoherent",
            "triangles": tuple(triangles),
            "phases": phases,
        }

    def solve(self, problem):
        return _query(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        try:
            if not _materialized(proposed):
                return False
            return _same_materialized(proposed, _verify_rays(problem))
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = DynamicExactRayQueriesTask()
