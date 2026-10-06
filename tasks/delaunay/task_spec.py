"""2D Delaunay triangulation checked with exact integer predicates.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 returns
SciPy Qhull's simplices and hull. The reference is that same Qhull call.
Coordinates are integers, so a triangle is Delaunay exactly when no other
point has a positive incircle predicate; cocircular points may use either
diagonal. Clustered, nearly collinear, and nearly cocircular families are
intentional. See README.md.
"""

import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _family(seed):
    return ("clustered", "nearly_collinear", "nearly_cocircular")[seed % 3]


def _need():
    try:
        import numpy
        import scipy.spatial
    except ImportError as exc:
        raise ImportError(
            "delaunay requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.spatial


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _incircle(a, b, c, d):
    """Positive when d is strictly inside the circumcircle of ccw triangle abc."""
    adx, ady = a[0] - d[0], a[1] - d[1]
    bdx, bdy = b[0] - d[0], b[1] - d[1]
    cdx, cdy = c[0] - d[0], c[1] - d[1]
    ad2 = adx * adx + ady * ady
    bd2 = bdx * bdx + bdy * bdy
    cd2 = cdx * cdx + cdy * cdy
    return (
        adx * (bdy * cd2 - bd2 * cdy)
        - ady * (bdx * cd2 - bd2 * cdx)
        + ad2 * (bdx * cdy - bdy * cdx)
    )


def _ccw(points, triangle):
    a, b, c = (points[i] for i in triangle)
    if _orient(a, b, c) < 0:
        return (triangle[0], triangle[2], triangle[1])
    return triangle


def _boundary(triangles):
    counts = {}
    for tri in triangles:
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
            counts[key] = counts.get(key, 0) + 1
    if any(count not in (1, 2) for count in counts.values()):
        return None
    return counts


def _hull(points):
    """Independent convex boundary, retaining collinear boundary vertices."""
    ordered = sorted(range(len(points)), key=lambda index: points[index])

    def chain(vertices):
        result = []
        for vertex in vertices:
            while len(result) > 1 and _orient(points[result[-2]], points[result[-1]], points[vertex]) < 0:
                result.pop()
            result.append(vertex)
        return result

    return chain(ordered)[:-1] + chain(reversed(ordered))[:-1]


def _crosses(points, first, second):
    """Proper intersection of edges without a shared vertex."""
    if set(first) & set(second):
        return False
    a, b = (points[index] for index in first)
    c, d = (points[index] for index in second)
    return (_orient(a, b, c) * _orient(a, b, d) < 0
            and _orient(c, d, a) * _orient(c, d, b) < 0)


class DelaunayTask:
    name = "delaunay"
    task_version = "1.2.2"
    display_name = TASK_CATALOG[name].display_name
    default_n = 128
    grading_cases = (128, 256)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=48, random_seed=0):
        if n < 3:
            raise ValueError("n must be at least 3")
        rng = random.Random(random_seed)
        family = _family(random_seed)
        if family == "clustered":
            centers = [(rng.randint(-200, 200), rng.randint(-200, 200)) for _ in range(3)]
            # Even coincident centers must provide enough distinct lattice points.
            # A fixed radius of four has at most 243 points across three clusters.
            radius = max(4, math.isqrt(n) // 2 + 1)
            points = []
            while len(points) < n:
                center = centers[len(points) % 3]
                point = (center[0] + rng.randint(-radius, radius), center[1] + rng.randint(-radius, radius))
                if point not in points:
                    points.append(point)
        elif family == "nearly_collinear":
            # Long, thin clouds retain difficult orientation predicates while
            # changing the triangulation, not merely its coordinate frame.
            points = []
            x = 0
            for _ in range(n - 2):
                x += rng.randint(3, 10) * 32
                points.append((x, rng.randint(-2, 2)))
            points.extend(((0, rng.randint(5, 9) * 32),
                           (x + 32, -rng.randint(5, 9) * 32)))
        else:
            radius = rng.randint(4000 + 10 * n, 8000 + 10 * n)
            points = []
            for index in range(n):
                angle = 2 * math.pi * (index + rng.uniform(0.1, 0.9)) / n
                distance = radius + rng.randint(-3, 3)
                points.append((round(distance * math.cos(angle)),
                               round(distance * math.sin(angle))))
        # A tiny clustered draw can put every vertex on one line. Preserve the
        # two-dimensional contract with one nearby point off that line.
        if all(_orient(points[0], points[1], point) == 0 for point in points[2:]):
            origin = points[0]
            replacement = (origin[0] + 1, origin[1])
            if _orient(points[0], points[1], replacement) == 0:
                replacement = (origin[0], origin[1] + 1)
            points[-1] = replacement
        rng.shuffle(points)
        return {"points": [list(point) for point in points]}

    def solve(self, problem):
        numpy, spatial = _need()
        # Qhull matches the exact integer predicates on every generated family,
        # including collinear boundary points and rounded cocircular sets.
        triangulation = spatial.Delaunay(numpy.asarray(problem["points"], dtype=float))
        return {
            "simplices": [[int(vertex) for vertex in triangle] for triangle in triangulation.simplices],
            "convex_hull": [[int(vertex) for vertex in edge] for edge in triangulation.convex_hull],
        }

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"simplices", "convex_hull"}:
            return False
        points = problem["points"]
        if any(type(point) is not list or len(point) != 2 or any(type(coord) is not int for coord in point) for point in points):
            return False
        if len(set(map(tuple, points))) != len(points):
            return False
        simplices = proposed["simplices"]
        hull = proposed["convex_hull"]
        if type(simplices) is not list or type(hull) is not list:
            return False
        count = len(points)
        triangles = []
        unique = set()
        for tri in simplices:
            if type(tri) is not list or len(tri) != 3 or any(type(vertex) is not int or not 0 <= vertex < count for vertex in tri):
                return False
            if len(set(tri)) != 3:
                return False
            ordered = _ccw(points, tuple(tri))
            if _orient(*(points[vertex] for vertex in ordered)) <= 0:
                return False
            key = tuple(sorted(tri))
            if key in unique:
                return False
            unique.add(key)
            triangles.append(ordered)
        counts = _boundary(triangles)
        if counts is None:
            return False
        boundary = {edge for edge, used in counts.items() if used == 1}
        reported = set()
        for edge in hull:
            if type(edge) is not list or len(edge) != 2 or any(type(vertex) is not int or not 0 <= vertex < count for vertex in edge):
                return False
            key = tuple(sorted(edge))
            if edge[0] == edge[1] or key in reported:
                return False
            reported.add(key)
        if reported != boundary:
            return False
        convex = _hull(points)
        if len(convex) != len(set(convex)):
            return False  # All points collinear: no two-dimensional mesh exists.
        expected_boundary = {tuple(sorted((a, b))) for a, b in
                             zip(convex, convex[1:] + convex[:1])}
        if boundary != expected_boundary or len(triangles) != 2 * count - 2 - len(convex):
            return False
        # A planar oriented mesh must use each interior edge in opposite
        # directions. This also rejects overlapping faces with a shared edge.
        directed = set()
        for tri in triangles:
            for edge in zip(tri, tri[1:] + tri[:1]):
                if edge in directed:
                    return False
                directed.add(edge)
        if any((a, b) not in directed or (b, a) not in directed
               for (a, b), used in counts.items() if used == 2):
            return False
        edges = list(counts)
        if any(_crosses(points, edge, other)
               for index, edge in enumerate(edges) for other in edges[index + 1:]):
            return False
        used = {vertex for tri in triangles for vertex in tri}
        if used != set(range(count)):
            return False
        for tri in triangles:
            a, b, c = (points[vertex] for vertex in tri)
            for index, point in enumerate(points):
                if index in tri:
                    continue
                if _incircle(a, b, c, point) > 0:
                    return False
        return True


_need()
TASK = DelaunayTask()
