"""Reference implementation for delaunay; copied into fresh run candidates."""

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


def solve(problem):
    numpy, spatial = _need()
    # Qhull matches the exact integer predicates on every generated family,
    # including collinear boundary points and rounded cocircular sets.
    triangulation = spatial.Delaunay(numpy.asarray(problem["points"], dtype=float))
    return {
        "simplices": [[int(vertex) for vertex in triangle] for triangle in triangulation.simplices],
        "convex_hull": [[int(vertex) for vertex in edge] for edge in triangulation.convex_hull],
    }
