"""Reference implementation for low_rank_approximation; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _need():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "low_rank_approximation requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy


def _factor(matrix, rank, iterations, rng):
    numpy = _need()
    _rows, cols = matrix.shape
    width = min(cols, rank + 8)
    sketch = matrix @ rng.standard_normal((cols, width))
    for _ in range(iterations):
        sketch = matrix @ (matrix.T @ sketch)
    basis, _rest = numpy.linalg.qr(sketch, mode="reduced")
    small_u, values, vt = numpy.linalg.svd(basis.T @ matrix, full_matrices=False)
    return basis @ small_u[:, :rank], values[:rank], vt[:rank].T


def solve(problem):
    numpy = _need()
    matrix = numpy.asarray(problem["matrix"], dtype=float)
    rng = numpy.random.default_rng(0)
    left, values, right = _factor(matrix, problem["k"], problem["power_iterations"], rng)
    return {
        "U": left.tolist(),
        "S": [float(value) for value in values],
        "V": right.tolist(),
    }
