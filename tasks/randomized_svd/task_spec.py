"""Rank-k approximation with a fixed reconstruction-quality requirement.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 calls
sklearn's randomized SVD and accepts a loose relative residual. This adaptation
uses the Halko-Martinsson-Tropp range finder and accepts any orthonormal
factorization whose Frobenius error is at most ``quality`` times the optimal
rank-k tail. Rapid, slow, and clustered spectra are intentional. See README.md.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from speedupmark.task import load_candidate, plain_numeric


_candidate = load_candidate(__file__)


def _family(seed):
    return ("rapid", "slow", "clustered")[seed % 3]


def _need():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "randomized_svd requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "Smoke and lightweight do not include this task."
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


def _tail(matrix, rank):
    numpy = _need()
    singular = numpy.linalg.svd(matrix, compute_uv=False)
    return float(numpy.sqrt(numpy.sum(singular[rank:] ** 2)))


class RandomizedSVD:
    name = "randomized_svd"
    task_version = "1.1.1"
    default_n = 200
    grading_cases = (200, 320)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=80, random_seed=0):
        if n < 8:
            raise ValueError("n must be at least 8")
        numpy = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        rows = n
        cols = n + 5
        rank_full = min(rows, cols)
        left, _ = numpy.linalg.qr(rng.standard_normal((rows, rank_full)))
        right, _ = numpy.linalg.qr(rng.standard_normal((cols, rank_full)))
        if family == "rapid":
            singular = numpy.exp(-0.45 * numpy.arange(rank_full))
            iterations = 1
        elif family == "slow":
            singular = 1.0 / (1.0 + 0.15 * numpy.arange(rank_full))
            iterations = 4
        else:
            singular = numpy.concatenate([
                numpy.full(4, 3.0),
                numpy.full(4, 2.7),
                numpy.linspace(0.4, 0.05, rank_full - 8),
            ])
            iterations = 3
        matrix = (left * singular) @ right.T
        return {
            "matrix": [[float(value) for value in row] for row in matrix],
            "k": 6,
            "quality": 1.05,
            "power_iterations": iterations,
        }

    def solve(self, problem):
        numpy = _need()
        matrix = numpy.asarray(problem["matrix"], dtype=float)
        rng = numpy.random.default_rng(0)
        left, values, right = _factor(matrix, problem["k"], problem["power_iterations"], rng)
        return {
            "U": left.tolist(),
            "S": [float(value) for value in values],
            "V": right.tolist(),
        }

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()
        if type(proposed) is not dict or set(proposed) != {"U", "S", "V"}:
            return False
        if not all(plain_numeric(proposed[field]) for field in ('U', 'S', 'V')):
            return False
        matrix = numpy.asarray(problem["matrix"], dtype=float)
        rows, cols = matrix.shape
        rank = problem["k"]
        try:
            left = numpy.asarray(proposed["U"], dtype=float)
            values = numpy.asarray(proposed["S"], dtype=float)
            right = numpy.asarray(proposed["V"], dtype=float)
        except (TypeError, ValueError):
            return False
        if left.shape != (rows, rank) or right.shape != (cols, rank) or values.shape != (rank,):
            return False
        if not (numpy.isfinite(left).all() and numpy.isfinite(right).all() and numpy.isfinite(values).all()):
            return False
        if numpy.any(values < -1e-8) or numpy.any(values[:-1] + 1e-8 < values[1:]):
            return False
        if not numpy.allclose(left.T @ left, numpy.eye(rank), atol=1e-5):
            return False
        if not numpy.allclose(right.T @ right, numpy.eye(rank), atol=1e-5):
            return False
        residual = numpy.linalg.norm(matrix - left @ (values[:, None] * right.T), ord="fro")
        allowance = max(1e-8, float(problem["quality"]) * _tail(matrix, rank))
        return float(residual) <= allowance


_need()
TASK = RandomizedSVD()
