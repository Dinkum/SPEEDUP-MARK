"""Smallest eigenvalues of a sparse symmetric positive-definite matrix.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 passes a
live SciPy matrix and calls eigsh(which='SM'). This adaptation stores CSR
arrays, asks for the algebraically smallest eigenvalues, and checks them
against an independent dense eigensolver. Banded, clustered, and
ill-conditioned families are intentional. See README.md.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _family(seed):
    return ("banded", "clustered", "ill_conditioned")[seed % 3]


def _need():
    try:
        import numpy
        import scipy.sparse
        import scipy.sparse.linalg
    except ImportError as exc:
        raise ImportError(
            "smallest_eigenvalues_sparse_spd requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.sparse, scipy.sparse.linalg


def _matrix(problem):
    numpy, sparse, _linalg = _need()
    return sparse.csr_matrix(
        (problem["data"], problem["indices"], problem["indptr"]),
        shape=tuple(problem["shape"]),
        dtype=float,
    )


def _dense_smallest(problem):
    numpy, _sparse, _linalg = _need()
    values = numpy.linalg.eigvalsh(_matrix(problem).toarray())
    return [float(value) for value in values[: problem["k"]]]


class SparseLowestEigenvalues:
    name = "smallest_eigenvalues_sparse_spd"
    task_version = "1.2.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 512
    grading_cases = (512, 1536)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=512, random_seed=0):
        if n < 6:
            raise ValueError("n must be at least 6")
        numpy, sparse, _linalg = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        if family == "banded":
            diagonals = [
                rng.uniform(0.05, 0.3, n - 2),
                rng.uniform(0.2, 0.8, n - 1),
                rng.uniform(3.0, 6.0, n),
                rng.uniform(0.2, 0.8, n - 1),
                rng.uniform(0.05, 0.3, n - 2),
            ]
            matrix = sparse.diags(diagonals, offsets=(-2, -1, 0, 1, 2), shape=(n, n), format="csr")
            matrix = (matrix + matrix.T) * 0.5 + sparse.eye(n, format="csr")
        elif family == "clustered":
            spectrum = numpy.concatenate([
                numpy.full(5, 0.4),
                numpy.linspace(4.0, 8.0, n - 5),
            ])
            signs = rng.choice((-1.0, 1.0), size=n)
            basis = sparse.diags(signs, format="csr")
            # A similar orthogonal conjugation would be dense. A diagonally
            # dominant perturbation keeps five clustered small eigenvalues.
            noise = sparse.random(n, n, density=2 / n, format="csr", random_state=rng, data_rvs=rng.standard_normal)
            noise = (noise + noise.T) * 0.02
            matrix = sparse.diags(spectrum, format="csr") + noise + 0.05 * sparse.eye(n, format="csr")
        else:
            spectrum = numpy.logspace(-3, 2, n)
            noise = sparse.random(n, n, density=3 / n, format="csr", random_state=rng, data_rvs=rng.standard_normal)
            noise = (noise + noise.T) * 0.01
            matrix = sparse.diags(spectrum, format="csr") + noise
            # A Gershgorin lower bound guarantees positivity without generating
            # a dense eigenspectrum before the sparse benchmark even starts.
            diagonal = matrix.diagonal()
            off_diagonal = numpy.asarray(abs(matrix).sum(axis=1)).ravel() - abs(diagonal)
            lower_bound = float(numpy.min(diagonal - off_diagonal))
            matrix = matrix + (0.05 - min(0.0, lower_bound)) * sparse.eye(n)
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        return {
            "data": [float(value) for value in matrix.data],
            "indices": [int(value) for value in matrix.indices],
            "indptr": [int(value) for value in matrix.indptr],
            "shape": [n, n],
            "k": 5,
        }

    def solve(self, problem):
        numpy, _sparse, linalg = _need()
        matrix = _matrix(problem)
        count = problem["k"]
        if matrix.shape[0] < 2 * count + 2:
            return _dense_smallest(problem)
        # Fix solver randomness for replay. Repeated/clustered eigenvalues can
        # still defeat ARPACK; a bounded dense solve keeps the starter valid.
        start = numpy.random.default_rng(0).standard_normal(matrix.shape[0])
        try:
            values = linalg.eigsh(matrix, k=count, which="SA", return_eigenvectors=False,
                                  tol=1e-10, v0=start)
        except linalg.ArpackNoConvergence:
            return _dense_smallest(problem)
        return [float(value) for value in numpy.sort(numpy.real(values))]

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        count = problem["k"]
        if type(proposed) is not list or len(proposed) != count:
            return False
        if any(type(value) not in (int, float) or isinstance(value, bool) for value in proposed):
            return False
        try:
            values = sorted(float(value) for value in proposed)
        except (TypeError, ValueError):
            return False
        if any(value != value or abs(value) == float("inf") for value in values):
            return False
        reference = _dense_smallest(problem)
        scale = max(1.0, max(abs(value) for value in reference))
        return all(abs(got - want) <= 1e-6 * scale for got, want in zip(values, reference))


_need()
TASK = SparseLowestEigenvalues()
