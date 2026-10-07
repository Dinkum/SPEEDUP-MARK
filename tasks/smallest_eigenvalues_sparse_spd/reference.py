"""Reference implementation for smallest_eigenvalues_sparse_spd; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


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


def solve(problem):
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
