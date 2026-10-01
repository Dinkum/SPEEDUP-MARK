"""Positive matrix completion minimizing the spectral radius.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 solves the
geometric program with CVXPY. The objective, observed entries, and unit product
of the missing entries are unchanged. The reference is damped fixed-point
iteration on the Perron derivative. At a minimizer, B_ij u_i v_j is constant
on every free entry. See README.md.
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
    return ("sparse_obs", "half_obs", "scaled")[seed % 3]


def _need():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "matrix_completion requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "Smoke and lightweight do not include this task."
        ) from exc
    return numpy


def _perron(matrix):
    numpy = _need()
    values, right = numpy.linalg.eig(matrix)
    index = int(numpy.argmax(values.real))
    radius = float(values[index].real)
    left_values, left = numpy.linalg.eig(matrix.T)
    left_index = int(numpy.argmax(left_values.real))
    right_vector = numpy.abs(right[:, index].real)
    left_vector = numpy.abs(left[:, left_index].real)
    left_vector = numpy.maximum(left_vector, 1e-30)
    right_vector = numpy.maximum(right_vector, 1e-30)
    left_vector /= numpy.dot(left_vector, right_vector)
    return radius, left_vector, right_vector


def _free_positions(indices, size):
    observed = {(int(row), int(col)) for row, col in indices}
    return [(row, col) for row in range(size) for col in range(size) if (row, col) not in observed]


def _complete(indices, values, size):
    numpy = _need()
    matrix = numpy.ones((size, size))
    for (row, col), value in zip(indices, values):
        matrix[int(row), int(col)] = float(value)
    free = _free_positions(indices, size)
    rows = numpy.array([row for row, _col in free])
    cols = numpy.array([col for _row, col in free])
    for _ in range(400):
        _radius, left, right = _perron(matrix)
        raw = 1.0 / (left[rows] * right[cols])
        target = numpy.log(raw) - numpy.mean(numpy.log(raw))
        current = numpy.log(matrix[rows, cols])
        matrix[rows, cols] = numpy.exp(0.6 * current + 0.4 * target)
        sensitivity = matrix[rows, cols] * left[rows] * right[cols]
        if float((sensitivity.max() - sensitivity.min()) / sensitivity.mean()) < 1e-6:
            break
    radius, left, right = _perron(matrix)
    return matrix, radius, left, right


class MatrixCompletion:
    name = "matrix_completion"
    task_version = "1.1.1"
    default_n = 4
    grading_cases = (4, 6)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=4, random_seed=0):
        if n < 2:
            raise ValueError("n must be at least 2")
        numpy = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        fraction = {"sparse_obs": 0.25, "half_obs": 0.5, "scaled": 0.4}[family]
        low, high = (0.2, 0.8) if family == "scaled" else (0.5, 1.5)
        observed = []
        values = []
        for row in range(n):
            for col in range(n):
                if rng.random() < fraction:
                    observed.append([row, col])
                    values.append(float(rng.uniform(low, high)))
        # The product constraint needs at least two free positive entries.
        while n * n - len(observed) < 2:
            observed.pop()
            values.pop()
        return {"inds": observed, "a": values, "n": n}

    def solve(self, problem):
        matrix, radius, _left, _right = _complete(problem["inds"], problem["a"], problem["n"])
        return {"B": matrix.tolist(), "optimal_value": float(radius)}

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()
        if type(proposed) is not dict or set(proposed) != {"B", "optimal_value"}:
            return False
        if not all(plain_numeric(proposed[field]) for field in ('B',)):
            return False
        size = problem["n"]
        try:
            matrix = numpy.asarray(proposed["B"], dtype=float)
        except (TypeError, ValueError):
            return False
        if matrix.shape != (size, size) or not numpy.isfinite(matrix).all() or numpy.any(matrix <= 0):
            return False
        if type(proposed["optimal_value"]) not in (int, float) or isinstance(proposed["optimal_value"], bool):
            return False
        if not numpy.isfinite(proposed["optimal_value"]):
            return False
        for (row, col), value in zip(problem["inds"], problem["a"]):
            if abs(matrix[int(row), int(col)] - float(value)) > 1e-6 * max(1.0, abs(float(value))):
                return False
        free = _free_positions(problem["inds"], size)
        rows = numpy.array([row for row, _col in free])
        cols = numpy.array([col for _row, col in free])
        if abs(float(numpy.sum(numpy.log(matrix[rows, cols])))) > 1e-5:
            return False
        radius, left, right = _perron(matrix)
        if abs(float(proposed["optimal_value"]) - radius) > 1e-4 * max(1.0, radius):
            return False
        sensitivity = matrix[rows, cols] * left[rows] * right[cols]
        return float((sensitivity.max() - sensitivity.min()) / sensitivity.mean()) <= 1e-3


_need()
TASK = MatrixCompletion()
