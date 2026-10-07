"""Reference implementation for spectral_radius_matrix_completion; copied into fresh run candidates."""

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
            "spectral_radius_matrix_completion requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
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


def solve(problem):
    matrix, radius, _left, _right = _complete(problem["inds"], problem["a"], problem["n"])
    return {"B": matrix.tolist(), "optimal_value": float(radius)}
