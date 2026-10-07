"""Radial-basis interpolation with SciPy's kernel and polynomial convention.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 evaluates
scipy.interpolate.RBFInterpolator. This adaptation keeps that definition,
including the default polynomial degree, and checks predictions against an
independent assembly of the same saddle-point system. Kernels, smoothing,
dimension, clustering, and query counts vary on purpose. See README.md.
"""

import math
import os
from itertools import combinations_with_replacement

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_need = _reference._need


_MIN_DEGREE = {"multiquadric": 0, "linear": 0, "thin_plate_spline": 1, "cubic": 1, "quintic": 2}


def _family(seed):
    return ("gaussian_cluster", "thin_plate", "multiquadric_query")[seed % 3]


def _powers(dimension, degree):
    count = math.comb(degree + dimension, dimension)
    powers = [[0] * dimension for _ in range(count)]
    cursor = 0
    for total in range(degree + 1):
        for monomial in combinations_with_replacement(range(dimension), total):
            for variable in monomial:
                powers[cursor][variable] += 1
            cursor += 1
    return powers


def _kernel(distance, name):
    if name == "linear":
        return -distance
    if name == "thin_plate_spline":
        return 0.0 if distance == 0 else distance * distance * math.log(distance)
    if name == "gaussian":
        return math.exp(-distance * distance)
    if name == "multiquadric":
        return -math.sqrt(distance * distance + 1)
    raise ValueError(name)


def _predict(train, values, queries, config):
    numpy = _need()[0]
    train = numpy.asarray(train, dtype=float)
    values = numpy.asarray(values, dtype=float)
    queries = numpy.asarray(queries, dtype=float)
    kernel = config["kernel"]
    epsilon = float(config["epsilon"])
    degree = max(_MIN_DEGREE.get(kernel, -1), 0)
    powers = numpy.asarray(_powers(train.shape[1], degree), dtype=int)
    shift = (train.max(axis=0) + train.min(axis=0)) / 2
    scale = (train.max(axis=0) - train.min(axis=0)) / 2
    scale = numpy.where(scale == 0, 1.0, scale)
    samples = len(train)
    monomials = len(powers)

    def kernel_matrix(left, right):
        delta = epsilon * (left[:, None, :] - right[None, :, :])
        distance = numpy.linalg.norm(delta, axis=2)
        return numpy.vectorize(_kernel)(distance, kernel)

    system = numpy.zeros((samples + monomials, samples + monomials))
    system[:samples, :samples] = kernel_matrix(train, train)
    system[:samples, :samples] += float(config["smoothing"]) * numpy.eye(samples)
    transformed = (train - shift) / scale
    if monomials:
        basis = numpy.prod(transformed[:, None, :] ** powers[None, :, :], axis=2)
        system[:samples, samples:] = basis
        system[samples:, :samples] = basis.T
    rhs = numpy.zeros(samples + monomials)
    rhs[:samples] = values
    coefficients = numpy.linalg.solve(system, rhs)
    query_kernel = kernel_matrix(queries, train)
    query_basis = numpy.prod(((queries - shift) / scale)[:, None, :] ** powers[None, :, :], axis=2) if monomials else numpy.zeros((len(queries), 0))
    design = numpy.column_stack([query_kernel, query_basis]) if monomials else query_kernel
    return design @ coefficients


class RBFInterpolation:
    name = "rbf_interpolation"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 250
    grading_cases = (250, 500)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=20, random_seed=0):
        if n < 6:
            raise ValueError("n must be at least 6")
        numpy, _interpolate = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        if family == "gaussian_cluster":
            dimension, queries, kernel, epsilon, smoothing = 2, n // 2, "gaussian", 0.7, 1e-6
            centers = rng.random((3, dimension))
            train = centers[rng.integers(0, 3, n)] + 0.05 * rng.normal(size=(n, dimension))
        elif family == "thin_plate":
            dimension, queries, kernel, epsilon, smoothing = 3, n, "thin_plate_spline", 1.0, 1e-4
            train = rng.random((n, dimension))
        else:
            dimension, queries, kernel, epsilon, smoothing = 2, 3 * n, "multiquadric", 1.2, 1e-3
            train = rng.random((n, dimension))
        values = numpy.sin(train.sum(axis=1)) + 0.1 * train[:, 0]
        test = rng.random((queries, dimension))
        return {
            "x_train": train.tolist(),
            "y_train": [float(value) for value in values],
            "x_test": test.tolist(),
            "rbf_config": {"kernel": kernel, "epsilon": epsilon, "smoothing": smoothing},
        }

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        numpy = _need()[0]
        if type(proposed) is not dict or set(proposed) != {"y_pred"} or type(proposed["y_pred"]) is not list:
            return False
        if len(proposed["y_pred"]) != len(problem["x_test"]):
            return False
        if any(type(value) not in (int, float) or isinstance(value, bool) for value in proposed["y_pred"]):
            return False
        values = numpy.asarray(proposed["y_pred"], dtype=float)
        if not numpy.isfinite(values).all():
            return False
        reference = _predict(problem["x_train"], problem["y_train"], problem["x_test"], problem["rbf_config"])
        return bool(numpy.allclose(values, reference, rtol=1e-5, atol=1e-6))


_need()
TASK = RBFInterpolation()
