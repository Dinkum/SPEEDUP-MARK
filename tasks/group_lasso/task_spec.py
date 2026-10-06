"""Logistic group lasso with the unsquared group norm.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 describes
λ Σ w_j ||β_j||_2^2 but implements λ Σ sqrt(|group|) ||β_j||_2 in CVXPY.
The graded penalty is the implemented unsquared norm. Weights remain
sqrt(group size), and the intercept is unpenalized. Stationarity is the
optimality certificate. See README.md.
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
    return ("balanced", "correlated", "strong_penalty")[seed % 3]


def _need():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "group_lasso requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy


def _groups(labels):
    found = {}
    for index, label in enumerate(labels):
        found.setdefault(label, []).append(index)
    return [(indexes, len(indexes) ** 0.5) for indexes in found.values()]


def _logistic(features, labels, beta):
    numpy = _need()
    labels = numpy.asarray(labels, dtype=float)
    linear = features @ beta
    positive = linear >= 0
    softplus = numpy.where(positive, linear + numpy.log1p(numpy.exp(-linear)), numpy.log1p(numpy.exp(linear)))
    loss = float(numpy.sum(softplus) - labels @ linear)
    sigmoid = numpy.where(positive, 1.0 / (1.0 + numpy.exp(-linear)), numpy.exp(linear) / (1.0 + numpy.exp(linear)))
    return loss, features.T @ (sigmoid - labels)


def _lipschitz(features):
    numpy = _need()
    vector = numpy.ones(features.shape[1])
    vector /= numpy.linalg.norm(vector)
    value = 0.0
    for _ in range(40):
        vector = features.T @ (features @ vector)
        value = float(numpy.linalg.norm(vector))
        vector /= value
    return max(value / 4.0, 1e-8)


def _apply_prox(coefficients, labels, step, penalty):
    numpy = _need()
    updated = coefficients.copy()
    for indexes, weight in _groups(labels):
        slot = [index + 1 for index in indexes]
        vector = updated[slot]
        norm = float(numpy.linalg.norm(vector))
        limit = step * penalty * weight
        if norm <= limit:
            updated[slot] = 0.0
        else:
            updated[slot] *= 1.0 - limit / norm
    return updated


def _objective(features, labels, group_labels, penalty, beta):
    numpy = _need()
    loss, _grad = _logistic(features, labels, beta)
    extra = 0.0
    for indexes, weight in _groups(group_labels):
        extra += weight * float(numpy.linalg.norm(beta[[index + 1 for index in indexes]]))
    return loss + penalty * extra


def _stationary(features, labels, group_labels, penalty, beta):
    numpy = _need()
    _loss, gradient = _logistic(features, labels, beta)
    scale = 1.0 + float(numpy.linalg.norm(gradient))
    tolerance = 1e-4 * scale
    if abs(float(gradient[0])) > tolerance:
        return False
    slopes = gradient[1:]
    for indexes, weight in _groups(group_labels):
        block = beta[[index + 1 for index in indexes]]
        slope = slopes[indexes]
        norm = float(numpy.linalg.norm(block))
        if norm <= 1e-8:
            if float(numpy.linalg.norm(slope)) > penalty * weight + tolerance:
                return False
        elif float(numpy.linalg.norm(slope + penalty * weight * block / norm)) > tolerance:
            return False
    return True


class GroupLasso:
    name = "group_lasso"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 24
    grading_cases = (24, 40)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=24, random_seed=0):
        if n < 8:
            raise ValueError("n must be at least 8")
        numpy = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        groups = 4 if family == "balanced" else 6
        width = max(2, n // groups)
        labels = []
        for group in range(groups):
            size = width if group < groups - 1 else max(2, n - width * (groups - 1))
            labels.extend([group + 1] * size)
        features_only = rng.normal(size=(n, len(labels)))
        if family == "correlated":
            features_only[:, width: 2 * width] = features_only[:, :width] + 0.05 * rng.normal(size=(n, width))
        features = numpy.column_stack([numpy.ones(n), features_only])
        truth = numpy.zeros(len(labels))
        truth[:width] = rng.normal(size=width)
        probability = 1.0 / (1.0 + numpy.exp(-(features @ numpy.concatenate([[0.2], truth]))))
        labels_y = rng.random(n) < probability
        # Keep both classes present so the logistic term is informative.
        labels_y[0] = True
        labels_y[1] = False
        penalty = {"balanced": 0.4, "correlated": 0.3, "strong_penalty": 2.5}[family]
        return {
            "X": [[float(value) for value in row] for row in features],
            "y": [int(value) for value in labels_y],
            "gl": labels,
            "lba": float(penalty),
        }

    def solve(self, problem):
        numpy = _need()
        features = numpy.asarray(problem["X"], dtype=float)
        labels = numpy.asarray(problem["y"], dtype=float)
        penalty = float(problem["lba"])
        step = 1.0 / _lipschitz(features)
        current = numpy.zeros(features.shape[1])
        extrapolated = current.copy()
        momentum = 1.0
        for _ in range(12000):
            _loss, gradient = _logistic(features, labels, extrapolated)
            nxt = _apply_prox(extrapolated - step * gradient, problem["gl"], step, penalty)
            momentum_next = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
            extrapolated = nxt + ((momentum - 1.0) / momentum_next) * (nxt - current)
            current = nxt
            momentum = momentum_next
            if _stationary(features, labels, problem["gl"], penalty, current):
                break
        if not _stationary(features, labels, problem["gl"], penalty, current):
            raise ValueError("group lasso reference did not reach stationarity")
        value = _objective(features, labels, problem["gl"], penalty, current)
        return {
            "beta0": float(current[0]),
            "beta": [float(value) for value in current[1:]],
            "optimal_value": float(value),
        }

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()
        if type(proposed) is not dict or set(proposed) != {"beta0", "beta", "optimal_value"}:
            return False
        beta = proposed["beta"]
        width = len(problem["X"][0]) - 1
        if type(beta) is not list or len(beta) != width:
            return False
        if any(type(value) not in (int, float) or isinstance(value, bool) for value in beta):
            return False
        if type(proposed["beta0"]) not in (int, float) or isinstance(proposed["beta0"], bool):
            return False
        if type(proposed["optimal_value"]) not in (int, float) or isinstance(proposed["optimal_value"], bool):
            return False
        coefficients = numpy.concatenate([[float(proposed["beta0"])], numpy.asarray(beta, dtype=float)])
        if not numpy.isfinite(coefficients).all() or not _finite(proposed["optimal_value"]):
            return False
        features = numpy.asarray(problem["X"], dtype=float)
        if not _stationary(features, problem["y"], problem["gl"], float(problem["lba"]), coefficients):
            return False
        value = _objective(features, problem["y"], problem["gl"], float(problem["lba"]), coefficients)
        return abs(float(proposed["optimal_value"]) - value) <= 1e-4 * max(1.0, abs(value))


def _finite(value):
    return value == value and abs(value) != float("inf")


_need()
TASK = GroupLasso()
