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
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_need = _reference._need
_objective = _reference._objective
_stationary = _reference._stationary


def _family(seed):
    return ("balanced", "correlated", "strong_penalty")[seed % 3]


class GroupLasso:
    name = "group_lasso"
    task_version = "2.0.0"
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

    solve = staticmethod(_reference.solve)


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
