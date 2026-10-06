"""Entropy-regularized transport at a declared accuracy.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 calls
ot.sinkhorn and requires one numerical plan. This adaptation keeps the entropic
objective and accepts every strictly positive plan whose marginals and
Gibbs residual meet the problem's accuracy. Stabilized log-domain Sinkhorn is
the reference. See README.md.
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
    return ("weak_reg", "strong_reg", "uneven")[seed % 3]


def _need():
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "sinkhorn requires optional dependency: numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy


def _logsumexp(values):
    numpy = _need()
    peak = numpy.max(values, axis=1, keepdims=True)
    return (peak + numpy.log(numpy.sum(numpy.exp(values - peak), axis=1, keepdims=True))).ravel()


def _sinkhorn(source, target, cost, reg, accuracy):
    numpy = _need()
    source = numpy.asarray(source, dtype=float)
    target = numpy.asarray(target, dtype=float)
    matrix = numpy.asarray(cost, dtype=float)
    log_source = numpy.log(source)
    log_target = numpy.log(target)
    kernel = -matrix / reg
    potential_f = numpy.zeros(source.shape)
    potential_g = numpy.zeros(target.shape)
    plan = None
    for _ in range(20000):
        potential_f = log_source - _logsumexp(kernel + potential_g)
        potential_g = log_target - _logsumexp((kernel + potential_f[:, None]).T)
        plan = numpy.exp(potential_f[:, None] + kernel + potential_g)
        row_error = numpy.abs(plan.sum(axis=1) - source).sum()
        col_error = numpy.abs(plan.sum(axis=0) - target).sum()
        if row_error + col_error <= accuracy * 0.2:
            break
    return plan


def _gibbs_residual(plan, cost, reg):
    numpy = _need()
    plan = numpy.asarray(plan, dtype=float)
    matrix = numpy.asarray(cost, dtype=float)
    if numpy.any(plan <= 0):
        return float("inf")
    latent = numpy.log(plan) + matrix / reg
    row_mean = latent.mean(axis=1, keepdims=True)
    reconstruction = row_mean + (latent - row_mean).mean(axis=0, keepdims=True)
    return float(numpy.max(numpy.abs(latent - reconstruction)))


class SinkhornTask:
    name = "sinkhorn"
    task_version = "1.2.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 128
    grading_cases = (128, 384)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=128, random_seed=0):
        if n < 2:
            raise ValueError("n must be at least 2")
        numpy = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        if family == "uneven":
            rows, cols = n, max(2, n // 2)
            source = rng.random(rows) ** 4
            target = rng.random(cols) ** 4
            reg = 0.05
            scale = 3.0
        elif family == "weak_reg":
            rows = cols = n
            source = numpy.full(rows, 1.0 / rows)
            target = numpy.full(cols, 1.0 / cols)
            reg = 0.02
            scale = 1.0
        else:
            rows = cols = n
            source = rng.random(rows)
            target = rng.random(cols)
            reg = 1.0
            scale = 8.0
        source = source / source.sum()
        target = target / target.sum()
        left = rng.random((rows, 2))
        right = rng.random((cols, 2))
        delta = left[:, None, :] - right[None, :, :]
        costs = scale * numpy.sqrt((delta * delta).sum(axis=2))
        return {
            "source_weights": [float(value) for value in source],
            "target_weights": [float(value) for value in target],
            "cost_matrix": [[float(value) for value in row] for row in costs],
            "reg": float(reg),
            "accuracy": 1e-5,
        }

    def solve(self, problem):
        plan = _sinkhorn(
            problem["source_weights"], problem["target_weights"], problem["cost_matrix"],
            problem["reg"], problem["accuracy"],
        )
        return {"transport_plan": plan.tolist()}

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()
        if type(proposed) is not dict or set(proposed) != {"transport_plan"}:
            return False
        plan = proposed["transport_plan"]
        source = problem["source_weights"]
        target = problem["target_weights"]
        if type(plan) is not list or len(plan) != len(source):
            return False
        parsed = []
        for row in plan:
            if type(row) is not list or len(row) != len(target):
                return False
            values = []
            for value in row:
                if type(value) not in (int, float) or isinstance(value, bool) or not math_finite(value) or value < 0:
                    return False
                values.append(float(value))
            parsed.append(values)
        array = numpy.asarray(parsed, dtype=float)
        accuracy = float(problem["accuracy"])
        row_error = numpy.abs(array.sum(axis=1) - numpy.asarray(source)).sum()
        col_error = numpy.abs(array.sum(axis=0) - numpy.asarray(target)).sum()
        if row_error + col_error > accuracy:
            return False
        # A feasible coupling that is not the entropic plan fails this residual
        # even when its marginals are exact.
        return _gibbs_residual(array, problem["cost_matrix"], problem["reg"]) <= 1e-4


def math_finite(value):
    return value == value and abs(value) != float("inf")


_need()
TASK = SinkhornTask()
