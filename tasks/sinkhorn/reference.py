"""Reference implementation for sinkhorn; copied into fresh run candidates."""

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


def solve(problem):
    plan = _sinkhorn(
        problem["source_weights"], problem["target_weights"], problem["cost_matrix"],
        problem["reg"], problem["accuracy"],
    )
    return {"transport_plan": plan.tolist()}
