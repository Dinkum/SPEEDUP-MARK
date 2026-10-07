"""Reference implementation for earth_movers_distance; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _need():
    try:
        import numpy
        import scipy.optimize
    except ImportError as exc:
        raise ImportError(
            "earth_movers_distance requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.optimize


def _optimal_cost(source, target, cost):
    numpy, optimize = _need()
    source = numpy.asarray(source, dtype=float)
    target = numpy.asarray(target, dtype=float)
    matrix = numpy.asarray(cost, dtype=float)
    rows, cols = matrix.shape
    # One marginal equation is implied by equal total mass.
    equalities = numpy.zeros((rows + cols - 1, rows * cols))
    for i in range(rows):
        equalities[i, i * cols:(i + 1) * cols] = 1.0
    for j in range(cols - 1):
        equalities[rows + j, j::cols] = 1.0
    result = optimize.linprog(
        matrix.ravel(),
        A_eq=equalities,
        b_eq=numpy.concatenate([source, target[:-1]]),
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise ValueError(f"transport solver failed: {result.message}")
    plan = result.x.reshape(rows, cols)
    return float(result.fun), plan


def solve(problem):
    _cost, plan = _optimal_cost(problem["source_weights"], problem["target_weights"], problem["cost_matrix"])
    numpy, _optimize = _need()
    return {"transport_plan": numpy.asarray(plan, dtype=float).tolist()}
