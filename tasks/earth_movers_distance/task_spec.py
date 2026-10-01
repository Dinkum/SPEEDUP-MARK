"""Minimum-cost transport with a HiGHS reference and a dual-free cost certificate.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 calls POT's
network simplex and accepts only that solver's basis. This adaptation keeps the
coupling contract, uses SciPy's HiGHS linear solver, and accepts every feasible
plan whose cost matches the optimum. Unequal supports, sparse mass, and tied
costs are intentional. See README.md.
"""

import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _family(seed):
    return ("unequal", "sparse", "tied")[seed % 3]


def _need():
    try:
        import numpy
        import scipy.optimize
    except ImportError as exc:
        raise ImportError(
            "earth_movers_distance requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "Smoke and lightweight do not include this task."
        ) from exc
    return numpy, scipy.optimize


def _plan_cost(cost, plan):
    total = 0.0
    for row, costs in zip(plan, cost):
        for flow, value in zip(row, costs):
            total += flow * value
    return total


def _marginals_ok(plan, source, target, tol):
    rows = [sum(row) for row in plan]
    cols = [sum(plan[i][j] for i in range(len(plan))) for j in range(len(target))]
    return (
        all(abs(got - want) <= tol for got, want in zip(rows, source))
        and all(abs(got - want) <= tol for got, want in zip(cols, target))
    )


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


class EarthMoversDistance:
    name = "earth_movers_distance"
    task_version = "1.1.0"
    default_n = 48
    grading_cases = (48, 90)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=20, random_seed=0):
        if n < 2:
            raise ValueError("n must be at least 2")
        numpy, _optimize = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        if family == "unequal":
            rows, cols = n, max(2, (2 * n) // 3)
            source = rng.random(rows)
            target = rng.random(cols)
            costs = rng.random((rows, cols)) * 10
        elif family == "sparse":
            rows = cols = n
            source = rng.random(rows)
            target = rng.random(cols)
            source[: max(1, rows // 5)] = 0
            target[: max(1, cols // 5)] = 0
            costs = rng.random((rows, cols)) * 8
        else:
            rows = cols = n
            source = numpy.full(rows, 1.0 / rows)
            target = numpy.full(cols, 1.0 / cols)
            # Many equal integer costs, so several plans share the optimum.
            costs = rng.integers(0, 4, size=(rows, cols)).astype(float)
        source = source / source.sum()
        target = target / target.sum()
        return {
            "source_weights": [float(value) for value in source],
            "target_weights": [float(value) for value in target],
            "cost_matrix": [[float(value) for value in row] for row in costs],
        }

    def solve(self, problem):
        _cost, plan = _optimal_cost(problem["source_weights"], problem["target_weights"], problem["cost_matrix"])
        numpy, _optimize = _need()
        return {"transport_plan": numpy.asarray(plan, dtype=float).tolist()}

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"transport_plan"}:
            return False
        plan = proposed["transport_plan"]
        source = problem["source_weights"]
        target = problem["target_weights"]
        cost = problem["cost_matrix"]
        if type(plan) is not list or len(plan) != len(source):
            return False
        parsed = []
        for row in plan:
            if type(row) is not list or len(row) != len(target):
                return False
            values = []
            for value in row:
                if type(value) not in (int, float) or isinstance(value, bool) or value < -1e-9 or value != value or abs(value) == float("inf"):
                    return False
                values.append(float(value))
            parsed.append(values)
        scale = max(1.0, sum(source))
        if not _marginals_ok(parsed, source, target, 1e-6 * scale):
            return False
        optimal, _plan = _optimal_cost(source, target, cost)
        return abs(_plan_cost(cost, parsed) - optimal) <= 1e-6 * max(1.0, abs(optimal))


_need()
TASK = EarthMoversDistance()
