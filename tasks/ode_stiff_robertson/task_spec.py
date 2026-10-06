"""Stiff Robertson chemical kinetics.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 integrates
the standard three-species system with Radau and a second BDF check. This
adaptation keeps that vector field. Mass conservation is required and is not
treated as proof of accuracy. Transient, near-equilibrium, and rescaled-rate
families are intentional. See README.md.
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
    return ("transient", "equilibrium", "rescaled")[seed % 3]


def _need():
    try:
        import numpy
        import scipy.integrate
    except ImportError as exc:
        raise ImportError(
            "ode_stiff_robertson requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.integrate


def robertson_rhs(_time, state, rates):
    y1, y2, y3 = state
    k1, k2, k3 = rates
    return (
        -k1 * y1 + k3 * y2 * y3,
        k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
        k2 * y2 * y2,
    )


def _integrate(problem, method):
    numpy, integrate = _need()
    solution = integrate.solve_ivp(
        lambda time, state: robertson_rhs(time, state, problem["k"]),
        (problem["t0"], problem["t1"]),
        problem["y0"],
        method=method,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise ValueError(f"Robertson integrator failed: {solution.message}")
    return [float(value) for value in solution.y[:, -1]]


class Robertson:
    name = "ode_stiff_robertson"
    task_version = "1.2.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 4
    grading_cases = (4, 10)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=4, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        numpy = _need()[0]
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        scale = float(rng.uniform(0.8, 1.2))
        if family == "transient":
            rates = (0.04 * scale, 3e7 * scale, 1e4 * scale)
            horizon = float(n)
            initial = [1.0, 0.0, 0.0]
        elif family == "equilibrium":
            rates = (0.04, 3e7, 1e4)
            horizon = float(40 * n)
            initial = [0.7, 1e-5, 0.3 - 1e-5]
        else:
            rates = (0.04 * 2.0, 3e7 * 0.5, 1e4 * 1.5)
            horizon = float(5 * n)
            initial = [0.4, 0.0, 0.6]
        # Change rate ratios as well as the time scale, retaining the large
        # separation of reaction rates that makes this system stiff.
        rates = [float(rate * rng.uniform(0.85, 1.15)) for rate in rates]
        horizon *= float(rng.uniform(0.85, 1.15))
        transfer = float(rng.uniform(-0.03, 0.03))
        if family == "transient":
            transfer = abs(transfer)
            initial = [1.0 - transfer, 0.0, transfer]
        else:
            initial = [initial[0] + transfer, initial[1], initial[2] - transfer]
        return {"t0": 0.0, "t1": horizon, "y0": initial, "k": rates}

    def solve(self, problem):
        return _integrate(problem, "Radau")

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()[0]
        if type(proposed) is not list or len(proposed) != 3:
            return False
        if any(type(value) not in (int, float) or isinstance(value, bool) for value in proposed):
            return False
        values = numpy.asarray(proposed, dtype=float)
        if not numpy.isfinite(values).all() or numpy.any(values < -1e-8):
            return False
        if abs(float(numpy.sum(values) - sum(problem["y0"]))) > 1e-5:
            return False
        reference = numpy.asarray(_integrate(problem, "BDF"), dtype=float)
        return bool(numpy.allclose(values, reference, rtol=1e-5, atol=1e-7))


_need()
TASK = Robertson()
