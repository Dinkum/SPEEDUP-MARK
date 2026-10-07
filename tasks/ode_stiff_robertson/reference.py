"""Reference implementation for ode_stiff_robertson; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


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


def solve(problem):
    return _integrate(problem, "Radau")
