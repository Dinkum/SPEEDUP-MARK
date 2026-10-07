"""Reference implementation for pde_burgers1d; copied into fresh run candidates."""

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
            "pde_burgers1d requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.integrate


def burgers_rhs(_time, state, nu, dx):
    """Dirichlet-zero upwind advection and central diffusion on interior nodes."""
    numpy = _need()[0]
    padded = numpy.pad(state, 1)
    diffusion = (padded[2:] - 2 * padded[1:-1] + padded[:-2]) / dx**2
    center = padded[1:-1]
    forward = (padded[2:] - center) / dx
    backward = (center - padded[:-2]) / dx
    advection = numpy.where(center >= 0, center * backward, center * forward)
    return -advection + nu * diffusion


def _integrate(problem, method):
    _numpy, integrate = _need()
    params = problem["params"]
    solution = integrate.solve_ivp(
        lambda time, state: burgers_rhs(time, state, params["nu"], params["dx"]),
        (problem["t0"], problem["t1"]),
        problem["y0"],
        method=method,
        rtol=1e-6,
        atol=1e-8,
    )
    if not solution.success:
        raise ValueError(f"Burgers integrator failed: {solution.message}")
    return [float(value) for value in solution.y[:, -1]]


def solve(problem):
    return _integrate(problem, "BDF")
