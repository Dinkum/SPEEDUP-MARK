"""Final state of the specified semi-discrete viscous Burgers equation.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 integrates
an upwind/central method of lines with RK45 and compares with itself. This
adaptation keeps that stencil and accepts a final state that matches an
independent stiff integrator. Smooth, steep, and two-wave families are
intentional. See README.md.
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
    return ("smooth", "steep", "two_wave")[seed % 3]


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


class Burgers1D:
    name = "pde_burgers1d"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 120
    grading_cases = (120, 200)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=32, random_seed=0):
        if n < 8:
            raise ValueError("n must be at least 8")
        numpy, _integrate = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        dx = 2.0 / (n + 1)
        grid = numpy.linspace(-1.0 + dx, 1.0 - dx, n)
        if family == "smooth":
            initial = 0.4 * numpy.sin(numpy.pi * grid)
            # Fine-grid diffusion is stiff enough that an implicit step is competitive.
            nu, final = 0.08, 0.25
        elif family == "steep":
            initial = numpy.exp(-((grid + 0.3) ** 2) / 0.008)
            nu, final = 0.01, 0.12
        else:
            initial = numpy.exp(-((grid + 0.45) ** 2) / 0.02) - 0.6 * numpy.exp(-((grid - 0.4) ** 2) / 0.02)
            nu, final = 0.04, 0.2
        initial = initial + 0.01 * rng.uniform(-1, 1, n)
        return {
            "t0": 0.0,
            "t1": final,
            "y0": [float(value) for value in initial],
            "params": {"nu": nu, "dx": float(dx), "num_points": n},
            "x_grid": [float(value) for value in grid],
        }

    def solve(self, problem):
        return _integrate(problem, "BDF")

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()[0]
        if type(proposed) is not list or len(proposed) != len(problem["y0"]):
            return False
        if any(type(value) not in (int, float) or isinstance(value, bool) for value in proposed):
            return False
        values = numpy.asarray(proposed, dtype=float)
        if not numpy.isfinite(values).all():
            return False
        reference = numpy.asarray(_integrate(problem, "Radau"), dtype=float)
        return numpy.allclose(values, reference, rtol=1e-4, atol=1e-5)


_need()
TASK = Burgers1D()
