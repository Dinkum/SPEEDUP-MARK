"""Huber-robust state estimation with exact linear dynamics.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 solves the
same convex problem in CVXPY. Process noise stays quadratic, the measurement
penalty is the Huber function of the measurement-noise norm, and the dynamics
are equalities. The reference optimizes the reduced process-noise variable
with an analytic adjoint. Isolated outliers, bursts, and longer horizons are
intentional. See README.md.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_numeric


_reference = load_reference(__file__)
_gradient = _reference._gradient
_need = _reference._need


def _family(seed):
    return ("isolated", "burst", "long")[seed % 3]


class RobustKalmanFilter:
    name = "robust_kalman_filter"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 16
    grading_cases = (16, 30)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=16, random_seed=0):
        if n < 4:
            raise ValueError("n must be at least 4")
        numpy, _optimize = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        horizon = n
        state_dim = 3 if family == "long" else 2
        process_dim = 2 if family == "long" else 1
        measure_dim = 2 if family == "burst" else 1
        raw = rng.standard_normal((state_dim, state_dim))
        spectral = numpy.max(numpy.abs(numpy.linalg.eigvals(raw)))
        transition = raw / (spectral + 0.2)
        process = 0.4 * rng.standard_normal((state_dim, process_dim))
        measure = rng.standard_normal((measure_dim, state_dim))
        initial = rng.standard_normal(state_dim)
        disturbance = 0.05 * rng.standard_normal((horizon, process_dim))
        measurement_noise = 0.05 * rng.standard_normal((horizon, measure_dim))
        if family == "burst":
            start = horizon // 3
            measurement_noise[start:start + max(2, horizon // 8)] += 4.0
        else:
            count = 1 if family == "isolated" else max(1, horizon // 10)
            picks = rng.choice(horizon, size=count, replace=False)
            measurement_noise[picks] += 5.0 * rng.standard_normal((count, measure_dim))
        state = numpy.zeros((horizon + 1, state_dim))
        state[0] = initial
        observations = numpy.zeros((horizon, measure_dim))
        for time in range(horizon):
            observations[time] = measure @ state[time] + measurement_noise[time]
            state[time + 1] = transition @ state[time] + process @ disturbance[time]
        return {
            "A": transition.tolist(),
            "B": process.tolist(),
            "C": measure.tolist(),
            "y": observations.tolist(),
            "x_initial": initial.tolist(),
            "tau": 1.5,
            "M": 0.8,
        }

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        numpy, _optimize = _need()
        if type(proposed) is not dict or set(proposed) != {"x_hat", "w_hat", "v_hat"}:
            return False
        if not all(plain_numeric(proposed[field]) for field in ('x_hat', 'w_hat', 'v_hat')):
            return False
        transition = numpy.asarray(problem["A"], dtype=float)
        process = numpy.asarray(problem["B"], dtype=float)
        measure = numpy.asarray(problem["C"], dtype=float)
        observations = numpy.asarray(problem["y"], dtype=float)
        initial = numpy.asarray(problem["x_initial"], dtype=float)
        horizon, measure_dim = observations.shape
        state_dim, process_dim = process.shape
        try:
            state = numpy.asarray(proposed["x_hat"], dtype=float)
            noise = numpy.asarray(proposed["w_hat"], dtype=float)
            residual = numpy.asarray(proposed["v_hat"], dtype=float)
        except (TypeError, ValueError):
            return False
        if state.shape != (horizon + 1, state_dim) or noise.shape != (horizon, process_dim) or residual.shape != (horizon, measure_dim):
            return False
        if not (numpy.isfinite(state).all() and numpy.isfinite(noise).all() and numpy.isfinite(residual).all()):
            return False
        if numpy.linalg.norm(state[0] - initial) > 1e-6:
            return False
        for time in range(horizon):
            if numpy.linalg.norm(state[time + 1] - (transition @ state[time] + process @ noise[time])) > 1e-6:
                return False
            if numpy.linalg.norm(observations[time] - (measure @ state[time] + residual[time])) > 1e-6:
                return False
        _state, _residual, gradient = _gradient(problem, noise)
        return float(numpy.linalg.norm(gradient)) <= 1e-4 * max(1.0, float(numpy.linalg.norm(noise)))


_need()
TASK = RobustKalmanFilter()
