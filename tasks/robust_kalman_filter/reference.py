"""Reference implementation for robust_kalman_filter; copied into fresh run candidates."""

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
            "robust_kalman_filter requires optional dependencies: numpy, scipy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return numpy, scipy.optimize


def _rollout(problem, noise):
    numpy, _optimize = _need()
    transition = numpy.asarray(problem["A"], dtype=float)
    process = numpy.asarray(problem["B"], dtype=float)
    measure = numpy.asarray(problem["C"], dtype=float)
    observations = numpy.asarray(problem["y"], dtype=float)
    state = numpy.zeros((len(noise) + 1, transition.shape[0]))
    state[0] = numpy.asarray(problem["x_initial"], dtype=float)
    for time, disturbance in enumerate(noise):
        state[time + 1] = transition @ state[time] + process @ disturbance
    residual = observations - state[:-1] @ measure.T
    return state, residual


def _huber(norm, threshold):
    if norm <= threshold:
        return norm * norm
    return 2 * threshold * norm - threshold * threshold


def _huber_gradient(residual, threshold):
    numpy = _need()[0]
    norm = float(numpy.linalg.norm(residual))
    if norm == 0 or norm <= threshold:
        return 2 * residual
    return (2 * threshold / norm) * residual


def _objective(problem, noise):
    numpy = _need()[0]
    _state, residual = _rollout(problem, noise)
    value = float(numpy.sum(noise * noise))
    threshold = float(problem["M"])
    weight = float(problem["tau"])
    for row in residual:
        value += weight * _huber(float(numpy.linalg.norm(row)), threshold)
    return value


def _gradient(problem, noise):
    numpy = _need()[0]
    transition = numpy.asarray(problem["A"], dtype=float)
    process = numpy.asarray(problem["B"], dtype=float)
    measure = numpy.asarray(problem["C"], dtype=float)
    state, residual = _rollout(problem, noise)
    horizon = len(noise)
    adjoint = numpy.zeros_like(state)
    threshold = float(problem["M"])
    weight = float(problem["tau"])
    for time in range(horizon - 1, -1, -1):
        measurement = weight * _huber_gradient(residual[time], threshold)
        adjoint[time] = transition.T @ adjoint[time + 1] - measure.T @ measurement
    gradient = 2 * noise.copy()
    for time in range(horizon):
        gradient[time] += process.T @ adjoint[time + 1]
    return state, residual, gradient


def solve(problem):
    numpy, optimize = _need()
    process = numpy.asarray(problem["B"], dtype=float)
    horizon, process_dim = len(problem["y"]), process.shape[1]

    def flat_objective(vector):
        return _objective(problem, vector.reshape(horizon, process_dim))

    def flat_gradient(vector):
        _state, _residual, gradient = _gradient(problem, vector.reshape(horizon, process_dim))
        return gradient.ravel()

    result = optimize.minimize(
        flat_objective, numpy.zeros(horizon * process_dim), jac=flat_gradient, method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 400},
    )
    if not result.success:
        raise ValueError(f"robust Kalman solver failed: {result.message}")
    noise = result.x.reshape(horizon, process_dim)
    state, residual, gradient = _gradient(problem, noise)
    if float(numpy.linalg.norm(gradient)) > 1e-5 * max(1.0, float(numpy.linalg.norm(noise))):
        raise ValueError("robust Kalman reference is not stationary")
    return {
        "x_hat": state.tolist(),
        "w_hat": noise.tolist(),
        "v_hat": residual.tolist(),
    }
