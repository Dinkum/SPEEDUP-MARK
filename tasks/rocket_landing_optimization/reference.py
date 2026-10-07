"""Reference implementation for rocket_landing_optimization; copied into fresh run candidates."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _need():
    try:
        import cvxpy
        import numpy
    except ImportError as exc:
        raise ImportError(
            "rocket_landing_optimization requires optional dependencies: cvxpy, numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "The smoke and extended suites do not include this task."
        ) from exc
    return cvxpy, numpy


def solve(problem):
    cvxpy, numpy = _need()
    steps = int(problem["K"])
    mass = float(problem["m"])
    step = float(problem["h"])
    gravity = float(problem["g"])
    position = cvxpy.Variable((steps + 1, 3))
    velocity = cvxpy.Variable((steps + 1, 3))
    thrust = cvxpy.Variable((steps, 3))
    constraints = [
        velocity[0] == numpy.asarray(problem["v0"], dtype=float),
        position[0] == numpy.asarray(problem["p0"], dtype=float),
        velocity[steps] == 0,
        position[steps] == numpy.asarray(problem["p_target"], dtype=float),
        position[:, 2] >= 0,
        velocity[1:, :2] == velocity[:-1, :2] + step * thrust[:, :2] / mass,
        velocity[1:, 2] == velocity[:-1, 2] + step * (thrust[:, 2] / mass - gravity),
        position[1:] == position[:-1] + (step / 2) * (velocity[:-1] + velocity[1:]),
        cvxpy.norm(thrust, axis=1) <= float(problem["F_max"]),
    ]
    fuel = float(problem["gamma"]) * cvxpy.sum(cvxpy.norm(thrust, axis=1))
    problem_model = cvxpy.Problem(cvxpy.Minimize(fuel), constraints)
    problem_model.solve(solver=cvxpy.CLARABEL, verbose=False)
    if position.value is None:
        raise ValueError("rocket landing reference is infeasible")
    return {
        "position": position.value.tolist(),
        "velocity": velocity.value.tolist(),
        "thrust": thrust.value.tolist(),
        "fuel_consumption": float(problem_model.value),
    }
