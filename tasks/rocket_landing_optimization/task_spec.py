"""Minimum-fuel rocket landing under discrete thrust constraints.

Upstream AlgoTune revision dff9914c10800c7a031c9e8c3d4d1c8cd1b38906 solves this
second-order-cone program with CVXPY. The dynamics, thrust cone, altitude, and
endpoint constraints are unchanged. Any feasible trajectory within a tight
fuel gap is accepted. Horizons and near-limit initial states vary on purpose.
See README.md.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from speedupmark.task import load_candidate, plain_numeric


_candidate = load_candidate(__file__)


def _family(seed):
    return ("short", "offset", "near_limit")[seed % 3]


def _need():
    try:
        import cvxpy
        import numpy
    except ImportError as exc:
        raise ImportError(
            "rocket_landing_optimization requires optional dependencies: cvxpy, numpy. "
            "Install the pinned numerical extra in requirements-numerical.txt. "
            "Smoke and lightweight do not include this task."
        ) from exc
    return cvxpy, numpy


def _fuel(thrust, gamma):
    numpy = _need()[1]
    thrust = numpy.asarray(thrust, dtype=float)
    return float(gamma * numpy.linalg.norm(thrust, axis=1).sum())


def _optimal_fuel(problem):
    cvxpy, numpy = _need()
    steps = int(problem["K"])
    mass = float(problem["m"])
    step = float(problem["h"])
    gravity = float(problem["g"])
    limit = float(problem["F_max"])
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
        cvxpy.norm(thrust, axis=1) <= limit,
    ]
    fuel = float(problem["gamma"]) * cvxpy.sum(cvxpy.norm(thrust, axis=1))
    result = cvxpy.Problem(cvxpy.Minimize(fuel), constraints).solve(solver=cvxpy.CLARABEL, verbose=False)
    if position.value is None or result is None:
        raise ValueError("rocket landing reference is infeasible")
    return float(result)


class RocketLanding:
    name = "rocket_landing_optimization"
    task_version = "1.2.1"
    default_n = 30
    grading_cases = (30, 60)

    def workload_family(self, n, random_seed=0):
        return _family(random_seed)

    def generate_problem(self, n=10, random_seed=0):
        if n < 4:
            raise ValueError("n must be at least 4")
        _cvxpy, numpy = _need()
        rng = numpy.random.default_rng(random_seed)
        family = _family(random_seed)
        horizon = 0.5 * n
        altitude = float(rng.uniform(5, 8) if family == "short" else rng.uniform(8, 12))
        horizontal = rng.uniform(-1.5, 1.5, 2) if family == "short" else rng.uniform(-4, 4, 2)
        start = numpy.array([*horizontal, altitude])
        # Discrete trapezoidal dynamics admit a constant-acceleration descent
        # ending at rest. Construct one to bound the required thrust exactly.
        acceleration = 2 * start / horizon**2
        # A zero-sum acceleration profile changes position and force direction
        # while preserving the total velocity change of the sampled descent.
        profile = rng.uniform(-0.15, 0.15, (n, 3)) * acceleration
        profile -= profile.mean(axis=0)
        accelerations = acceleration + profile
        velocities = numpy.zeros((n + 1, 3))
        positions = numpy.zeros((n + 1, 3))
        for step in range(n - 1, -1, -1):
            velocities[step] = velocities[step + 1] - 0.5 * accelerations[step]
            positions[step] = positions[step + 1] - 0.25 * (velocities[step] + velocities[step + 1])
        forces = accelerations + numpy.array([0.0, 0.0, 1.0])
        margin = rng.uniform(1.02, 1.12) if family == "near_limit" else rng.uniform(1.4, 2.0)
        return {
            "p0": positions[0].tolist(), "v0": velocities[0].tolist(),
            "p_target": [0.0, 0.0, 0.0], "g": 1.0, "m": 1.0, "h": 0.5,
            "K": int(n), "F_max": float(numpy.linalg.norm(forces, axis=1).max() * margin),
            "gamma": 1.0,
        }

    def solve(self, problem):
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

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        numpy = _need()[1]
        required = {"position", "velocity", "thrust", "fuel_consumption"}
        if type(proposed) is not dict or set(proposed) != required:
            return False
        if not all(plain_numeric(proposed[field]) for field in ('position', 'velocity', 'thrust')):
            return False
        steps = int(problem["K"])
        try:
            position = numpy.asarray(proposed["position"], dtype=float)
            velocity = numpy.asarray(proposed["velocity"], dtype=float)
            thrust = numpy.asarray(proposed["thrust"], dtype=float)
        except (TypeError, ValueError):
            return False
        if position.shape != (steps + 1, 3) or velocity.shape != (steps + 1, 3) or thrust.shape != (steps, 3):
            return False
        if not (numpy.isfinite(position).all() and numpy.isfinite(velocity).all() and numpy.isfinite(thrust).all()):
            return False
        if type(proposed["fuel_consumption"]) not in (int, float) or isinstance(proposed["fuel_consumption"], bool):
            return False
        if not numpy.isfinite(proposed["fuel_consumption"]):
            return False
        mass, step, gravity = float(problem["m"]), float(problem["h"]), float(problem["g"])
        if numpy.linalg.norm(position[0] - problem["p0"]) > 1e-5:
            return False
        if numpy.linalg.norm(velocity[0] - problem["v0"]) > 1e-5:
            return False
        if numpy.linalg.norm(position[-1] - problem["p_target"]) > 1e-5 or numpy.linalg.norm(velocity[-1]) > 1e-5:
            return False
        if numpy.any(position[:, 2] < -1e-6):
            return False
        if numpy.any(numpy.linalg.norm(thrust, axis=1) > float(problem["F_max"]) + 1e-5):
            return False
        for time in range(steps):
            expected_velocity = velocity[time].copy()
            expected_velocity[:2] = velocity[time, :2] + step * thrust[time, :2] / mass
            expected_velocity[2] = velocity[time, 2] + step * (thrust[time, 2] / mass - gravity)
            if numpy.linalg.norm(velocity[time + 1] - expected_velocity) > 1e-5:
                return False
            expected_position = position[time] + (step / 2) * (velocity[time] + velocity[time + 1])
            if numpy.linalg.norm(position[time + 1] - expected_position) > 1e-5:
                return False
        fuel = _fuel(thrust, problem["gamma"])
        if abs(float(proposed["fuel_consumption"]) - fuel) > 1e-5 * max(1.0, fuel):
            return False
        optimal = _optimal_fuel(problem)
        return fuel <= optimal * (1 + 1e-4) + 1e-5


_need()
TASK = RocketLanding()
