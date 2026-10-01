# Rocket Landing

Choose a thrust sequence that lands a unit-mass rocket at the origin with zero final velocity, nonnegative altitude, and thrust magnitude at most `F_max`, while minimizing `gamma * Σ ||F_t||`.

The discrete dynamics use step `h`. Horizontal acceleration is `F / m`. Vertical acceleration is `F_z / m - g`. Position uses the trapezoid of the neighboring velocities. `n` is the number of steps `K`. Input also contains `p0`, `v0`, `p_target`, `g`, `m`, and `gamma`. Return `position` with shape `(K+1, 3)`, `velocity` with the same shape, `thrust` with shape `(K, 3)`, and `fuel_consumption`. Arrays must be finite. Initial/final states and per-step dynamics have Euclidean error at most `1e-5`; altitude must be at least `-1e-6` and force norms at most `F_max + 1e-5`. Reported fuel must be a finite Python int or float matching the thrust objective within `1e-5 * max(1, fuel)`. The objective must be at most `optimal_fuel * (1 + 1e-4) + 1e-5`. A feasible trajectory that spends extra fuel is rejected.

Numerical arrays may be completed built-in lists/tuples of Python int/float values or exact real numeric NumPy arrays. Booleans, complex data, subclasses, and deferred array-conversion objects are rejected.

Seeds rotate through `short`, `offset`, and `near_limit` initial states. `g = 1`, `m = 1`, and `h = 0.5` in every family.

The reference is CVXPY with Clarabel. Upstream uses the same model. Any optimal trajectory is accepted; the check is the fuel gap, not one particular thrust history.

Optimize `candidate.py:solve(problem, reference_solve)`. CVXPY, Clarabel, and NumPy are the optional numerical extra. One BLAS thread is set before they load. Do not install packages.

Provenance: AlgoTune `AlgoTuneTasks/rocket_landing_optimization` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.

## Workload distribution

- **Size:** n is trajectory step count (minimum 4).
- **Selection:** seed % 3 selects short, offset, or near_limit.
- **Randomized:** Draw nominal altitude 5–8 for short, 8–12 otherwise; horizontal coordinates -1.5..1.5 for short, -4..4 otherwise. Perturb each component of a constant acceleration by up to 15%, center the perturbations, and integrate backward from rest at the origin. Initial position/velocity therefore vary and have an explicit feasible witness. The thrust bound is peak witness force times a random margin 1.02–1.12 for near_limit or 1.4–2.0 otherwise.
- **Fixed structure:** Mass and gravity are 1; step is 0.5. Positive vertical acceleration in the witness guarantees nonnegative altitude. Near_limit means close to this witness's peak force, not a certified minimum feasible thrust limit. The witness is not included in the input; the solver still minimizes fuel.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
