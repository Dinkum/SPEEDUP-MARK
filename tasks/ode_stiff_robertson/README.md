# Robertson Kinetics

Integrate the Robertson chemical system and return the three concentrations at `t1`.

The vector field is `y' = (-k1 y1 + k3 y2 y3, k1 y1 - k2 y2² - k3 y2 y3, k2 y2²)`. Input is `t0`, `t1`, `y0`, and `k`. Return three Python floats. They must be finite and nonnegative, conserve the initial mass within `1e-5`, and match an independent BDF solution within relative `1e-5` and absolute `1e-7`. Mass conservation alone does not pass: the equilibrium `(0, 0, 1)` fails a short transient.

Seeds rotate through `transient`, `equilibrium`, and `rescaled` rate constants, with horizons varied around `n`, `40 n`, and `5 n`. Every family varies reaction-rate ratios and initial mixtures while preserving stiffness and total mass.

The reference is SciPy Radau at relative `1e-8` and absolute `1e-10`. The grader's BDF solve is a different method with the same tolerances. Upstream already used that pair; the workload families are the intentional addition.

Optimize `candidate.py:solve(problem, reference_solve)`. NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

Provenance: AlgoTune `AlgoTuneTasks/ode_stiff_robertson` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.

## Workload distribution

- **Size:** n scales integration horizon (positive).
- **Selection:** seed % 3 selects transient, equilibrium, or rescaled.
- **Randomized:** Every reaction rate independently scales by 0.85–1.15; transient also has a shared multiplier 0.8–1.2. Horizon scales by 0.85–1.15 around n, 40*n, or 5*n. Up to 0.03 initial mass moves between species one and three, preserving nonnegative concentrations and total mass.
- **Fixed structure:** Family base reaction-rate ratios retain stiffness. Initial species-two concentration is zero except 1e-5 for equilibrium. Rates, horizons, and initial mixtures vary in every family.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
