# Robertson Chemical Kinetics

## Input and submission

Integrate the Robertson chemical system and return the three concentrations at `t1`.

The vector field is `y' = (-k1 y1 + k3 y2 y3, k1 y1 - k2 y2² - k3 y2 y3, k2 y2²)`. Input is `t0`, `t1`, `y0`, and `k`. Return three Python floats. They must be finite and nonnegative, conserve the initial mass within `1e-5`, and match an independent BDF solution within relative `1e-5` and absolute `1e-7`. Mass conservation alone does not pass: the equilibrium `(0, 0, 1)` fails a short transient.

Seeds rotate through `transient`, `equilibrium`, and `rescaled` rate constants, with horizons varied around `n`, `40 n`, and `5 n`. Every family varies reaction-rate ratios and initial mixtures while preserving stiffness and total mass.

NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

## Reference and verification

The output accuracy budget is the task objective. Candidates may choose solver tolerances, reduced states, or methods that meet it; they need not reproduce the reference integration tolerances. The tighter Radau/BDF tolerances reduce oracle error below the public acceptance budget. Speed gained by reducing unnecessary accuracy is valid; returning values outside the stated budget fails.

The reference is SciPy Radau at relative `1e-8` and absolute `1e-10`. The grader's BDF solve is a different method with the same tolerances. Upstream already used that pair; the workload families are the intentional addition.

## Workload distribution

- **Size:** n scales integration horizon (positive).
- **Selection:** seed % 3 selects transient, equilibrium, or rescaled.
- **Randomized:** Every reaction rate independently scales by 0.85–1.15; transient also has a shared multiplier 0.8–1.2. Horizon scales by 0.85–1.15 around n, 40*n, or 5*n. Up to 0.03 initial mass moves between species one and three, preserving nonnegative concentrations and total mass.
- **Fixed structure:** Family base reaction-rate ratios retain stiffness. Initial species-two concentration is zero except 1e-5 for equilibrium. Rates, horizons, and initial mixtures vary in every family.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark ode_stiff_robertson
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/ode_stiff_robertson` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
