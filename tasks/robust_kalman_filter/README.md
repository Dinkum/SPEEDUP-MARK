# Robust State Estimation

## Input and submission

Estimate a state trajectory with exact linear dynamics, quadratic process noise, and a Huber penalty on the measurement-noise norm.

`x[0]` is `x_initial`, `x[t+1] = A x[t] + B w[t]`, and `y[t] = C x[t] + v[t]`. The objective is `Σ ||w_t||^2 + tau * φ(||v_t||)`, where `φ(r) = r^2` for `r ≤ M` and `2 M r - M^2` otherwise. `n` is the horizon. Return `x_hat`, `w_hat`, and `v_hat` with shapes `(T+1, state_dimension)`, `(T, process_dimension)`, and `(T, measurement_dimension)`, where `T = len(y)`. Entries must be finite real numbers. Dynamics, measurements, and the initial state must hold within `1e-6`. The process-noise sequence must be stationary for the reduced objective: its analytic gradient norm must be at most `1e-4` times `max(1, ||w||)`. A least-squares fit that ignores the Huber kink fails that test when outliers are present.

Numerical arrays may be completed built-in lists/tuples of Python int/float values or exact real numeric NumPy arrays. Boolean and complex values are invalid.

Seeds rotate through `isolated` outliers, a `burst`, and a `long` three-dimensional state.

NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

## Reference and verification

The reference is L-BFGS on `w` with the adjoint gradient. Upstream uses CVXPY on the same convex model. The reduced-space solver is intentional.

## Workload distribution

- **Size:** n is observation horizon.
- **Selection:** seed % 3 selects isolated, burst, or long.
- **Randomized:** Stable transition/process/measurement matrices, initial state, Gaussian noise, and outlier magnitudes. Burst noise is concentrated in an interval; other families inject isolated impulses.
- **Fixed structure:** State dimension is 3 for long and 2 otherwise; process dimension 2 for long and 1 otherwise; measurement dimension 2 for burst and 1 otherwise. Robust-loss constants tau=1.5 and M=0.8.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark robust_kalman_filter
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/robust_kalman_filter` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
