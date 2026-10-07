# Logistic Group Lasso

## Input and submission

Solve logistic regression with an unsquared group-lasso penalty.

The upstream description writes `λ Σ w_j ||β_j||_2^2`. The upstream CVXPY model uses `λ Σ sqrt(|group|) ||β_j||_2`. The graded objective is the implemented one: unpenalized intercept, weights `sqrt(group size)`, and the Euclidean norm not squared. Logistic loss is `-y·Xβ + Σ log(1 + exp(Xβ))`, with the first column of `X` equal to one.

Input is `X`, `y` in `{0, 1}`, `gl` (group labels for the non-intercept columns), and `lba`. Return `beta0`, `beta`, and `optimal_value`. A solution is accepted when the logistic gradient is stationary for the group subdifferential and `optimal_value` matches that objective. Different coefficients are accepted when both are stationary, which correlated groups allow. A coefficient vector that merely has a small loss is rejected.

Seeds rotate through `balanced`, `correlated`, and `strong_penalty`.

NumPy is the optional numerical extra. One BLAS thread is set before it loads.

## Reference and verification

The reference is FISTA with the group soft-threshold. It stops at the same stationarity test the grader uses.

## Workload distribution

- **Size:** n is sample count and approximately feature count (minimum 8).
- **Selection:** seed % 3 selects balanced, correlated, or strong_penalty.
- **Randomized:** Gaussian features and active-group coefficients, with Bernoulli labels from logistic probabilities. Correlated features add 0.05 Gaussian noise to another group.
- **Fixed structure:** Four groups for balanced, six otherwise; penalties 0.4, 0.3, 2.5 respectively. The first two labels force both classes to occur.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark group_lasso
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/group_lasso` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. The squared-norm discrepancy is resolved in favor of the implementation. No upstream source was copied.
