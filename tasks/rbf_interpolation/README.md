# RBF Interpolation

## Input and submission

Fit SciPy's radial-basis interpolant and evaluate it at the query points.

Input is `x_train`, `y_train`, `x_test`, and `rbf_config` with `kernel`, `epsilon`, and `smoothing`. The kernel, polynomial degree, and smoothing follow `scipy.interpolate.RBFInterpolator` defaults: Gaussian gets a constant polynomial, thin-plate spline gets degree 1, and multiquadric gets degree 0. Distances passed to the kernel are multiplied by `epsilon`. Return `y_pred`, one Python float per query. Predictions must match an independent assembly of that saddle-point system within relative `1e-5` and absolute `1e-6`.

Seeds rotate through clustered Gaussian samples, a 3-D thin-plate spline, and a multiquadric problem with three times as many queries as samples. `n` is the training-set size.

NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

## Reference and verification

The reference calls `RBFInterpolator`. The grader rebuilds the kernel matrix, the shifted polynomial basis, and the diagonal smoothing and solves the system with NumPy.

## Workload distribution

- **Size:** n is training-point count; query count/dimension depend on family.
- **Selection:** seed % 3 selects gaussian_cluster, thin_plate, or multiquadric_query.
- **Randomized:** Training/query coordinates and values computed as sin(sum(coordinates)) + 0.1*x[0]. Gaussian family uses three clusters.
- **Fixed structure:** Families use dimensions 2/3/2, query counts n//2/n/3*n, smoothing 1e-6/1e-4/1e-3, and Gaussian/multiquadric epsilon 0.7/1.2.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark rbf_interpolation
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/rbf_interpolation` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. The generator is narrower than upstream's many random modes. No upstream source was copied.
