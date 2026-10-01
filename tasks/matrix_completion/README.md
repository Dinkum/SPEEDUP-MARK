# Spectral-Radius Matrix Completion

Fill the missing entries of an elementwise-positive matrix to minimize its Perron root. Observed entries stay fixed. The product of the missing entries is 1.

Input is `inds`, `a`, and `n`. Return `B` and `optimal_value`. `B` must be finite and positive, match the observations, and have missing-entry product 1. `optimal_value` must be a finite Python int or float equal to the spectral radius within `1e-4 * max(1, radius)`. Observations allow absolute error `1e-6 * max(1, abs(observed_value))`; the absolute sum of logarithms of free entries must be at most `1e-5`. On the free entries, `B_ij u_i v_j` must have relative spread `(max - min) / mean <= 1e-3`, where `u` and `v` are the left and right Perron vectors. That identity is the first-order condition of the convex log-domain problem. A feasible completion that fails it is rejected.

Numerical arrays may be completed built-in lists/tuples of Python int/float values or exact real numeric NumPy arrays. Booleans, complex data, subclasses, and deferred array-conversion objects are rejected.

Seeds rotate through `sparse_obs`, `half_obs`, and `scaled` observation magnitudes. At least two entries are free.

The reference is a damped fixed-point iteration on that identity. Upstream solves the geometric program with CVXPY. The fixed-point reference is a dependency-free-of-CVXPY adaptation; NumPy supplies the eigensolver.

Optimize `candidate.py:solve(problem, reference_solve)`. NumPy is the optional numerical extra. One BLAS thread is set before it loads.

Provenance: AlgoTune `AlgoTuneTasks/matrix_completion` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.

## Workload distribution

- **Size:** n is square matrix dimension (minimum 2).
- **Selection:** seed % 3 selects sparse_obs, half_obs, or scaled.
- **Randomized:** Each entry is observed independently with probability 0.25, 0.5, or 0.4. Observed values are uniform [0.5,1.5), or [0.2,0.8) for scaled.
- **Fixed structure:** Trailing observations are removed if needed to retain at least two free entries; this conditioning is part of the distribution.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
