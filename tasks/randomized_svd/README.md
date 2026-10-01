# Randomized SVD

Return a rank-`k` factorization whose reconstruction is within a fixed factor of the optimal rank-k tail.

Input is `matrix`, `k`, `quality`, and `power_iterations`. `quality` is `1.05`. Return `U` with shape `(rows, k)`, nonnegative nonincreasing `S` of length `k`, and `V` with shape `(columns, k)`. `U` and `V` must have orthonormal columns. The Frobenius error of `U diag(S) V.T` must be at most `quality` times the true tail, or `1e-8` when the tail vanishes. All entries must be finite real numbers. Orthonormality uses `numpy.allclose` with absolute tolerance `1e-5` and its default relative tolerance. Singular values allow `1e-8` roundoff in nonnegativity and ordering. Sign and subspace freedom inside that budget are accepted. A zero factorization is rejected.

Numerical arrays may be completed built-in lists/tuples of Python int/float values or exact real numeric NumPy arrays. Booleans, complex data, subclasses, and deferred array-conversion objects are rejected.

Seeds rotate through `rapid`, `slow`, and `clustered` singular spectra. The reference's power-iteration count is 1, 4, or 3 in that order. It is part of the input so the budget is visible; using fewer iterations is allowed when the quality check still passes.

The reference is a Halko-Martinsson-Tropp range finder with NumPy QR and SVD on the small matrix. Upstream calls scikit-learn's randomized SVD and uses a loose residual cap. The tail-relative quality test is intentional.

Optimize `candidate.py:solve(problem, reference_solve)`. NumPy is the optional numerical extra. One BLAS thread is set before it loads.

Provenance: AlgoTune `AlgoTuneTasks/randomized_svd` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.

## Workload distribution

- **Size:** n is row count; the matrix has n+5 columns.
- **Selection:** seed % 3 selects rapid, slow, or clustered singular spectra.
- **Randomized:** Left/right orthogonal factors obtained from QR decompositions of Gaussian matrices.
- **Fixed structure:** Spectra, target rank 6, quality multiplier 1.05, and family power-iteration counts 1/4/3 are fixed. The reference algorithm's internal RNG seed 0 controls its sketch, not benchmark input selection.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
