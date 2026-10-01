# Smallest Sparse Eigenvalues

Return the `k` smallest eigenvalues of a sparse symmetric positive-definite matrix, `k = 5`. Any output order is accepted.

The matrix is CSR: `data`, `indices`, `indptr`, and `shape`. Values are Python floats and indexes are Python ints. `n` is the matrix order. Any order of the returned eigenvalues is accepted after sorting. Each value must be an int or float, not a bool, and must match an independent dense `eigvalsh` to `1e-6` times the scale of the spectrum.

Seeds rotate through `banded`, `clustered`, and `ill_conditioned` spectra.

The reference is `scipy.sparse.linalg.eigsh(..., which="SA")`, with a deterministic starting vector and a dense fallback when the matrix is too small for Lanczos or ARPACK does not converge. Upstream used `which="SM"` on a live sparse object. Algebraically smallest eigenvalues and plain CSR lists are intentional; for these positive-definite matrices the two selectors agree.

Optimize `candidate.py:solve(problem, reference_solve)`. NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

Provenance: AlgoTune `AlgoTuneTasks/sparse_lowest_eigenvalues_posdef` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.

## Workload distribution

- **Size:** n is square matrix dimension; request the five lowest eigenvalues.
- **Selection:** seed % 3 selects banded, clustered, or ill_conditioned.
- **Randomized:** Banded entries or sparse perturbations of the prescribed spectrum, with a positive-definite shift.
- **Fixed structure:** Clustered has five eigenvalues near 0.4 and the rest 4–8 before perturbation; ill_conditioned spans 1e-3..1e2. Family formulas define sparsity and conditioning.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
