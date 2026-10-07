# Smallest Eigenvalues of Sparse SPD Matrices

## Input and submission

Return the `k` smallest eigenvalues of a sparse symmetric positive-definite matrix, `k = 5`. Any output order is accepted.

The matrix is CSR: `data`, `indices`, `indptr`, and `shape`. Values are Python floats and indexes are Python ints. `n` is the matrix order. Any order of the returned eigenvalues is accepted after sorting. Each value must be an int or float, not a bool, and must match independent dense `eigvalsh` within `1e-6 * max(1, max(abs(target)))`, where target contains only the five requested eigenvalues.

Seeds rotate through `banded`, `clustered`, and `ill_conditioned` spectra.

NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

## Reference and verification

The reference is `scipy.sparse.linalg.eigsh(..., which="SA")`, with a deterministic starting vector and a dense fallback when the matrix is too small for Lanczos or ARPACK does not converge. Upstream used `which="SM"` on a live sparse object. Algebraically smallest eigenvalues and plain CSR lists are intentional; for these positive-definite matrices the two selectors agree.

## Workload distribution

- **Size:** n is square matrix dimension; managed cases use 512 and 1536, requesting five eigenvalues.
- **Selection:** seed % 3 selects banded, clustered, or ill_conditioned.
- **Randomized:** Banded entries or sparse perturbations of the prescribed spectrum. The ill-conditioned family uses a Gershgorin lower bound to choose a positive-definite shift without a dense generator solve.
- **Fixed structure:** Clustered has five eigenvalues near 0.4 and the rest 4–8 before perturbation; ill_conditioned spans 1e-3..1e2. Family formulas define sparsity and conditioning.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark smallest_eigenvalues_sparse_spd
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/sparse_lowest_eigenvalues_posdef` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
