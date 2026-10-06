# Earth Mover's Distance

## Input and submission

Move the source mass onto the target mass at minimum cost.

Input is `source_weights`, `target_weights`, and `cost_matrix`. The two weight vectors are nonnegative and sum to the same total. Their lengths may differ, and some bins may have zero mass. Return `transport_plan`, a matrix of nonnegative Python floats. The plan must reproduce both marginals with absolute error at most `1e-6 * max(1, total_mass)` per marginal entry, and its cost must match the optimum with absolute error at most `1e-6 * max(1, abs(optimal_cost))`. Entry values down to `-1e-9` are accepted as numerical roundoff. Every optimal plan is accepted, including plans that differ where costs are tied. Correct marginals with a worse cost are rejected.

Seeds rotate through `unequal`, `sparse`, and `tied`.

NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

## Reference and verification

The verifier solves the transport dual, validates its cost inequalities, and compares the feasible plan cost to that lower bound. It does not rerun the primal reference model. The executable checker is `_transport_dual_bound` in `task_spec.py`.

The reference is SciPy's HiGHS linear solver on the transportation equalities. Upstream compares one POT network-simplex basis elementwise. Accepting every optimal plan is intentional.

## Workload distribution

- **Size:** n is source-support size; destination size depends on family.
- **Selection:** seed % 3 selects unequal, sparse, or tied.
- **Randomized:** Unequal uses random masses, max(2,2*n//3) destinations, and costs up to 10; sparse has zero-mass prefixes and random costs up to 8; tied has uniform masses and integer costs 0–3.
- **Fixed structure:** Mass vectors are normalized. Support-size ratios and zero-mass construction are family-specific.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark earth_movers_distance
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/earth_movers_distance` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
