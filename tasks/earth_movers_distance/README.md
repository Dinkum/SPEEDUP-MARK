# Earth Mover's Distance

Move the source mass onto the target mass at minimum cost.

Input is `source_weights`, `target_weights`, and `cost_matrix`. The two weight vectors are nonnegative and sum to the same total. Their lengths may differ, and some bins may have zero mass. Return `transport_plan`, a matrix of nonnegative Python floats. The plan must reproduce both marginals with absolute error at most `1e-6 * max(1, total_mass)` per marginal entry, and its cost must match the optimum with absolute error at most `1e-6 * max(1, abs(optimal_cost))`. Entry values down to `-1e-9` are accepted as numerical roundoff. Every optimal plan is accepted, including plans that differ where costs are tied. Correct marginals with a worse cost are rejected.

Seeds rotate through `unequal`, `sparse`, and `tied`.

The reference is SciPy's HiGHS linear solver on the transportation equalities. Upstream compares one POT network-simplex basis elementwise. Accepting every optimal plan is intentional.

Optimize `candidate.py:solve(problem, reference_solve)`. NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

Provenance: AlgoTune `AlgoTuneTasks/earth_movers_distance` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.

## Workload distribution

- **Size:** n is source-support size; destination size depends on family.
- **Selection:** seed % 3 selects unequal, sparse, or tied.
- **Randomized:** Unequal uses random masses, max(2,2*n//3) destinations, and costs up to 10; sparse has zero-mass prefixes and random costs up to 8; tied has uniform masses and integer costs 0–3.
- **Fixed structure:** Mass vectors are normalized. Support-size ratios and zero-mass construction are family-specific.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
