# Three-Resource 0/1 Knapsack

Choose each item zero or one time. Item `i` has a positive integer `profits[i]` and three positive integer resource weights `weights[i]`. The total selected weight must not exceed any entry in `capacities`. Return `{"profit": exact integer}` for the maximum achievable profit.

Each generated instance mixes items that favor different resources, so one profit-to-weight ordering does not dominate all cases. Sizes are 24 and 30 items. Capacities keep the exact state space bounded. The reference uses branch-and-bound with three one-resource fractional relaxations; the verifier independently computes the optimum by enumerating attainable three-dimensional resource vectors. The starter candidate delegates to the branch-and-bound reference. All code uses the Python standard library.

Optimize `candidate.py:solve(problem, reference_solve)`. The starter delegates to the reference so its baseline does not begin slower than the measured reference. Preprocessing and all candidate work belong inside the timed call. Inputs are read-only. The task is an independently authored SPEEDUP-MARK adaptation of the AlgoTune `multi_dim_knapsack` placeholder; this contract defines its complete input, output, and scoring semantics.

## Workload distribution

- **Size:** n is item count, from 1 through 40; three resource dimensions.
- **Selection:** One item distribution with seed % 3 rotating capacity ratios 0.25, 0.27, 0.26 across dimensions.
- **Randomized:** Resource weights 1–6 plus 1–5 on each item's favored dimension; profit coefficients 2–7 plus favored premium 3 and noise 0–12.
- **Fixed structure:** Favored dimension cycles by item index. Capacities are rounded fractions of total weights, clamped to at least one.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
