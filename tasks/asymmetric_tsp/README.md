# Asymmetric Traveling Salesperson

## Input and submission

Input `{"distances": matrix}` is a complete directed city graph with positive integer travel costs and a zero diagonal. Return `{"tour": [0, ..., 0], "cost": exact integer}`. The route starts at city 0, visits every other city exactly once, returns to 0, and has minimum total cost. Ties between optimal tours are accepted.

Managed cases contain 12, 48, and 96 cities. Up to 16 cities, general asymmetric regional costs are generated. Larger legal inputs must admit a tour attaining the minimum cycle-cover assignment bound; the generator guarantees this property without supplying the tour or dual potentials.

Input-dependent preprocessing belongs inside the timed call. Inputs are read-only.

## Reference and verification

For n<=16 the reference uses branch-and-bound and verification independently computes the optimum with subset DP. Larger cases use a Hungarian cycle-cover relaxation and tight-edge cycle splicing, with exact tight-edge search as fallback. Verification checks the tour, derives assignment dual potentials, checks every off-diagonal inequality `u[i]+v[j] <= distances[i][j]`, and requires `tour_cost == sum(u)+sum(v)`. This proves optimality in O(n²) certificate checks without exponential optimum recomputation. Solving the relaxation still costs polynomial time. A bug in the bound routine cannot certify a wrong answer unless the inequalities and cost equality actually hold.

Large instances have a planted assignment-tight tour among many tight edges. Candidates still need to recover a Hamiltonian tour, rather than return a bound or a cycle cover with subtours. This structured distribution does not measure arbitrary large NP-hard TSP instances. Only the Python standard library is used.

## Workload distribution

- **Size:** n is city count, 2–256; managed cases are 12, 48, and 96.
- **Selection:** n<=16 uses asymmetric regional costs. At larger n, seed % 3 selects sparse tight arcs, clustered tight arcs, or dense tight arcs.
- **Randomized:** Small cases use regional edge costs and depot increments. Large cases draw row/column potentials 1–59, positive slacks 1–99, extra zero-slack arcs with probabilities 0.18 or 0.42 (clustered: 0.50 within groups, 0.16 across groups), and a shuffled planted zero-slack cycle.
- **Fixed structure:** Small-case regions use city index modulo three. Large-case costs are row potential + column potential + nonnegative slack, and the planted cycle attains the assignment bound. The depot is fixed; all off-diagonal arcs exist. Neither potentials nor the planted tour appear in the input.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark asymmetric_tsp
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

The task adapts the closed-tour contract described by the upstream [AlgoTune TSP task](https://github.com/oripress/AlgoTune/blob/main/AlgoTuneTasks/tsp/description.txt); SPEEDUP-MARK defines its own generator, bounds and exact verifier.
