# Asymmetric Traveling Salesperson

Input `{"distances": matrix}` is a complete directed city graph with positive integer travel costs and a zero diagonal. Return `{"tour": [0, ..., 0], "cost": exact integer}`. The route starts at city 0, visits every other city exactly once, returns to 0, and has minimum total cost. Ties between optimal tours are accepted.

The 10- and 12-city generators combine regional cost structure, directed costs and hub variation in every instance. The reference uses branch-and-bound with independent incoming- and outgoing-edge relaxations. The verifier checks the route and independently computes the exact optimum with subset dynamic programming. The small bounded instances keep the exact verifier practical. Only the Python standard library is used.

Optimize `candidate.py:solve(problem, reference_solve)`. Input-dependent preprocessing belongs inside the timed call. Inputs are read-only. The task adapts the closed-tour contract described by the upstream [AlgoTune TSP task](https://github.com/oripress/AlgoTune/blob/main/AlgoTuneTasks/tsp/description.txt); SPEEDUP-MARK defines its own generator, bounds and exact verifier.

## Workload distribution

- **Size:** n is city count, from 2 through 16.
- **Selection:** One asymmetric three-region distance distribution.
- **Randomized:** Within-region edge costs 3–42, cross-region costs 24–115, and extra depot edge costs 0–18.
- **Fixed structure:** Region membership is city index modulo three; the depot is fixed and every off-diagonal edge exists.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
