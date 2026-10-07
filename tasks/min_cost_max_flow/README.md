# Exact Minimum Cost Maximum Flow

## Input and submission

Input has `num_nodes`, `source`, `sink`, and `edges` containing `[u,v,capacity,cost]`. Graphs are simple directed acyclic networks with positive integer capacities and nonnegative integer costs. Return `{"flow": [integer,...]}` aligned to the original edges. Among all feasible flows, maximize source-to-sink flow first, then minimize total cost. All optimal witnesses are accepted.

Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The baseline repeatedly augments shortest residual paths using Bellman-Ford. Verification independently checks capacities/conservation, absence of an augmenting path, and absence of any negative residual cycle. Defaults: 100 and 180 vertices; seeds vary topology, density, capacities, and cost ties. Unlike upstream's dense matrices, this adaptation uses explicit edge lists and exact integer arithmetic.

## Workload distribution

- **Size:** n is graph vertex count, clamped to at least 2.
- **Selection:** seed % 5 selects 3–7 random edge attempts per vertex; seed parity selects cost range. Joint cycle: 10.
- **Randomized:** Forward-only edge endpoints (u < v), capacities 1–14, and nonnegative costs 0–2 on odd seeds or 0–39 on even seeds.
- **Fixed structure:** A structural path connects source 0 to sink n-1. Every generated graph is acyclic; duplicate edge attempts collapse.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark min_cost_max_flow
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/max_flow_min_cost/description.txt` (https://github.com/oripress/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.
