# Minimum Spanning Tree

## Input and submission

Input is `{"num_nodes": n, "edges": [[u,v,weight], ...]}`, a connected simple undirected graph with exact signed integer weights. Return `{"edge_indices": [...]}`: strictly increasing indices of `n-1` original edges forming any minimum-weight spanning tree. Ties permit different trees.

Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The baseline is heap Prim; an independent union-find Kruskal verifier checks optimal weight and explicitly rejects cyclic witnesses. Defaults: 500 and 1,000 nodes; seeded families vary graph density and include repeated and negative weights. Unlike upstream, witnesses reference input edges and all arithmetic is exact integer arithmetic.

## Workload distribution

- **Size:** n is vertex count.
- **Selection:** seed % 3 selects 4, 12, or 25 extra edge attempts per vertex.
- **Randomized:** A random parent for each non-root vertex, additional endpoint pairs, edge weights -100..1000, and shuffled edge order.
- **Fixed structure:** The parent tree guarantees connectivity; self-loops are omitted and duplicate pairs collapse.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark minimum_spanning_tree
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/minimum_spanning_tree/description.txt` (https://github.com/oripress/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.
