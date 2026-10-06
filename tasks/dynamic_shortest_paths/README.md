# Dynamic Shortest-Path Query Engine

## Input and submission

Process changes and exact distance queries on directed graphs with positive Python integer edge weights. The input is `{"scenarios": (...)}`; each independent scenario contains a descriptive `family`, `node_count`, initial `edges`, and `operations`. Vertices are integers from zero through `node_count - 1`. Initial edges are unique `(source, target, weight)` triples. An edge is identified by its ordered endpoint pair, so the two directions are distinct.

Operations are `("set", source, target, weight)`, `("delete", source, target)`, and `("query", source, target)`. A set inserts an edge or replaces its weight, including reopening a deleted edge. Delete removes only that direction; deleting a missing edge is a no-op. Return one list or tuple per scenario with one shortest-distance answer per query, in order. The outer result may also be a built-in list or tuple. An unreachable target returns `None`; a source equal to its target returns integer zero, even if isolated. Distances must be exact Python integers; booleans and floats are rejected. General legal inputs can include positive self-loops.

Every generated problem contains a directed grid, a graph with high-degree hubs, and two sparse regions connected by a narrow bottleneck. Sizes 400 and 800 each run every family. `n` controls vertices per scenario and roughly twice that many stream operations. Each stream mixes repeated sources, repeated targets, cold queries, and bursts of edge updates. Bottleneck scenarios explicitly close both bridge directions, query the disconnected graph, and reopen the bridge. Weight replacements include increases and decreases.

## Reference and verification

The starting reference builds adjacency maps and runs heap-based Dijkstra with target early termination for each query. An independent bidirectional search checks the answers. Candidate improvements can combine bidirectional search, source/target distance caches, dependency-aware invalidation, shortest-path-tree repair, batched query planning, and topology-specific indexes. All candidate preprocessing, updates, and queries are timed; no prebuilt input-specific state is provided. There is no separately enforced memory budget. The task remains exact: approximation, stale caches, and route estimates are invalid.

## Workload distribution

- **Size:** n scales graph vertices, clamped to at least 8.
- **Selection:** Every input contains grid, hubs, and bottleneck graphs.
- **Randomized:** Edge weights (initially 1–30), updates (1–80), graph choices, and hot/cold query endpoints.
- **Fixed structure:** Each graph has roughly 2*n operations; phase query probabilities are 0.95, 0.70, 0.15, and 0.90, with scheduled bridge closures and reopenings.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark dynamic_shortest_paths
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
