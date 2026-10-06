# Graph Coloring

## Input and submission

Color an undirected graph properly with as few colors as possible.

The input is a symmetric 0/1 adjacency matrix. `n` is the number of vertices. There is no diagonal. Return a list of Python ints, one color per vertex. Colors are positive and `bool` is rejected. Any labeling is accepted; optimality is the number of distinct colors.

Seeds rotate through `sparse`, `dense`, `symmetric`, and `planted`. Symmetric instances expand random base graphs into interchangeable vertex classes. Planted instances only put edges between a fixed 3-coloring (two groups below six vertices).

Only the standard library is required.

## Reference and verification

The reference is a maximum-clique lower bound, a largest-first greedy coloring, and DSATUR branch and bound when those differ. Upstream uses CP-SAT with the same input and output contract. The exact search is a dependency-free adaptation.

## Workload distribution

- **Size:** n is vertex count.
- **Selection:** seed % 4 selects sparse, dense, symmetric, or planted.
- **Randomized:** Sparse/dense edges use probabilities 0.18/0.45; planted edges use probability 0.75 across three groups (two below n=6). Symmetric graphs draw a base graph with density uniform in [0.25,0.7], then replace base vertices by shuffled, variable-size false-twin classes.
- **Fixed structure:** Symmetric base size is min(n,max(3,n//2)); both base adjacency and twin multiplicities vary, so the coloring optimum is not fixed. Planted group membership is structural; its edges vary.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark graph_coloring
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/graph_coloring_assign` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
