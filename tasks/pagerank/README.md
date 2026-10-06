# Sparse Weighted PageRank

## Input and submission

Input `{"weights": matrix, "damping": d, "tolerance": eps}` describes a directed weighted graph. Matrix entries are nonnegative integer edge weights; an all-zero row is a dangling node. Return `{"scores": [float, ...]}` in vertex order. Scores must be nonnegative, sum to one, and be within `eps` in L1 distance of the unique PageRank fixed point. Any vector meeting the residual certificate is accepted.

Each instance combines local community links, cross-community links, a shared hub and dangling vertices. Sizes are 256 and 384 vertices.

Inputs are read-only. A verified residual `||F(x)-x||_1 <= (1-damping)*tolerance` bounds the fixed-point error by the declared tolerance.

## Reference and verification

The reference uses dense power iteration; fresh managed runs delegate to that reference. Converting the matrix to sparse outgoing rows is an accessible first optimization. The verifier checks a contraction-based residual bound directly and does not run the reference. This is a standard-library task; input-dependent conversion and iteration are timed.

## Workload distribution

- **Size:** n is vertex count, from 4 through 600.
- **Selection:** One weighted community graph distribution with hubs and dangling vertices.
- **Randomized:** Additional edge targets and integer edge weights 1–9.
- **Fixed structure:** Structural successor/community/hub edges, scheduled dangling nodes, damping 0.85, and tolerance 1e-9 are fixed; target count is bounded near seven per non-dangling row.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark pagerank
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
