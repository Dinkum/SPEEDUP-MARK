# Vertex-Labeled Graph Isomorphism

## Input and submission

Return a tuple mapping every left vertex index to a right vertex index, or `None` when no isomorphism exists. A valid mapping is a label-preserving bijection that preserves the complete undirected edge set. Self-loops are absent. Generated graphs use the requested vertex count, including the 20-vertex grading case. Positive and negative pairs are connected and share degree and label histograms. Negative pairs use degree-preserving edge switches and are certified by exact search; connectivity alone cannot distinguish them.

Mappings must be built-in tuples or lists of exact integers.

## Reference and verification

The reference uses constrained backtracking. Candidates can improve refinement, branching order, canonical signatures, and symmetry handling. All search is timed; problem generation and verification are not.

## Workload distribution

- **Size:** n is the actual vertex count; declared cases have 10 and 20 vertices.
- **Selection:** seed % 3 selects labeled positive, uniform-label positive, or connected negative (negative only for n>=6).
- **Randomized:** All graphs begin with a cycle (an edge at n=2). Labeled positives add chords with a sampled density 0.15–0.35 and up to three labels. Other cases add a random matching to the cycle. Both sides are independently permuted. Negatives apply degree-preserving edge switches and retain connected, exactly certified non-isomorphic pairs.
- **Fixed structure:** Both sides have equal label and degree histograms and are connected. Negative generation is rejection-conditioned: at most 32 base graphs, each with 32 attempted switches, then a clear generation failure. The exact baseline orders vertices by adjacency to already selected vertices; there is no size clamp.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark labeled_graph_isomorphism
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

This is an original lightweight adaptation inspired by Terminal-Bench 4's [`vf2-speedup-networkx`](https://github.com/harbor-framework/terminal-bench/blob/v4.0.0/tasks/vf2-speedup-networkx/instruction.md). No source code was copied.
