# Articulation Points

## Input and submission

Input `{"num_nodes": n, "edges": [[u,v], ...]}` is a simple undirected graph, potentially disconnected. Return `{"articulation_points": [...]}`, the ascending list of vertices whose removal increases total connected-component count.

Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The exact baseline removes each vertex and recounts components; the grader checks output format and compares against the measured reference answer. Seeded graphs mix dense local blocks, bridges and disconnected regions. Defaults are 300 and 600 vertices. This is a bounded stdlib adaptation with the same graph-theoretic objective as upstream.

## Workload distribution

- **Size:** n is vertex count.
- **Selection:** seed % 2 chooses disconnected blocks (0) or a connected chain (1); seed % 4 chooses 1–4 local edge attempts per vertex. Joint cycle: 4.
- **Randomized:** Extra edges within blocks of up to 19 vertices.
- **Fixed structure:** Chain edges and block boundaries are structural; duplicate edge attempts collapse.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark articulation_points
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/articulation_points/description.txt` (https://github.com/oripress/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.
