# Exact k-Nearest Neighbors

## Input and submission

Input contains integer-coordinate `points`, `queries`, and `k`. Return `{"indices": [[point_index, ...], ...]}`, exactly `k` distinct nearest neighbors per query in ascending `(squared Euclidean distance, original index)` order. Index breaks ties deterministically.

Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The baseline sorts all distances; verification checks ordering and that no excluded point outranks the boundary. Defaults: 700 and 1,400 points, approximately one query per seven points; seeds vary 2/4/8 dimensions, 1/5/13 neighbors, uniform and clustered data, duplicate coordinates and exact-hit queries. Unlike upstream, no floating distances or external array/tree library are required. Building an index is part of timed work.

## Workload distribution

- **Size:** n is point count; max(1,n//7) queries.
- **Selection:** seed % 3 selects dimensions/k as (2,1), (4,5), (8,13); odd seeds use seven clusters, even seeds one region. Joint cycle: 6.
- **Randomized:** Integer point jitter -100..100; cluster centers at multiples of 1000; random queries within the family bounding range.
- **Fixed structure:** Every ninth query copies a data point exactly. k is capped by n.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark exact_k_nearest_neighbors
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/kd_tree/description.txt` (https://github.com/oripress/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.
