# Dynamic Exact Ray Queries

Given a triangle scene and batches of rays, return the nearest intersected
triangle id for each ray, with occasional geometry updates between batches.
Distances are compared with exact integer arithmetic; ids and tie-breaks are
deterministic. Nothing is rendered and no image is compared.

## Input and submission

`problem["scenes"]` is a tuple of scenes. Each contains `triangles` (tuples of
three integer vertices, `|coordinate| <= 2**34`) and `phases`: a sequence of
`{"rays": ((origin, direction), …)}` and `{"updates": (…)}` entries applied in
order, where an update is `("move", id, vertices)`, `("add", vertices)` — which
appends at the next unused id — or `("remove", id)`. Ids stay stable across
updates.

Implement `candidate.py::solve(problem, reference_solve)` and return, per scene,
one tuple of hit ids per ray phase, in ray order.

The scene, phase, and hit sequences must be built-in tuples or lists; hit ids must be exact Python `int` values.

## Semantics

The hit test is Möller–Trumbore in integers: `det`, `u_num`, `v_num` and
`t_num` are exact integers derived from scalar triple products, and every
decision is a cross-multiplication rather than a division. A hit requires
`det != 0` (a ray parallel to the triangle's plane never hits, whatever its
barycentric coordinates would have been), `t > 0`, and barycentric coordinates
inside the closed triangle including its boundary. Boundary contacts are
inclusive, so a ray aimed exactly at a shared vertex or edge is at exactly equal
distance from every triangle that touches it; the answer is then the smallest
triangle id. No hit is `-1`.

## Exact arithmetic

The coordinate and direction ranges produce integer products beyond the
53-bit precision of float64. Rounding can change boundary hits and the
ordering of nearly equal distances. Preserve the exact predicate and
smallest-id tie-break when optimizing intersection tests.

## Scoring and verification

Wall-clock milliseconds for the whole call, so scene preprocessing, hierarchy
construction, refits and updates are all inside the measurement. The verifier
re-derives every answer with the exact predicate and requires the returned
structure to match exactly, including types.

## Reference and verification

The verifier independently intersects rays with triangle planes and uses oriented-edge inclusion tests in exact integer arithmetic. It does not reuse the reference barycentric intersection routine. The executable checker is `_verify_rays` in `task_spec.py`.

`solve` tests every live triangle against every ray with the exact
predicate. Candidates can precompute triangle data, reject triangles with
bounding boxes, or build a spatial hierarchy. Preprocessing and update
costs count toward runtime.

Four families are generated into every problem, and they differ in which
lifecycle wins:

| Family | Regime | What it rewards |
| --- | --- | --- |
| `static_coherent` | One mesh, few large batches from a common origin | Coherent traversal, preprocessing amortized over many rays |
| `moving_bursts` | Four small batches, each preceded by moving a third of the triangles | Cheap updates or refitting; a rebuild-per-batch strategy loses |
| `mixed_sizes` | Unit-sized triangles next to triangles 8192 times larger | Split heuristics and leaf sizing, not median splits |
| `incoherent` | Large batches of unrelated origins and directions, including deliberate misses | Robust traversal; rejected candidates must be cheap |

A spatial hierarchy is the opening, not the whole task: the benchmark then asks
for a better build/refit/traverse strategy, and `moving_bursts` versus
`static_coherent` makes refit-versus-rebuild a genuine choice.

## Workload distribution

- **Size:** n scales scene triangle and ray counts (minimum 8).
- **Selection:** Every input contains static_coherent, moving_bursts, mixed_sizes, and incoherent.
- **Randomized:** Mesh heights, coherent/incoherent rays, and triangle movement jitter within bounded integer coordinates.
- **Fixed structure:** Family-specific geometry scales, phase counts, shared-edge probes, and add/remove schedules are fixed. The four scene builders define exact counts and coordinate bounds.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark dynamic_exact_ray_queries
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
