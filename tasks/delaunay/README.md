# Delaunay Triangulation

## Input and submission

Compute a Delaunay triangulation of a planar point set and the edges of its boundary.

Input is `points`, a list of at least three distinct integer `[x, y]` coordinates that are not all collinear. `n` is the number of points. Return `simplices`, a list of three integer vertex indexes per triangle, and `convex_hull`, a list of undirected boundary edges. The triangles must form a complete nonoverlapping triangulation of the convex hull, use every point, and contain no duplicates or crossing edges. Interior edges join two oppositely oriented triangles after orientation normalization; boundary edges must match the independently computed hull. Duplicate hull edges are invalid. Triangle order, starting vertex, and orientation do not matter. A point is inside a circumcircle only when the exact incircle predicate is positive. A zero predicate is cocircular: either diagonal is a valid Delaunay edge. Collinear boundary points stay in the triangulation and split the boundary edge.

Seeds rotate through `clustered`, `nearly_collinear`, and `nearly_cocircular`.

NumPy and SciPy are already the optional numerical extra; do not install packages. Set thread caps are one OpenMP / BLAS thread.

## Reference and verification

The reference is `scipy.spatial.Delaunay` (Qhull). On these integer families its triangles and hull satisfy the exact predicate, including the collinear and rounded-cocircular sets. A pure-Python Bowyer-Watson insertion is slower and is not the reference.

## Workload distribution

- **Size:** n is distinct planar point count (minimum 3).
- **Selection:** seed % 3 selects clustered, nearly_collinear, or nearly_cocircular.
- **Randomized:** Clustered: centers in [-200,200]^2 with lattice radius max(4,isqrt(n)//2+1). Nearly collinear: x gaps 96–320, y jitter -2..2, and two off-axis anchors with vertical magnitude 160–288. Nearly cocircular: radius 4000+10*n through 8000+10*n, one random angle in each angular stratum, and radial jitter -3..3. All point orders are shuffled.
- **Fixed structure:** Every family preserves its geometric regime while changing geometry. An entirely collinear draw moves its final point one unit off the line to preserve a planar instance. Integer coordinates support exact predicate verification; these are not uniform samples of all legal point sets.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark delaunay
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/delaunay` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. Integer coordinates and the exact-predicate acceptance rule are intentional. No upstream source was copied.
