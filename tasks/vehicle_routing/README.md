# Vehicle Routing

## Input and submission

Visit every non-depot location exactly once using exactly `K` nonempty routes that start and end at the depot. Minimize total distance. There are no capacity constraints.

Input is `D`, `K`, and `depot`. `D` is a symmetric matrix of positive integer distances with a zero diagonal. `n` is the number of customers; the depot is an extra location at index 0. Return a list of `K` routes. Each route is a list of integer indexes beginning and ending at the depot. Route order does not matter. Any optimal set of routes is accepted.

Seeds rotate through `clustered`, `spread`, and `lattice`. Clustered instances use one vehicle per cluster. Lattice instances repeat distances.

Only the standard library is required.

## Reference and verification

The verifier independently computes minimum completion cost over customer visits and route boundaries, rather than reusing the reference subset-tour and partition tables. The executable checker is `_route_trace_cost` in `task_spec.py`.

The reference builds a Held-Karp table of depot-returning subset costs, then partitions the customers into exactly `K` nonempty subsets. Upstream's description requires symmetric distances, while its generator draws asymmetric integers and its solver is CP-SAT with MTZ constraints. Symmetry and the subset DP are intentional.

## Workload distribution

- **Size:** n is customer count, plus one depot.
- **Selection:** seed % 3 selects clustered, spread, or lattice.
- **Randomized:** Clustered centers in [-40,40]^2 with jitter -3..3; spread coordinates in [-50,50]^2. Lattice samples n+1 distinct occupied cells from a square grid with independently drawn row/column gaps 3–9; the first sampled cell is the depot.
- **Fixed structure:** Lattice side is max(3,isqrt(n)+1)+1 grid coordinates. Vehicle count depends on family/size, 2 or 3; symmetric distances are max(1,round(10*Euclidean distance)). Occupancy and unequal spacing change route costs rather than only rescaling a fixed instance.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark vehicle_routing
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/vehicle_routing` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
