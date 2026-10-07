# Battery-Limited Fleet Tours

## Input and submission

Compute the exact minimum dispatch cost on a changing directed road network. Each problem contains independent `scenarios`. A scenario has `family`, `node_count`, initial `edges`, and an ordered `operations` stream. Vertices are numbered from zero. Each edge is `(u, v, time, energy)` with positive integer travel time and energy use. The two directions are distinct, and a scenario has at most one edge for each ordered pair. `("set", u, v, time, energy)` inserts or replaces an edge; `("delete", u, v)` removes it. Deleting a missing edge does nothing.

Vehicle reports arrive as `("telemetry", vehicle_id, timestamp, location, capacity, skills, battery)`. A dispatch is `("dispatch", cutoff, vehicle_ids, jobs)`; vehicle IDs are distinct within that operation. Only reports already received in the stream count. For each requested vehicle, use its report with the greatest timestamp at or before `cutoff`; equal timestamps choose the later arrival. A vehicle without a qualifying report is unavailable. Cutoffs need not increase, and reports may arrive out of event-time order. Capacity, skills, and battery are nonnegative integers; each job's required skill is one positive bit.

Each job is `(location, required_skill, unserved_penalty)`, with a nonnegative integer penalty. A selected vehicle may perform **one closed tour**: start at its reported location, explicitly serve at most `capacity` distinct jobs whose skill bit it has, and return to its starting location. Road walks between stops may revisit vertices; merely passing through a job's location does not serve it. Energy use accumulates across the whole tour, including the return, and cannot exceed that vehicle's battery. Each job is served by at most one vehicle or incurs its penalty. The objective is the sum of road travel times across all tours plus penalties for unserved jobs. There is no charging or shared road-capacity constraint.

Return one built-in list or tuple of exact minimum integer costs per scenario, with one cost per dispatch in order. The outer result may also be a built-in list or tuple. Booleans, floats, and approximate answers are invalid. The complete stream is available to the candidate; all graph preparation, indexing, update handling, path search, tour planning, and fleet allocation are timed. No separate memory limit is enforced.

Every generated input includes grid, hub, and bottleneck networks with slow, low-energy roads and faster, energy-intensive shortcuts. Reports and jobs persist across rounds; later rounds change a few or many roads and jobs. The bottleneck closes and reopens. Default grading uses 60 vertices per network; managed grading uses 60 and 96. These sizes carry respectively 10/12 jobs, 3/4 vehicles, and 8/10 dispatch rounds per network. Vehicle capacities alternate between three and four jobs, with battery budgets 30–44 and overlapping skills. Jobs initially share a bounded set of stops, then disperse during update bursts. Longer feasible tours compete for the same jobs, exposing subset routing and allocation rather than adding operational rules.

The task combines resource-constrained path search, multi-stop routing, and fleet allocation. Optimizations can change search direction, prune dominated labels, cache profiles across stable rounds, repair after road changes, reorder tour enumeration, reuse route choices, or bound the fleet search. A scalar shortest travel time for each leg is insufficient: a slower leg can save battery needed later in a tour. The bounded instances make exact grading practical; they are not claims of a new asymptotic shortest-path result.

## Reference and verification

The reference computes exact-energy road profiles, retains nondominated time/energy options, plans bounded tours by subset and endpoint, and allocates disjoint tours by dynamic programming. Verification uses reverse sparse Pareto-label searches, enumerates ordered tours, and solves the fleet choice with a separate recursive exact-cover search. Both calculate the optimum exactly.

## Workload distribution

- **Size:** n is road-node count, clamped to 20.
- **Selection:** Every input contains grid, hubs, and bottleneck, each with concentrated and dispersed job phases.
- **Randomized:** Road costs, job stops/rewards/skills, depots, vehicle telemetry, batteries, and road updates. Normal roads cost 5–9 time/1 energy; express roads cost 1–3 time/4–6 energy; initial batteries are 30–44.
- **Fixed structure:** At size <=64 there are 10 jobs, 3 vehicles, 8 rounds; otherwise 12 jobs, 4 vehicles, 10 rounds. Capacities alternate 3/4. Phase and bottleneck-closure schedules are fixed.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark live_fleet_dispatch
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
