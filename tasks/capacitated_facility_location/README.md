# Capacitated Facility Location

## Input and submission

Assign every customer to exactly one facility. Opening a facility pays its fixed cost and the facility's total demand cannot exceed its capacity. Minimize fixed cost plus transportation cost.

Input keys are `fixed_costs`, `capacities`, `demands`, and `transportation_costs`, all Python ints. `n` is the number of customers (minimum 3). There are 4 facilities up to 9 customers and 5 after that. Return `objective_value` (the exact integer cost, as an int or float), `facility_status` (one real bool per facility), and `assignments` (0/1 ints, one row per facility). A facility is open exactly when it serves someone. Every optimal assignment is accepted.

Seeds rotate through `tight`, `uneven`, and `competing`. Tight instances need all but one facility. Uneven instances have one large customer. Competing instances place facilities next to each other so many costs are close.

Only the standard library is required. Integers must be exact `int`, never `bool`.

## Reference and verification

The verifier independently searches customer assignments for a cheaper feasible solution, using capacity and transport-cost bounds; it does not rerun the reference facility-subset DP. The executable checker is `_no_cheaper_assignment` in `task_spec.py`.

The reference is the standard facility-indexed subset DP: `F * 3^C` with integer demands. It is a dependency-free adaptation of the upstream binary model. Upstream solves the same model with CVXPY and HiGHS on floating-point data. Integer data is an intentional change so the optimum is exact.

## Workload distribution

- **Size:** n is customer count (minimum 3); four facilities through n=9, otherwise five.
- **Selection:** seed % 3 selects tight, uneven, or competing, in that order.
- **Randomized:** Customer demands/locations, facility locations, and costs. Tight uses demands 2–8 and fixed costs 15–60; uneven has one demand of 30 and others 1–3; competing has fixed costs 30–34 and nearby facility sites.
- **Fixed structure:** Family-specific capacity and cost construction deliberately creates tight capacity, one dominant customer, or competing sites; see _family and generate_problem for exact formulas.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark capacitated_facility_location
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/capacitated_facility_location` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
