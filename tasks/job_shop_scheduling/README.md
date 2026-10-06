# Fixed-Route Job-Shop Makespan

## Input and submission

Each job is a fixed sequence of three operations. An operation is `(machine, duration)`, has a positive integer duration, and cannot be interrupted. Operations within a job must follow their listed order, and each machine can process only one operation at a time. Return `{"start_times": [[...], ...], "makespan": exact integer}` with one start time per operation in job order. Minimize the maximum operation finish time; any optimal schedule is accepted. Intentional idle time is allowed.

Every instance has machine routes that differ by job and a long operation on each machine, so route precedence and machine bottlenecks interact. Sizes are three and four jobs (nine or twelve operations).

All schedule construction and input-dependent preprocessing are timed. Inputs are read-only. This is a SPEEDUP-MARK adaptation of the standard fixed-route job-shop makespan problem, with an intentionally small exact-search domain.

## Reference and verification

The reference searches machine operation orders and prunes partial orders with critical-path bounds. The independent verifier checks feasibility and enumerates complete machine orders using longest-path constraint relaxation. The bounded exact instances use only the Python standard library.

## Workload distribution

- **Size:** n is job count, from 2 through 4; each job visits three machines.
- **Selection:** One randomized routing distribution; (machine + seed) % n assigns each machine's bottleneck job.
- **Randomized:** Independent shuffled machine routes; bottleneck durations 8–13, other durations 2–7.
- **Fixed structure:** Each job has exactly three operations; the bottleneck assignment has period n.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark job_shop_scheduling
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
