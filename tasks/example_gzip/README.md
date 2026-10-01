# Example gzip task

A minimal task template. Return gzip-compressed bytes that decompress exactly to the input. The reference uses `gzip.compress`; verification uses `gzip.decompress`. Copy this directory to start a new task.

## Workload distribution

- **Size:** n is byte count.
- **Selection:** One deliberately simple template distribution.
- **Randomized:** SHA-256 of the decimal seed supplies a 32-byte motif.
- **Fixed structure:** The motif repeats to length n. This example is not a general compression workload and has no compressed-size bound.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
