# Temporal Per-Entity As-Of Join

## Input and submission

Each dimension row is `(entity, timestamp, sequence, value)`, and each event is `(event_id, entity, event_time)`. For every event in input order, select the dimension row for the same entity with the greatest timestamp less than or equal to the event time. Equal timestamps choose the greatest integer sequence; if both timestamp and sequence tie, the later dimension input position wins. Return `(event_id, value)` or `(event_id, None)` when no eligible row exists. Event IDs need not be sorted or unique and do not affect selection.

The complete result and each result row must be a built-in list or tuple. Leaf values must retain the exact types and values produced by the contract.

## Reference and verification

The verifier independently sorts each entity history and binary-searches the latest eligible version, including timestamp/sequence/input-position ties. The executable checker is `_verify_join` in `task_spec.py`.

The reference scans every dimension row for every event. Candidates can group by entity, collapse duplicate keys with the input-position rule, sort versions, and binary-search event times, or sort both sides for a merge join. All indexing and joining is timed.

## Workload distribution

- **Size:** n is base version count and event count, plus scheduled duplicate versions.
- **Selection:** One temporal-join distribution with repeated times/sequences and unmatched events.
- **Randomized:** Entities from min(120,n//35 + 1), version times, sequence values 0–3, event times, and shuffled versions.
- **Fixed structure:** Every 67th version is duplicated. Event times extend before and after the version timestamp range.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark temporal_asof_join
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
