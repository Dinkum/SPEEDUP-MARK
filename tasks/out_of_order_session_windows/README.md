# Out-of-Order Session Window Trace

## Input and submission

Process `(event_id, key, timestamp)` records in arrival order. After observing each timestamp, the watermark is `max_seen_timestamp - allowed_lateness`; an event strictly older than that watermark is dropped. Accepted points join every live same-key session within `gap` on either side, so one point may bridge and merge sessions. A session expires when `end + gap < watermark`. Output records are `("drop", event_id, key, timestamp, watermark)`, `("upsert", event_id, key, start, end, count, merged_session_count, watermark)`, and `("emit", key, start, end, count, watermark)`. Expiries after one arrival are ordered by key then session tuple. The final flush is ordered the same way and uses `None` as its watermark.

More precisely, an accepted timestamp joins every live same-key session satisfying `start - gap <= timestamp <= end + gap`. The resulting session has the minimum touched start or event timestamp, the maximum touched end or event timestamp, and `count = 1 + sum(touched session counts)`. `merged_session_count` is the number of existing sessions absorbed, or zero for a new session. Every arrival first appends exactly one drop or upsert record; expiry records caused by that arrival follow it and use that arrival's watermark. Expiry and final-flush records are ordered by `(key, start, end, count)`.

The complete trace and each record may be a list or tuple. Leaf values must retain their exact contract types and values.

## Reference and verification

The verifier independently reclusters retained raw timestamps at every event and watermark, rather than updating the reference session intervals. The executable checker is `_verify_sessions` in `task_spec.py`.

The reference keeps sorted per-key session lists and scans them. Candidates can use interval indexes, heaps for expiry, and key-local search. Event processing, merging, and trace construction are timed.

## Workload distribution

- **Size:** n is event count.
- **Selection:** One event distribution combining local disorder and deliberately late records.
- **Randomized:** User keys from up to 24 users, timestamp jitter -18..18 about 3*event_id, and local swaps.
- **Fixed structure:** Session gap 20 and lateness allowance 45; swaps every seventh position; records at positions 97,308,... are moved 240 time units into the past.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark out_of_order_session_windows
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

This is an original lightweight adaptation inspired by Terminal-Bench 4's [`session-window-debug`](https://github.com/harbor-framework/terminal-bench/blob/v4.0.0/tasks/session-window-debug/instruction.md). No source code was copied.
