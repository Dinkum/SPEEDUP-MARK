# Checksummed Durable Log Recovery

## Input and submission

Each segment has an integer `segment_id` and a physical record sequence. A record is `(lsn, payload_bytes, crc32)`, where the checksum covers the unsigned little-endian 64-bit LSN followed by the payload. Only the valid prefix before the first malformed or corrupt record in each segment is durable. For duplicate LSNs, choose the record from the lowest segment ID, then the earliest physical offset. Starting at `start_lsn`, return `(lsn, payload, segment_id)` records until the first missing LSN; later records are never recovered across that gap.

LSNs must be exact integers in `0 <= lsn < 2**64`; an out-of-range LSN ends that segment's durable prefix just like any other malformed record. Return a built-in list or tuple of built-in list or tuple rows; scalar leaves must have the exact expected types and values.

## Reference and verification

The verifier sorts valid physical records by logical sequence and priority, independently of the reference dictionary selection. The executable checker is `_verify_recovery` in `task_spec.py`.

The reference scans all durable prefixes into a dictionary and then walks the sequence. Candidates can reduce allocations, reject losing duplicates earlier, and fuse selection with continuity tracking. Parsing, checksumming, and recovery are timed.

## Workload distribution

- **Size:** n is the base record count.
- **Selection:** Every input combines valid records, duplicates, a sequence gap, and a torn segment tail.
- **Randomized:** Record payloads, segment assignment, and record/segment ordering.
- **Fixed structure:** Segment count is min(16, n//250 + 1); duplicate and gap/torn-tail schedules are prescribed by generate_problem.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark durable_log_recovery
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

This is an original, narrower adaptation inspired by Terminal-Bench 4's [`wal-recovery-ordering`](https://github.com/harbor-framework/terminal-bench/blob/v4.0.0/tasks/wal-recovery-ordering/instruction.md). Its checksum-defined durable-prefix contract is specific to SPEEDUP-MARK, and no source code was copied.
