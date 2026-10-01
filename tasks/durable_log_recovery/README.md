# Checksummed Durable Log Recovery

Each segment has an integer `segment_id` and a physical record sequence. A record is `(lsn, payload_bytes, crc32)`, where the checksum covers the unsigned little-endian 64-bit LSN followed by the payload. Only the valid prefix before the first malformed or corrupt record in each segment is durable. For duplicate LSNs, choose the record from the lowest segment ID, then the earliest physical offset. Starting at `start_lsn`, return `(lsn, payload, segment_id)` records until the first missing LSN; later records are never recovered across that gap.

The reference scans all durable prefixes into a dictionary and then walks the sequence. Candidates can reduce allocations, reject losing duplicates earlier, and fuse selection with continuity tracking. Parsing, checksumming, and recovery are timed.

LSNs must be exact integers in `0 <= lsn < 2**64`; an out-of-range LSN ends that segment's durable prefix just like any other malformed record. Return a plain built-in list or tuple of plain built-in list or tuple rows. Subclasses and lazy sequences are rejected before their callbacks can run; scalar leaves must have the exact expected types and values.

Edit `candidate.py` and run `python -m speedupmark.harness tasks/durable_log_recovery`.

This is an original, narrower adaptation inspired by Terminal-Bench 4's [`wal-recovery-ordering`](https://github.com/harbor-framework/terminal-bench/blob/v4.0.0/tasks/wal-recovery-ordering/instruction.md), not a faithful port. Its checksum-defined durable-prefix contract is specific to SPEEDUP-MARK, and no source code was copied.

## Workload distribution

- **Size:** n is the base record count.
- **Selection:** Every input combines valid records, duplicates, a sequence gap, and a torn segment tail.
- **Randomized:** Record payloads, segment assignment, and record/segment ordering.
- **Fixed structure:** Segment count is min(16, n//250 + 1); duplicate and gap/torn-tail schedules are prescribed by generate_problem.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
