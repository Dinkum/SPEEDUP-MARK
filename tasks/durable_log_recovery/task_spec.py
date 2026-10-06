"""Recover a contiguous durable sequence from checksummed log segments."""

from __future__ import annotations

import random
import zlib
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            type(actual) in (tuple, list)
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _checksum(lsn, payload):
    return zlib.crc32(lsn.to_bytes(8, "little", signed=False) + payload) & 0xFFFFFFFF


def _recover(problem):
    chosen = {}
    for segment in problem["segments"]:
        segment_id = segment["segment_id"]
        for offset, record in enumerate(segment["records"]):
            if not isinstance(record, tuple) or len(record) != 3:
                break
            lsn, payload, checksum = record
            if (
                type(lsn) is not int
                or not 0 <= lsn < 1 << 64
                or not isinstance(payload, bytes)
                or type(checksum) is not int
                or checksum != _checksum(lsn, payload)
            ):
                break
            priority = (segment_id, offset)
            if lsn not in chosen or priority < chosen[lsn][0]:
                chosen[lsn] = (priority, payload)

    recovered = []
    lsn = problem["start_lsn"]
    while lsn in chosen:
        (segment_id, _), payload = chosen[lsn]
        recovered.append((lsn, payload, segment_id))
        lsn += 1
    return tuple(recovered)


def _verify_recovery(problem):
    """Sort physical candidates by LSN and priority, then consume the prefix."""
    from itertools import groupby
    records = []
    for segment in problem["segments"]:
        for offset, record in enumerate(segment["records"]):
            if type(record) is not tuple or len(record) != 3:
                break
            lsn, payload, checksum = record
            if (type(lsn) is not int or not 0 <= lsn < 2**64
                    or type(payload) is not bytes or type(checksum) is not int):
                break
            if zlib.crc32(lsn.to_bytes(8, "little") + payload) & 0xffffffff != checksum:
                break
            records.append((lsn, segment["segment_id"], offset, payload))
    result, expected = [], problem["start_lsn"]
    # Keep stable input order if two physical priorities are identical; payload
    # bytes must not silently become an additional duplicate tie-break.
    for lsn, copies in groupby(sorted(records, key=lambda row: row[:3]), key=lambda row: row[0]):
        if lsn < expected:
            continue
        if lsn != expected:
            break
        _, segment_id, _, payload = next(copies)
        result.append((lsn, payload, segment_id))
        expected += 1
    return tuple(result)

class DurableLogRecoveryTask:
    name = "durable_log_recovery"
    task_version = "1.1.2"
    display_name = TASK_CATALOG[name].display_name
    default_n = 12000
    grading_cases = (12000, 24000)

    def generate_problem(self, n=12000, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        segment_count = max(1, min(16, n // 250 + 1))
        buckets = [[] for _ in range(segment_count)]
        gap = n // 2 + 1 if n >= 8 else n + 1
        for lsn in range(1, n + 1):
            if lsn == gap:
                continue
            payload = f"record:{lsn}:{rng.randrange(1 << 30)}".encode()
            segment = rng.randrange(segment_count)
            buckets[segment].append((lsn, payload, _checksum(lsn, payload)))
            if lsn % 11 == 0:
                duplicate_segment = (segment + 1) % segment_count
                duplicate = payload + b":duplicate"
                buckets[duplicate_segment].append(
                    (lsn, duplicate, _checksum(lsn, duplicate))
                )

        segments = []
        for segment_id, records in enumerate(buckets):
            rng.shuffle(records)
            if records:
                bad_lsn = n + segment_id + 100
                bad_payload = b"torn"
                records.append((bad_lsn, bad_payload, _checksum(bad_lsn, bad_payload) ^ 1))
                tempting = b"after-torn"
                records.append((gap, tempting, _checksum(gap, tempting)))
            segments.append({"segment_id": segment_id, "records": tuple(records)})
        rng.shuffle(segments)
        return {"start_lsn": 1, "segments": tuple(segments)}

    def solve(self, problem):
        return _recover(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        try:
            return _same_materialized(proposed, _verify_recovery(problem))
        except (KeyError, TypeError, ValueError, OverflowError, IndexError):
            return False


TASK = DurableLogRecoveryTask()
