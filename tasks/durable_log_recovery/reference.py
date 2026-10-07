"""Reference implementation for durable_log_recovery; copied into fresh run candidates."""

from __future__ import annotations

import zlib


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


def solve(problem):
    return _recover(problem)
