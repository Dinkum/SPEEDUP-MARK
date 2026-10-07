"""Reference implementation for example_gzip; copied into fresh run candidates."""

import gzip


def solve(problem: bytes) -> bytes:
    return gzip.compress(problem)
