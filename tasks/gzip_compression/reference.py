"""Reference implementation for gzip_compression; copied into fresh run candidates."""

import zlib


def compress(data):
    stream = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=31)
    return stream.compress(data) + stream.flush()


def solve(problem):
    return {"compressed_data": compress(problem["plaintext"])}
