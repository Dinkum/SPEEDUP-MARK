"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_candidate

_candidate = load_candidate(__file__)

import zlib


def compress(data):
    stream = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=31)
    return stream.compress(data) + stream.flush()


class Task:
    name = "gzip_compression"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 100000
    grading_cases = (100000, 250000)

    def generate_problem(self, n=100000, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        mode = random_seed % 4
        if mode == 0:
            data = bytes(rng.randrange(256) for _ in range(n))
        elif mode == 1:
            words = [bytes(rng.randrange(97, 123) for _ in range(rng.randrange(3, 16))) for _ in range(100)]
            chunks, length = [], 0
            while length < n:
                chunk = words[int(rng.random() ** 3 * len(words))] + b" "
                chunks.append(chunk)
                length += len(chunk)
            data = b"".join(chunks)[:n]
        elif mode == 2:
            motif = bytes(rng.randrange(16) for _ in range(4096))
            data = (motif * (n // len(motif) + 1))[:n]
        else:
            chunks, length = [], 0
            while length < n:
                chunk = (bytes(rng.randrange(256) for _ in range(512)) if len(chunks) % 2
                         else b"event=read status=ok resource=record\n" * 16)
                chunks.append(chunk)
                length += len(chunk)
            data = b"".join(chunks)[:n]
        return {"plaintext": data}

    def solve(self, problem):
        return {"compressed_data": compress(problem["plaintext"])}

    def is_solution(self, problem, proposed):
        if (type(proposed) is not dict or set(proposed) != {"compressed_data"}
                or type(proposed["compressed_data"]) is not bytes):
            return False
        data, encoded = problem["plaintext"], proposed["compressed_data"]
        limit = (len(compress(data)) * 1001 + 999) // 1000
        if len(encoded) > limit:
            return False
        try:
            stream = zlib.decompressobj(wbits=31)
            decoded = stream.decompress(encoded, len(data) + 1)
            return decoded == data and stream.eof and not stream.unused_data and not stream.unconsumed_tail
        except zlib.error:
            return False

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)


TASK = Task()
