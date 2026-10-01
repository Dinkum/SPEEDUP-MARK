"""Apply simultaneous literal replacements across byte-stream chunks."""

from __future__ import annotations

import random
from speedupmark.task import load_candidate


_candidate = load_candidate(__file__)


def _replace(problem):
    data = b"".join(problem["chunks"])
    patterns = problem["replacements"]
    output = bytearray()
    position = 0
    while position < len(data):
        matches = [
            (len(needle), -precedence, replacement)
            for precedence, (needle, replacement) in enumerate(patterns)
            if data.startswith(needle, position)
        ]
        if matches:
            length, _, replacement = max(matches)
            output.extend(replacement)
            position += length
        else:
            output.append(data[position])
            position += 1
    return bytes(output)


class StreamingLiteralReplacementTask:
    name = "streaming_literal_replacement"
    task_version = "1.1.1"
    display_name = "Chunked Simultaneous Literal Replacement"
    default_n = 8000
    grading_cases = (8000, 16000)

    def generate_problem(self, n=8000, random_seed=0):
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(random_seed)
        motifs = (b"alpha", b"alphabet", b"beta", b"\x00\xff", b"aaaa", b"aba")
        data = bytearray()
        while len(data) < n:
            if rng.random() < 0.7:
                data.extend(rng.choice(motifs))
            else:
                data.extend(rng.randrange(256) for _ in range(rng.randrange(1, 8)))
        data = bytes(data[:n])
        replacements = [
            (b"alphabet", b"A"),
            (b"alpha", b"X"),
            (b"aba", b"Z"),
            (b"ab", b"Q"),
            (b"aaaa", b"aa"),
            (b"aa", b"B"),
            (b"\x00\xff", b"binary"),
        ]
        target_patterns = min(128, max(32, n // 125))
        while len(replacements) < target_patterns:
            length = rng.randrange(2, 9)
            if data and rng.random() < 0.75:
                start = rng.randrange(max(1, len(data) - length + 1))
                needle = data[start : start + length]
            else:
                needle = bytes(rng.randrange(256) for _ in range(length))
            if needle not in {item[0] for item in replacements}:
                replacement = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 6)))
                replacements.append((needle, replacement))
        chunks = []
        position = 0
        while position < len(data):
            width = rng.randrange(1, 19)
            chunks.append(data[position : position + width])
            position += width
        return {"chunks": tuple(chunks), "replacements": tuple(replacements)}

    def solve(self, problem):
        return _replace(problem)

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def is_solution(self, problem, proposed):
        try:
            return type(proposed) is bytes and proposed == _replace(problem)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = StreamingLiteralReplacementTask()
