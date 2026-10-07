"""Apply simultaneous literal replacements across byte-stream chunks."""

from __future__ import annotations

import random
from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


def _verify_replacement(problem):
    """Trie walk with longest-prefix selection, separate from pattern scans."""
    trie = {}
    for needle, replacement in problem["replacements"]:
        node = trie
        for byte in needle:
            node = node.setdefault(byte, {})
        node.setdefault(None, replacement)  # First rule wins duplicate needles.
    data = b"".join(problem["chunks"])
    output, position = bytearray(), 0
    while position < len(data):
        node, end, match = trie, position, None
        while end < len(data) and data[end] in node:
            node = node[data[end]]
            end += 1
            if None in node:
                match = end, node[None]
        if match is None:
            output.append(data[position])
            position += 1
        else:
            position, replacement = match
            output.extend(replacement)
    return bytes(output)

class StreamingLiteralReplacementTask:
    name = "multi_literal_replacement"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
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

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            return type(proposed) is bytes and proposed == _verify_replacement(problem)
        except (KeyError, TypeError, ValueError, IndexError):
            return False


TASK = StreamingLiteralReplacementTask()
