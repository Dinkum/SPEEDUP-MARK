"""Reference implementation for multi_literal_replacement; copied into fresh run candidates."""

from __future__ import annotations

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


def solve(problem):
    return _replace(problem)
