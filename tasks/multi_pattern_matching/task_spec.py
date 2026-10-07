"""Compile a pattern set, then report which patterns each chunked stream matches.

Patterns arrive as parsed structures -- literals, character classes,
concatenation, alternation and bounded or unbounded repetition -- and streams
arrive as chunks, with matches allowed to span a chunk boundary. The submission
returns one integer bitmask per stream, so the answer is exact: a wrong
"no match" or a wrong "match" is a failed task, not a near miss.

The reference compiles one Thompson automaton per pattern and scans every byte
of every stream, with no literal prefiltering, no sharing between patterns and
no representation choice. What stays open is the architecture real multi-pattern
engines use: literal factors that reject positions before an automaton runs,
automata combined across patterns, bit-parallel state sets, and different
representations per pattern family. Compilation is timed, and the families vary
how many streams reuse one compiled rule set.
"""

from __future__ import annotations

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_check_patterns = _reference._check_patterns
_compile = _reference._compile
_matches = _reference._matches


# Handing the whole job to an existing regex engine is a contract violation:
# building the matcher is the task. Byte-search primitives, sets and heaps are
# primitives, not engines.
ENGINE_ROOTS = ("re", "_sre", "sre_compile", "sre_parse", "regex", "hyperscan",
                "pyre2", "rure", "re2", "oniguruma")


def _witness(pattern, rng):
    """One concrete byte string the pattern matches (used to seed streams)."""
    kind = pattern[0]
    if kind == "lit":
        return pattern[1]
    if kind == "class":
        return bytes([rng.choice(sorted(pattern[1]))])
    if kind == "cat":
        return b"".join(_witness(child, rng) for child in pattern[1:])
    if kind == "alt":
        return _witness(rng.choice(pattern[1:]), rng)
    if kind == "rep":
        _, child, low, high = pattern
        ceiling = low + 2 if high is None else high
        return b"".join(_witness(child, rng) for _ in range(rng.randrange(low, ceiling + 1)))
    raise ValueError(f"unsupported pattern node {kind!r}")


def _verify_patterns(problem):
    """Independent byte-regex oracle, used only outside candidate timing.

    The engine restriction applies to submissions; the verifier deliberately
    does not reuse Thompson construction or state transitions.
    """
    import re
    def expression(node):
        kind = node[0]
        if kind == "lit":
            return re.escape(node[1])
        if kind == "class":
            # Hex escapes keep every byte literal, including ], -, and \.
            return b"[" + b"".join(f"\\x{byte:02x}".encode() for byte in sorted(node[1])) + b"]"
        if kind in ("cat", "alt"):
            separator = b"" if kind == "cat" else b"|"
            return b"(?:" + separator.join(expression(child) for child in node[1:]) + b")"
        if kind == "rep":
            low, high = node[2:]
            quantifier = f"{{{low},{'' if high is None else high}}}".encode()
            return b"(?:" + expression(node[1]) + b")" + quantifier
        raise ValueError(kind)
    answers = []
    for family in problem["families"]:
        patterns = [re.compile(expression(pattern)) for pattern in family["patterns"]]
        if any(pattern.match(b"") is not None for pattern in patterns):
            raise ValueError("whole patterns must have positive minimum match length")
        masks = []
        for chunks in family["streams"]:
            data = b"".join(chunks)
            masks.append(sum(1 << i for i, pattern in enumerate(patterns) if pattern.search(data)))
        answers.append(tuple(masks))
    return tuple(answers)


def _materialized(value):
    """Plain containers only: reject subclasses that defer work until verification."""
    if type(value) is int:
        return True
    if type(value) in (tuple, list):
        return all(_materialized(item) for item in value)
    return False


def _same_materialized(actual, expected):
    if isinstance(expected, tuple):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(_same_materialized(a, e) for a, e in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _informative(patterns, chunked):
    """False when any stream matches every pattern.

    A stream whose answer is "everything matched" is a free win for a constant
    answer, and it means the noise accidentally satisfied every pattern -- a sign
    the family is saturated and worth no measurement. Accidental matches are
    probabilistic, so the family is regenerated rather than rejected.
    """
    programs = [_compile(pattern) for pattern in patterns]
    full = (1 << len(patterns)) - 1
    for chunks in chunked:
        data = b"".join(chunks)
        mask = 0
        for index, program in enumerate(programs):
            if _matches(data, program):
                mask |= 1 << index
        if mask == full:
            return False
    return True


def _chunks(data, boundaries):
    """Split ``data`` at the given offsets; a boundary may fall inside a match."""
    pieces = []
    start = 0
    for stop in sorted(boundaries):
        if start < stop <= len(data):
            pieces.append(data[start:stop])
            start = stop
    pieces.append(data[start:])
    return tuple(piece for piece in pieces if piece)


def _walking_boundaries(rng, data, patterns):
    """Chunk sizes that ignore where matches happen to sit."""
    boundaries = []
    cursor = rng.randrange(7, 33)
    while cursor < len(data):
        boundaries.append(cursor)
        cursor += rng.randrange(7, 64)
    return boundaries


def _seam_boundaries(rng, data, patterns):
    """Chunk boundaries placed in the middle of each witness occurrence."""
    boundaries = []
    for pattern in patterns:
        witness = _witness(pattern, rng)
        if len(witness) > 1:
            found = data.find(witness)
            if found >= 0:
                boundaries.append(found + len(witness) // 2)
    return boundaries or [max(1, len(data) // 2)]


class CompiledStreamingPatternMatchingTask:
    forbidden_import_roots = ENGINE_ROOTS
    name = "multi_pattern_matching"
    task_version = "3.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 2048
    grading_cases = (2048, 3072)

    def generate_problem(self, n=2048, random_seed=0):
        if n < 64:
            raise ValueError("n must be at least 64")
        rng = random.Random(random_seed)
        families = (
            self._literal_heavy(n, rng),
            self._automata_heavy(n, rng),
            self._many_patterns_few_streams(n, rng),
            self._few_patterns_many_streams(n, rng),
            self._boundary_crossing(n, rng),
        )
        return {"families": families}

    def _noise(self, rng, size, alphabet):
        return bytes(rng.choice(alphabet) for _ in range(max(0, size)))

    def _payloads(self, rng, patterns, streams, size, alphabet, injection):
        """Noise plus injected witnesses, some of them deliberately corrupted."""
        payloads = []
        for _ in range(streams):
            pieces = []
            for pattern in patterns:
                # Patterns left out of a stream are the misses the answer needs;
                # a witness for one pattern is already a near miss for another,
                # so no separate corruption step is required.
                if rng.random() < injection:
                    pieces.append(_witness(pattern, rng))
            span = max(1, size // (len(pieces) + streams + 1))
            data = bytearray()
            for piece in pieces:
                data.extend(self._noise(rng, rng.randrange(1, span + 1), alphabet))
                data.extend(piece)
            data.extend(self._noise(rng, size - len(data), alphabet))
            payloads.append(bytes(data[:size]))
        return payloads

    def _family(self, rng, name, patterns, streams, size, alphabet,
                injection=0.5, boundaries=_walking_boundaries):
        _check_patterns(patterns)
        # An all-matching stream would make a constant answer correct, so keep
        # drawing noise until every stream has at least one missing pattern.
        for _ in range(16):
            payloads = self._payloads(rng, patterns, streams, size, alphabet, injection)
            chunked = [_chunks(data, boundaries(rng, data, patterns)) for data in payloads]
            if _informative(patterns, chunked):
                return {"family": name, "patterns": tuple(patterns),
                        "streams": tuple(chunked)}
        raise ValueError(f"{name}: could not generate informative streams")

    def _literal_heavy(self, n, rng):
        words = (b"gadget", b"harmless", b"headroom", b"feedback", b"magenta")
        patterns = []
        for index in range(8):
            first = rng.choice(words)
            second = rng.choice(words)
            if index % 4 == 3:
                patterns.append(("cat", ("lit", first), ("lit", b"-"), ("lit", second)))
            else:
                patterns.append(("cat", ("lit", first), ("lit", second)))
        return self._family(rng, "literal_heavy", patterns, 4, max(64, n // 4), b"abcdefgh")

    def _automata_heavy(self, n, rng):
        # No mandatory literal for a byte search to lock onto, and every atom
        # covers at most a quarter of the stream alphabet, so an accidental
        # match needs six specific letters in a row (about 1 in 4096 positions).
        # Wide classes over the whole alphabet would instead match everywhere in
        # letter noise and the family's answer would be "everything matched".
        alphabet = b"abcdefghijklmnop"
        groups = (frozenset(b"abcd"), frozenset(b"efgh"), frozenset(b"ijkl"),
                  frozenset(b"mnop"))
        narrow = (frozenset(b"ab"), frozenset(b"cd"), frozenset(b"ef"), frozenset(b"gh"))
        patterns = []
        for index in range(8):
            first, second, third = (groups[index % 4], groups[(index + 1) % 4],
                                    groups[(index + 2) % 4])
            if index % 4 == 0:
                patterns.append(("cat", ("class", first), ("rep", ("class", second), 3, 3),
                                 ("class", third), ("class", first),
                                 ("rep", ("class", groups[(index + 3) % 4]), 2, 2)))
            elif index % 4 == 1:
                patterns.append(("cat", ("rep", ("class", first), 3, 3), ("class", second),
                                 ("class", third), ("rep", ("class", groups[(index + 3) % 4]), 2, 2),
                                 ("class", second)))
            elif index % 4 == 2:
                patterns.append(("cat", ("class", first),
                                 ("rep", ("alt", ("class", narrow[index % 4]),
                                          ("class", narrow[(index + 1) % 4])), 3, 3),
                                 ("class", third), ("class", first),
                                 ("rep", ("class", groups[(index + 3) % 4]), 2, 2)))
            else:
                patterns.append(("cat", ("class", narrow[index % 4]), ("class", first),
                                 ("rep", ("class", second), 3, 3), ("class", third),
                                 ("rep", ("class", groups[(index + 3) % 4]), 2, 2)))
        return self._family(rng, "automata_heavy", patterns, 4, max(64, n // 4), alphabet)

    def _many_patterns_few_streams(self, n, rng):
        alphabet = b"abcdefghi"
        patterns = []
        for index in range(16):
            head = b"tok%d" % index
            patterns.append(("cat", ("lit", head), ("class", frozenset(alphabet))))
            if index % 3 == 0:
                patterns.append(("cat", ("lit", head),
                                 ("rep", ("class", frozenset(alphabet)), 1, 2)))
        return self._family(rng, "many_patterns_few_streams", patterns, 2,
                            max(64, n // 2), alphabet)

    def _few_patterns_many_streams(self, n, rng):
        alphabet = b"abcdefgh"
        patterns = (
            ("cat", ("lit", b"delta"), ("class", frozenset(b"abc"))),
            ("cat", ("lit", b"echo"), ("lit", b"-"),
             ("rep", ("class", frozenset(b"xyz")), 1, 2)),
            ("alt", ("lit", b"foxtrot"), ("lit", b"golf")),
            ("cat", ("class", frozenset(b"ab")), ("lit", b"hotel"),
             ("class", frozenset(b"cd"))),
        )
        return self._family(rng, "few_patterns_many_streams", patterns, 8,
                            max(48, n // 8), alphabet)

    def _boundary_crossing(self, n, rng):
        patterns = tuple(("cat", ("lit", b"seam%d" % index), ("lit", b"edge"))
                         for index in range(6))
        return self._family(rng, "boundary_crossing", patterns, 4,
                            max(64, n // 4), b"abcdefgh",
                            injection=0.45, boundaries=_seam_boundaries)

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        try:
            if not _materialized(proposed):
                return False
            if type(proposed) not in (tuple, list) or len(proposed) != len(problem["families"]):
                return False
            for family_answer, family in zip(proposed, problem["families"]):
                if type(family_answer) not in (tuple, list):
                    return False
                if len(family_answer) != len(family["streams"]):
                    return False
                for mask in family_answer:
                    if type(mask) is not int or mask < 0:
                        return False
            return _same_materialized(proposed, _verify_patterns(problem))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            return False


TASK = CompiledStreamingPatternMatchingTask()
