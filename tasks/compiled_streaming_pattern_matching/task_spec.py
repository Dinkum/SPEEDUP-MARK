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
import sys

from speedupmark.task import forbidden_imports, load_candidate, watch_imports


_candidate = load_candidate(__file__)

# Handing the whole job to an existing regex engine is a contract violation:
# building the matcher is the task. Byte-search primitives, sets and heaps are
# primitives, not engines.
ENGINE_ROOTS = ("re", "_sre", "sre_compile", "sre_parse", "regex", "hyperscan",
                "pyre2", "rure", "re2", "oniguruma")
MAX_STATES = 400
BYTE_RANGE = range(256)


def _min_length(pattern):
    """Shortest string the pattern can match.

    Legal whole patterns require at least one byte; subpatterns may match the
    empty string, as long as the complete pattern cannot.
    """
    kind = pattern[0]
    if kind == "lit":
        return len(pattern[1])
    if kind == "class":
        return 1
    if kind == "cat":
        return sum(_min_length(child) for child in pattern[1:])
    if kind == "alt":
        return min(_min_length(child) for child in pattern[1:])
    if kind == "rep":
        return pattern[2] * _min_length(pattern[1])
    raise ValueError(f"unsupported pattern node {kind!r}")


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


def _compile(pattern):
    """Thompson construction with an unanchored scan prefix.

    Returns ``(transitions, epsilon, start, accepting)``. The start state loops
    on every byte, so one simulation pass covers every start position instead of
    restarting the automaton once per byte. Bounded repetitions are expanded,
    which keeps the automaton counter-free and makes a bit-parallel state set a
    legal implementation choice.
    """
    # The scan prefix consumes arbitrary bytes, so acceptance must require a
    # byte from the pattern itself. Nullable subpatterns remain legal.
    _check_patterns((pattern,))
    transitions = {}
    epsilon = {}
    accepting = set()

    def new_state():
        state = len(transitions)
        if state >= MAX_STATES:
            raise ValueError("pattern compiles to too many states")
        transitions[state] = {}
        epsilon[state] = []
        return state

    def step(state, byte, target):
        transitions[state].setdefault(byte, []).append(target)

    def link(source, target):
        epsilon[source].append(target)

    def build(node):
        """Return ``(entry, exit)`` states for a node."""
        kind = node[0]
        if kind == "lit":
            entry = exit_state = new_state()
            for byte in node[1]:
                following = new_state()
                step(exit_state, byte, following)
                exit_state = following
            return entry, exit_state
        if kind == "class":
            # One byte is consumed between distinct entry and exit states: a
            # self-looping state would make the exit reachable without matching
            # its byte, which is a false accept waiting to happen.
            entry, exit_state = new_state(), new_state()
            for byte in sorted(node[1]):
                step(entry, byte, exit_state)
            return entry, exit_state
        if kind == "cat":
            entry = exit_state = None
            for child in node[1:]:
                child_entry, child_exit = build(child)
                if entry is None:
                    entry = child_entry
                else:
                    link(exit_state, child_entry)
                exit_state = child_exit
            return entry, exit_state
        if kind == "alt":
            entry, exit_state = new_state(), new_state()
            for child in node[1:]:
                child_entry, child_exit = build(child)
                link(entry, child_entry)
                link(child_exit, exit_state)
            return entry, exit_state
        if kind == "rep":
            # Explicit join states: after the mandatory copies, and after every
            # optional copy, the repetition may stop. Without them a pattern
            # like ``x{1,2}`` could only accept the maximum count.
            _, child, low, high = node
            entry, exit_state = new_state(), new_state()
            cursor = entry
            for _ in range(low):
                child_entry, child_exit = build(child)
                link(cursor, child_entry)
                cursor = child_exit
            link(cursor, exit_state)
            if high is None:
                loop = new_state()
                link(cursor, loop)
                link(loop, exit_state)
                child_entry, child_exit = build(child)
                link(loop, child_entry)
                link(child_exit, loop)
            else:
                for _ in range(high - low):
                    child_entry, child_exit = build(child)
                    link(cursor, child_entry)
                    link(child_exit, exit_state)
                    cursor = child_exit
            return entry, exit_state
        raise ValueError(f"unsupported pattern node {kind!r}")

    entry, exit_state = build(pattern)
    accepting.add(exit_state)
    start = new_state()
    for byte in BYTE_RANGE:
        step(start, byte, start)
    link(start, entry)
    return transitions, epsilon, start, accepting


def _closure(states, epsilon):
    seen = set()
    stack = list(states)
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        stack.extend(epsilon[state])
    return seen


def _matches(data, program):
    """True when some non-empty substring of ``data`` is in the pattern's language."""
    transitions, epsilon, start, accepting = program
    current = _closure((start,), epsilon)
    for byte in data:
        following = set()
        for state in current:
            targets = transitions[state].get(byte)
            if targets:
                following.update(targets)
        if not following:
            return False
        current = _closure(following, epsilon)
        if current & accepting:
            return True
    return False


def _scan(problem):
    """Reference answer: one compiled automaton per pattern, one pass per stream."""
    answers = []
    for family in problem["families"]:
        programs = [_compile(pattern) for pattern in family["patterns"]]
        family_answers = []
        for chunks in family["streams"]:
            data = b"".join(chunks)
            mask = 0
            for index, program in enumerate(programs):
                if _matches(data, program):
                    mask |= 1 << index
            family_answers.append(mask)
        answers.append(tuple(family_answers))
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


def _check_patterns(patterns):
    """Require positive minimum match length for every complete pattern."""
    for pattern in patterns:
        if _min_length(pattern) < 1:
            raise ValueError("whole patterns must have positive minimum match length")


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
    name = "compiled_streaming_pattern_matching"
    task_version = "2.0.0"
    display_name = "Compiled Streaming Pattern Matching"
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

    def solve(self, problem):
        return _scan(problem)

    def candidate_solve(self, problem):
        self.policy_violations = ()
        loaded = set(sys.modules)
        with watch_imports(ENGINE_ROOTS) as imported_during:
            result = _candidate.solve(problem, self.solve)
        violations = forbidden_imports(_candidate, ENGINE_ROOTS, loaded, imported_during)
        if violations:
            # An invalid submission must not be scored as a fast one; the value
            # returned here cannot be a valid answer, so only the candidate fails.
            self.policy_violations = violations
            print(f"candidate uses forbidden engine imports: {', '.join(violations)}")
            return None
        return result

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
            return _same_materialized(proposed, _scan(problem))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            return False


TASK = CompiledStreamingPatternMatchingTask()
