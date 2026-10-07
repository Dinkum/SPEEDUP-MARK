"""Reference implementation for multi_pattern_matching; copied into fresh run candidates."""

from __future__ import annotations

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


def _check_patterns(patterns):
    """Require positive minimum match length for every complete pattern."""
    for pattern in patterns:
        if _min_length(pattern) < 1:
            raise ValueError("whole patterns must have positive minimum match length")


def solve(problem):
    return _scan(problem)
