"""Packed simulation with one-step inference for single-consumer gates."""

from operator import and_, or_, xor


_OPERATIONS = {"and": and_, "or": or_, "xor": xor}


def solve(problem):
    inputs = problem["inputs"]
    gates = problem["gates"]
    outputs = problem["outputs"]
    width = problem["pattern_count"]
    mask = (1 << width) - 1
    first = len(inputs)
    end = first + len(gates)

    live = set(outputs)
    consumers = [[] for _ in range(end)]
    for wire in reversed(range(first, end)):
        if wire in live:
            _, a, b = gates[wire - first]
            live.update((a, b))
            consumers[a].append(wire)
            if a != b:
                consumers[b].append(wire)

    normal = list(inputs) + [0] * len(gates)
    program = []
    observed = set(outputs)
    detects_flip = [0] * end
    unresolved = []
    links = []
    for wire, (op, a, b) in enumerate(gates, first):
        if wire not in live:
            continue
        evaluate = _OPERATIONS[op]
        normal[wire] = evaluate(normal[a], normal[b])
        if wire in observed:
            detects_flip[wire] = mask
        elif len(consumers[wire]) == 1:
            links.append((wire, consumers[wire][0]))
        else:
            unresolved.append(len(program))
        program.append((wire, evaluate, a, b))

    # One lane in a Python integer holds all patterns for one injected fault.
    # Flipping that gate covers both stuck values once split by its normal value.
    batch_size = 16
    if unresolved:
        broadcast = ((1 << (width * batch_size)) - 1) // mask
        packed_normal = [value * broadcast for value in normal]
        for start in range(0, len(unresolved), batch_size):
            positions = unresolved[start:start + batch_size]
            injected = {program[pos][0]: mask << (lane * width)
                        for lane, pos in enumerate(positions)}
            values = packed_normal.copy()
            for wire, evaluate, a, b in program[positions[0]:]:
                values[wire] = evaluate(values[a], values[b]) ^ injected.get(wire, 0)
            detected = 0
            for output in outputs:
                detected |= values[output] ^ packed_normal[output]
            for pos in positions:
                detects_flip[program[pos][0]] = detected & mask
                detected >>= width

    # A change with just one live consumer is visible exactly when it changes
    # that consumer and the resulting flip is visible downstream.
    for wire, child in reversed(links):
        op, a, b = gates[child - first]
        evaluate = _OPERATIONS[op]
        left = normal[a] ^ mask if a == wire else normal[a]
        right = normal[b] ^ mask if b == wire else normal[b]
        detects_flip[wire] = (evaluate(left, right) ^ normal[child]) & detects_flip[child]
    return [(detects_flip[i] & normal[i], detects_flip[i] & (mask ^ normal[i]))
            for i in range(first, end)]
