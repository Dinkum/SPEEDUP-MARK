"""Which input patterns expose each single stuck-at gate fault?"""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


def _mix_block(kind, left, right, control, add):
    if kind == "adder":
        result = []
        for a, b in zip(left, right):
            different = add("xor", a, b)
            result.append(add("xor", different, control))
            control = add("or", add("and", a, b), add("and", different, control))
        return result[1:] + [control]
    if kind == "compare":
        # From least to most significant bit, the latest unequal bit decides.
        for a, b in zip(left, right):
            control = add("xor", control, add("and", add("xor", control, a), add("xor", a, b)))
        kind = "select"
    if kind == "select":
        return [add("xor", a, add("and", control, add("xor", a, b)))
                for a, b in zip(left, right)]
    result = []
    for a, b in zip(left, right):
        control = add("xor", control, add("xor", a, b))
        result.append(control)
    return result


def _reorder(rng, first, gates, outputs):
    """Randomize independent gate IDs while keeping operands before consumers."""
    users = [[] for _ in gates]
    pending = [0] * len(gates)
    ready = []
    for i, (_, a, b) in enumerate(gates):
        for source in (a, b):
            if source >= first:
                users[source - first].append(i)
                pending[i] += 1
        if not pending[i]:
            ready.append(i)
    renamed = list(range(first)) + [0] * len(gates)
    reordered = []
    while ready:
        index = rng.randrange(len(ready))
        ready[index], ready[-1] = ready[-1], ready[index]
        old = ready.pop()
        op, a, b = gates[old]
        renamed[first + old] = first + len(reordered)
        reordered.append((op, renamed[a], renamed[b]))
        for child in users[old]:
            pending[child] -= 1
            if not pending[child]:
                ready.append(child)
    return reordered, [renamed[output] for output in outputs]


class CircuitFaultSimulation:
    name = "circuit_fault_simulation"
    display_name = TASK_CATALOG[name].display_name
    task_version = "1.0.0"
    default_n = 2400
    grading_cases = (2400, 4200)
    family_period = 3
    uses_reference_output = True

    def generate_problem(self, n=2400, random_seed=0):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive gate count")
        rng = random.Random(random_seed)
        width = (64, 256, 512)[random_seed % 3]
        lanes = rng.randrange(4, 9)
        first = 8 * lanes
        wires = list(range(4 * lanes))
        gates, taps = [], []

        def add(op, a, b):
            wire = first + len(gates)
            gates.append((op, a, b))
            return wire

        while len(gates) < n:
            rng.shuffle(wires)
            kinds = ["adder", "compare", "select", "parity"]
            rng.shuffle(kinds)
            following = []
            for block, kind in enumerate(kinds):
                left = wires[block * lanes:(block + 1) * lanes]
                right = [rng.choice(wires) if rng.randrange(3) else rng.randrange(first)
                         for _ in left]
                following.extend(_mix_block(kind, left, right, rng.choice(wires), add))
            taps.append(rng.randrange(first, first + len(gates)))
            wires = following
        gates, outputs = _reorder(rng, first, gates, rng.sample(wires, lanes) + taps)
        return {"pattern_count": width, "inputs": [rng.getrandbits(width) for _ in range(first)],
                "gates": gates, "outputs": sorted(set(outputs))}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed, *, reference_output=None):
        if type(proposed) not in (list, tuple) or len(proposed) != len(problem["gates"]):
            return False
        if any(type(row) not in (list, tuple) or len(row) != 2
               or any(type(value) is not int for value in row) for row in proposed):
            return False
        if reference_output is None:
            reference_output = self.solve(problem)
        return all(tuple(row) == tuple(expected) for row, expected in zip(proposed, reference_output))


TASK = CircuitFaultSimulation()
