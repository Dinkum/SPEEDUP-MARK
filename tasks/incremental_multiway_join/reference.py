"""Reference implementation for incremental_multiway_join; copied into fresh run candidates."""

from __future__ import annotations

def _dot(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(key, 0) for key, weight in left.items())


def _indexed_join(problem):
    results = []
    for scenario in problem["scenarios"]:
        outgoing = {name: {} for name in ("R", "S", "T")}
        incoming = {name: {} for name in ("R", "S", "T")}

        def store(name, source, target, weight):
            forward = outgoing[name].setdefault(source, {})
            reverse = incoming[name].setdefault(target, {})
            if weight == 0:
                forward.pop(target, None)
                reverse.pop(source, None)
            else:
                forward[target] = weight
                reverse[source] = weight

        for name, rows in scenario["relations"].items():
            for source, target, weight in rows:
                store(name, source, target, weight)
        total = sum(
            weight * _dot(outgoing["S"].get(b, {}), incoming["T"].get(a, {}))
            for a, row in outgoing["R"].items()
            for b, weight in row.items()
        )
        answers = []
        for operation in scenario["operations"]:
            if operation[0] == "query":
                answers.append(total)
                continue
            kind, name, source, target, *value = operation
            weight = value[0] if kind == "set" else 0
            previous = outgoing[name].get(source, {}).get(target, 0)
            if name == "R":
                coefficient = _dot(outgoing["S"].get(target, {}), incoming["T"].get(source, {}))
            elif name == "S":
                coefficient = _dot(incoming["R"].get(source, {}), outgoing["T"].get(target, {}))
            else:
                coefficient = _dot(incoming["S"].get(source, {}), outgoing["R"].get(target, {}))
            # The changed relation occurs once in each triangle, so this delta
            # remains exact for replacement, signed weights, and deletion.
            total += (weight - previous) * coefficient
            store(name, source, target, weight)
        results.append(tuple(answers))
    return tuple(results)


def solve(problem):
    return _indexed_join(problem)
