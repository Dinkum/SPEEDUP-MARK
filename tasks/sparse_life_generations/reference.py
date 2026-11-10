"""Reference implementation for sparse_life_generations; copied into fresh run candidates."""

NEIGHBORS = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


def _step(live):
    """One generation: count live neighbors of every cell next to a live cell."""
    counts = {}
    for x, y in live:
        for dx, dy in NEIGHBORS:
            cell = (x + dx, y + dy)
            counts[cell] = counts.get(cell, 0) + 1
    return {cell for cell, count in counts.items()
            if count == 3 or (count == 2 and cell in live)}


def _advance(pattern):
    live = {(x, y) for x, y in pattern["cells"]}
    for _ in range(pattern["generations"]):
        live = _step(live)
    return [[x, y] for x, y in sorted(live)]


def solve(problem):
    return {"patterns": [_advance(pattern) for pattern in problem["patterns"]]}
