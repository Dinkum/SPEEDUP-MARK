"""Reference implementation for graph_coloring; copied into fresh run candidates."""

def _bits(n):
    return (1 << n) - 1


def _max_clique(adj):
    n = len(adj)
    best = 0

    def search(chosen, pool, excluded):
        nonlocal best
        if pool == 0 and excluded == 0:
            best = max(best, chosen.bit_count())
            return
        if chosen.bit_count() + pool.bit_count() <= best:
            return
        union = pool | excluded
        pivot = -1
        pivot_count = -1
        bit = 1
        index = 0
        while bit <= union:
            if union & bit:
                count = (adj[index] & pool).bit_count()
                if count > pivot_count:
                    pivot_count = count
                    pivot = index
            bit <<= 1
            index += 1
        candidates = pool & ~adj[pivot]
        bit = 1
        index = 0
        while bit <= candidates:
            if candidates & bit:
                search(chosen | bit, pool & adj[index], excluded & adj[index])
                pool &= ~bit
                excluded |= bit
            bit <<= 1
            index += 1

    search(0, _bits(n), 0)
    return best


def _greedy(adj):
    n = len(adj)
    order = sorted(range(n), key=lambda vertex: adj[vertex].bit_count(), reverse=True)
    colors = [0] * n
    for vertex in order:
        used = 0
        bit = 1
        index = 0
        neighbors = adj[vertex]
        while bit <= neighbors:
            if neighbors & bit and colors[index]:
                used |= 1 << (colors[index] - 1)
            bit <<= 1
            index += 1
        color = 1
        while used & (1 << (color - 1)):
            color += 1
        colors[vertex] = color
    return colors


def _minimum_coloring(adj):
    n = len(adj)
    if n == 0:
        return []
    upper = _greedy(adj)
    best_count = max(upper)
    lower = _max_clique(adj)
    if lower >= best_count:
        return upper
    best = upper
    colors = [0] * n
    uncolored = _bits(n)

    def search(remaining, used_count):
        nonlocal best, best_count
        if remaining == 0:
            if used_count < best_count:
                best_count = used_count
                best = colors[:]
            return
        if used_count >= best_count:
            return
        vertex = -1
        saturation = -1
        degree = -1
        bit = 1
        index = 0
        while bit <= remaining:
            if remaining & bit:
                colored_neighbors = adj[index] & ~remaining
                distinct = 0
                neighbor = colored_neighbors
                while neighbor:
                    low = neighbor & -neighbor
                    distinct |= 1 << (colors[low.bit_length() - 1] - 1)
                    neighbor ^= low
                score = distinct.bit_count()
                deg = adj[index].bit_count()
                if score > saturation or (score == saturation and deg > degree):
                    saturation = score
                    degree = deg
                    vertex = index
            bit <<= 1
            index += 1
        forbidden = 0
        neighbors = adj[vertex]
        bit = 1
        index = 0
        while bit <= neighbors:
            if bit & neighbors and colors[index]:
                forbidden |= 1 << (colors[index] - 1)
            bit <<= 1
            index += 1
        fresh = used_count + 1
        for color in range(1, min(fresh, best_count - 1) + 1):
            if forbidden & (1 << (color - 1)):
                continue
            if color > fresh:
                break
            colors[vertex] = color
            search(remaining ^ (1 << vertex), max(used_count, color))
            colors[vertex] = 0
            if color == fresh:
                break

    search(uncolored, 0)
    return best


def solve(problem):
    n = len(problem)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if problem[i][j]:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return _minimum_coloring(adj)
