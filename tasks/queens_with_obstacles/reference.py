"""Reference implementation for queens_with_obstacles; copied into fresh run candidates."""

def compatibility(board):
    height, width = len(board), len(board[0])
    squares = [(r, c) for r in range(height) for c in range(width) if not board[r][c]]
    index = {square: i for i, square in enumerate(squares)}
    masks = []
    for i, (r, c) in enumerate(squares):
        attacks = 1 << i
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            rr, cc = r + dr, c + dc
            while 0 <= rr < height and 0 <= cc < width and not board[rr][cc]:
                attacks |= 1 << index[(rr, cc)]
                rr, cc = rr + dr, cc + dc
        masks.append(((1 << len(squares)) - 1) ^ attacks)
    return squares, masks


def maximum_clique(neighbors):
    best = []
    def visit(chosen, available):
        nonlocal best
        # Greedy independent color classes give an upper bound on any
        # compatible clique remaining in this branch.
        order, bounds, remaining, color = [], [], available, 0
        while remaining:
            color += 1
            independent = remaining
            while independent:
                bit = independent & -independent
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                remaining ^= bit
                independent &= ~bit & ~neighbors[v]
        for position in range(len(order) - 1, -1, -1):
            if len(chosen) + bounds[position] <= len(best):
                return
            v = order[position]
            compatible = available & neighbors[v]
            if compatible:
                visit(chosen + [v], compatible)
            elif len(chosen) + 1 > len(best):
                best = chosen + [v]
            available &= ~(1 << v)
    visit([], (1 << len(neighbors)) - 1)
    return best


def solve(problem):
    squares, neighbors = compatibility(problem["obstacles"])
    return {"queens": [list(square) for square in sorted(squares[i] for i in maximum_clique(neighbors))]}
