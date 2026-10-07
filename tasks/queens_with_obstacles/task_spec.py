"""Independent stdlib adaptation of the AlgoTune task; see README for scope."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)


def has_larger_placement(board, count):
    """Prove optimality through independent line-segment constraint search."""
    squares = [(r, c) for r in range(len(board)) for c in range(len(board[0]))
               if not board[r][c]]
    index = {square: i for i, square in enumerate(squares)}
    orientations = []
    conflicts = [0] * len(squares)
    # An obstacle splits a line into segments. A legal placement uses each
    # segment at most once, independently of the baseline's attack rays.
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        segments = []
        for r, c in squares:
            if (r - dr, c - dc) in index:
                continue
            segment = 0
            rr, cc = r, c
            while (rr, cc) in index:
                segment |= 1 << index[rr, cc]
                rr, cc = rr + dr, cc + dc
            segments.append(segment)
            remaining = segment
            while remaining:
                bit = remaining & -remaining
                remaining -= bit
                conflicts[bit.bit_length() - 1] |= segment
        orientations.append(segments)

    def search(available, needed):
        if needed == 0:
            return True
        if available.bit_count() < needed:
            return False
        smallest_partition = None
        for segments in orientations:
            domains = [segment & available for segment in segments if segment & available]
            # Each orientation partitions the remaining cells; its number of
            # nonempty segments bounds how many more queens can be placed.
            if len(domains) < needed:
                return False
            if smallest_partition is None or len(domains) < len(smallest_partition):
                smallest_partition = domains
        pivot = min(smallest_partition, key=int.bit_count)
        remaining = pivot
        while remaining:
            bit = remaining & -remaining
            remaining -= bit
            if search(available & ~conflicts[bit.bit_length() - 1], needed - 1):
                return True
        # Exhaustive alternatives: one queen at a cell in this segment,
        # or no queen in the segment at all.
        return search(available & ~pivot, needed)

    return search((1 << len(squares)) - 1, count + 1)


class Task:
    name = "queens_with_obstacles"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 10
    grading_cases = (10, 12)

    def generate_problem(self, n=10, random_seed=0):
        if n < 1:
            raise ValueError("n must be positive")
        rng = random.Random(random_seed)
        width = max(1, n - random_seed % 2)
        density = (0.15, 0.3, 0.45, 0.6)[random_seed % 4]
        board = [[rng.random() < density for _ in range(width)] for _ in range(n)]
        return {"obstacles": board}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"queens"}:
            return False
        board, queens = problem["obstacles"], proposed["queens"]
        if (type(queens) is not list
                or any(type(q) is not list or len(q) != 2
                       or any(type(x) is not int for x in q) for q in queens)):
            return False
        if (queens != sorted(queens) or len({tuple(q) for q in queens}) != len(queens)
                or any(not (0 <= r < len(board) and 0 <= c < len(board[0])) or board[r][c] for r, c in queens)):
            return False
        # Direct pairwise line-of-sight validation is independent of bit masks.
        for i, (r, c) in enumerate(queens):
            for rr, cc in queens[i + 1:]:
                if r != rr and c != cc and abs(r - rr) != abs(c - cc):
                    continue
                dr, dc = (rr > r) - (rr < r), (cc > c) - (cc < c)
                at_r, at_c = r + dr, c + dc
                while (at_r, at_c) != (rr, cc) and not board[at_r][at_c]:
                    at_r, at_c = at_r + dr, at_c + dc
                if (at_r, at_c) == (rr, cc):
                    return False
        return not has_larger_placement(board, len(queens))


TASK = Task()
