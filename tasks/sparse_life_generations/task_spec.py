"""Advance Conway's Life patterns on an unbounded plane by a fixed generation count.

Every problem mixes four regimes that reward different representations: dense
soups, small objects scattered across a huge area, slowly growing methuselahs,
and a glider gun. Packed bit rows win the first and lose the second; isolated
components and cycle skipping win the second and do nothing for the first.
See README.md.
"""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)

FAMILIES = ("soup", "scattered", "methuselah", "gun")

GLIDER = ((1, 0), (2, 1), (0, 2), (1, 2), (2, 2))
OBJECTS = (
    GLIDER,
    ((0, 0), (1, 0), (2, 0)),                                           # blinker
    ((0, 0), (1, 0), (0, 1), (1, 1)),                                   # block
    ((1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)),                   # toad
    ((0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)),                   # beacon
    ((1, 0), (4, 0), (0, 1), (0, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3)),  # spaceship
    tuple((x, 0) for x in range(10)),                                   # becomes a period-15 oscillator
)
METHUSELAHS = (
    ((1, 0), (2, 0), (0, 1), (1, 1), (1, 2)),                           # R-pentomino
    ((1, 0), (3, 1), (0, 2), (1, 2), (4, 2), (5, 2), (6, 2)),           # acorn
    ((6, 0), (0, 1), (1, 1), (1, 2), (5, 2), (6, 2), (7, 2)),           # diehard
)
GUN = (
    (24, 0), (22, 1), (24, 1), (12, 2), (13, 2), (20, 2), (21, 2), (34, 2), (35, 2),
    (11, 3), (15, 3), (20, 3), (21, 3), (34, 3), (35, 3), (0, 4), (1, 4), (10, 4),
    (16, 4), (20, 4), (21, 4), (0, 5), (1, 5), (10, 5), (14, 5), (16, 5), (17, 5),
    (22, 5), (24, 5), (10, 6), (16, 6), (24, 6), (11, 7), (15, 7), (12, 8), (13, 8),
)


def _placed(rng, cells, origin):
    """Apply one of the eight grid symmetries, then translate."""
    flip, turns = rng.random() < 0.5, rng.randrange(4)
    result = []
    for x, y in cells:
        if flip:
            x = -x
        for _ in range(turns):
            x, y = -y, x
        result.append((x + origin[0], y + origin[1]))
    return result


def _soup(rng, width, height, density):
    return [(x, y) for x in range(width) for y in range(height) if rng.random() < density]


def _pattern(rng, family):
    origin = (rng.randrange(-100000, 100001), rng.randrange(-100000, 100001))
    if family == "soup":
        generations = rng.randrange(32, 65)
        cells = _placed(rng, _soup(rng, rng.randrange(40, 65), rng.randrange(40, 65),
                                   rng.uniform(0.25, 0.5)), origin)
    elif family == "scattered":
        generations = rng.randrange(120, 281)
        # Influence spreads one cell per generation, so objects more than
        # 2 * generations apart can never interact within the run.
        spacing = 2 * generations + 64
        cells = set()
        columns = rng.randrange(4, 7)
        for index in range(rng.randrange(14, 25)):
            corner = (origin[0] + (index % columns) * spacing + rng.randrange(24),
                      origin[1] + (index // columns) * spacing + rng.randrange(24))
            if index % 5 == 4:
                # A small soup interacts with itself unpredictably; a solver
                # cannot treat every component as a known object.
                cells.update(_placed(rng, _soup(rng, 7, 7, 0.45), corner))
            else:
                cells.update(_placed(rng, rng.choice(OBJECTS), corner))
                if rng.random() < 0.25:
                    # A second object inside the same light cone may collide.
                    near = (corner[0] + rng.randrange(-14, 15), corner[1] + rng.randrange(-14, 15))
                    cells.update(_placed(rng, rng.choice(OBJECTS), near))
        cells = sorted(cells)
    elif family == "methuselah":
        generations = rng.randrange(300, 601)
        seed = rng.choice(METHUSELAHS) if rng.random() < 0.75 else _soup(rng, 6, 6, 0.4)
        cells = _placed(rng, seed, origin)
    else:
        generations = rng.randrange(240, 401)
        cells = _placed(rng, GUN, origin)
    cells = sorted(set(cells))
    rng.shuffle(cells)
    return {"cells": [list(cell) for cell in cells], "generations": generations}


class SparseLifeGenerationsTask:
    name = "sparse_life_generations"
    task_version = "1.0.0"
    display_name = TASK_CATALOG[name].display_name
    uses_reference_output = True
    default_n = 8
    grading_cases = (8, 16)

    def generate_problem(self, n=8, random_seed=0):
        """``n`` patterns, cycling through all four families."""
        if type(n) is not int or n < 4:
            raise ValueError("n must be an integer of at least 4 patterns")
        rng = random.Random(random_seed)
        patterns = [_pattern(rng, FAMILIES[index % len(FAMILIES)]) for index in range(n)]
        rng.shuffle(patterns)
        return {"patterns": patterns}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed, *, reference_output=None):
        if (type(proposed) is not dict or set(proposed) != {"patterns"}
                or not plain_containers(proposed["patterns"])
                or type(proposed["patterns"]) is not list
                or len(proposed["patterns"]) != len(problem["patterns"])):
            return False
        for cells in proposed["patterns"]:
            if (type(cells) is not list
                    or any(type(cell) is not list or len(cell) != 2
                           or type(cell[0]) is not int or type(cell[1]) is not int
                           for cell in cells)
                    or any(first >= second for first, second in zip(cells, cells[1:]))):
                return False
        # Direct checks can compute the existing reference; grading reuses its
        # measured output and never reruns a solver for verification.
        if reference_output is None:
            reference_output = self.solve(problem)
        return proposed == reference_output


TASK = SparseLifeGenerationsTask()
