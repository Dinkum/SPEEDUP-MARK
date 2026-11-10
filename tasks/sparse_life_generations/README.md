# Sparse Life Generations

## Input and submission

Advance patterns of Conway's Game of Life on an unbounded plane. The input is `{"patterns": [{"cells": [[x, y], ...], "generations": g}, ...]}`. Each pattern lists its distinct live cells as integer coordinate pairs in arbitrary order; coordinates may be negative and far from the origin. Return `{"patterns": [[[x, y], ...], ...]}`: for each pattern, in input order, the live cells after exactly `g` generations, sorted ascending by `x` and then `y`, with no duplicates. A pattern that dies out returns an empty list.

One generation applies the standard rule to every cell of the plane at once, using its eight neighbors: a live cell with two or three live neighbors stays live, a dead cell with exactly three live neighbors becomes live, and every other cell is dead. Patterns are independent of each other.

Only Python standard-library dependencies are required. All output containers must be materialized built-in lists; integers must be exact `int`, never `bool`. All preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The starting reference keeps each pattern as a set of live cells and, every generation, counts the live neighbors of each cell adjacent to a live one. Grading checks the output's structure and ordering and compares it exactly with the reference's measured answer; no second simulation runs. The reference itself is tested against known still lifes, oscillators, glider displacement, and the glider gun's growth of five cells every thirty generations. Candidate improvements include faster neighbor counting, rows packed into integers and stepped with bitwise operations, splitting a pattern into components too far apart to interact within the run, detecting a component that repeats (possibly displaced) and skipping ahead, and memoizing repeated blocks.

## Workload distribution

- **Size:** n is the number of patterns; declared sizes are 8 and 16.
- **Selection:** Every input cycles through all four families, then shuffles: dense soups, scattered objects, methuselahs, and a glider gun.
- **Randomized:** Soup dimensions (40–64 per side), density (0.25–0.5) and 32–64 generations; 14–24 scattered objects per pattern with 120–280 generations; methuselah choice with 300–600 generations; gun runs of 240–400 generations; one of eight grid symmetries and a translation within ±100,000 for every placement; input cell order.
- **Fixed structure:** Scattered objects sit on a grid whose spacing exceeds twice the generation count, so separate sites cannot interact, but a site may hold two nearby objects that collide, and every fifth site is a small random soup. Scattered objects are gliders, blinkers, blocks, toads, beacons, lightweight spaceships, and a ten-cell row. Methuselahs are the R-pentomino, acorn, or diehard three times in four, otherwise a 6×6 random soup. The gun is the Gosper glider gun.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark sparse_life_generations
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored for SPEEDUP-MARK. The rule and the named patterns are long-published public knowledge; the generator, representations and verification contract are original to this repository, and no external implementation was copied.
