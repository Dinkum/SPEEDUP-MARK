# Queens With Obstacles

Input `{"obstacles": [[bool,...],...]}` describes a rectangular chessboard (`True` means blocked). Return `{"queens": [[row,column],...]}` in lexicographic order. Maximize the number of queens; blocked cells cannot hold queens and block attacks along rows, columns and both diagonals. Multiple queens can therefore occupy the same row when separated by obstacles. Any maximum valid placement is accepted. Defaults use size parameters 10 and 12, with square/rectangular boards and obstacle densities 15/30/45/60 percent. The baseline reduces compatibility to exact maximum clique with greedy-color branch bounds. The verifier directly checks every queen pair's line of sight, then independently searches for a larger placement using obstacle-separated line segments and segment-count branch bounds. It does not call the baseline attack-graph builder or maximum-clique search. Small exhaustive tests and a deliberately broken baseline regression validate this separation. Exact verification has input-dependent runtime; the declared sizes keep it practical for laptop grading. The original upstream objective is preserved, but boards and outputs use plain Python containers with bounded laptop-scale generators.

Optimize `candidate.py:solve(problem, reference_solve)`. Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

Provenance: independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/queens_with_obstacles/description.txt` (https://github.com/ScalingIntelligence/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract below are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.

## Workload distribution

- **Size:** n is board height; width is max(1,n - seed % 2).
- **Selection:** seed % 4 selects obstacle probability 0.15, 0.30, 0.45, or 0.60, coupled to board width.
- **Randomized:** Each square independently becomes an obstacle with the selected probability.
- **Fixed structure:** Board aspect ratios and obstacle-density choices follow the four-seed cycle.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
