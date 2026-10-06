# Exact Minimum Weight Assignment

## Input and submission

Input `{"costs": [[integer, ...], ...]}` contains a dense square cost matrix. Return `{"assignment": [column_for_row_0, ...]}`, a permutation minimizing the exact sum of selected costs. Any optimum is accepted. Unlike upstream's sparse CSR floating-point representation, this adaptation uses signed integer dense matrices with uniform, highly tied and correlated seeded families. Defaults are 70 and 110 rows.

Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The baseline uses Hungarian augmentation; an independent negative exchange-cycle test verifies global optimality, not just matching feasibility.

## Workload distribution

- **Size:** n is square cost-matrix dimension.
- **Selection:** seed % 3 selects unstructured, strongly tied, or perturbed additive costs.
- **Randomized:** Unstructured entries are -500..500. Other families add row/column potentials -100..99 and independent noise 0..2 or 0..99.
- **Fixed structure:** Every instance is a dense square assignment problem; family choice controls ties and separability.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark min_weight_assignment
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/min_weight_assignment/description.txt` (https://github.com/oripress/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.
