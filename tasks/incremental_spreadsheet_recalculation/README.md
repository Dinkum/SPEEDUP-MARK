# Incremental Spreadsheet Recalculation

## Input and submission

The first `input_count` cells are mutable integers. Every later cell is a formula over earlier cells. Apply the ordered `set` and `query` stream and return each queried value. Sets target input cells only. All arithmetic uses exact Python integers.

The problem's `initial_inputs` field supplies the input-cell values. The first `input_count` entries in `formulas` are `None`. Subsequent entries are `(op, left, right)`, where `op` is `add`, `sub`, `mul`, `min`, or `max`, and both operands are cell indexes. A `("scale", left, constant)` formula multiplies a cell by the given integer constant. Operations are `("set", input_cell, value)` and `("query", cell)`. Queries may target input or formula cells. Return one exact Python `int` per query, in order. Booleans are rejected.

Return a built-in list or tuple. Direct grading uses 2,400 cells. Managed grading uses 2,400 and 4,800 cells. Scores compare reference and candidate times in host milliseconds. Candidate preparation is timed.

## Reference and verification

The grader compares timed outputs with the reference. `_verify_sheet` independently checks a small seeded sheet with nonlinear formulas by evaluating queried ancestors with a DFS stack.

The reference deliberately recalculates the entire sheet for every query. Candidates can build reverse dependencies, mark dirty cells, evaluate only queried ancestors, cache clean values, or batch invalidation. All edit handling and recalculation is timed. Some generated dependencies reach back to early input cells; scale factors stay small so values remain practical.

## Workload distribution

- **Size:** n is total cell count; max(1,n//6) cells are inputs.
- **Selection:** One formula-DAG distribution combining local dependencies, input fan-out, and repeated updates.
- **Randomized:** Initial and edited input values range from -100 to 100. Formulas combine nearby cells and scattered earlier dependencies. Scale factors range from -2 to 2. Generated `mul` formulas use one input-cell factor to keep chained values practical.
- **Fixed structure:** The `max(40, n)` operations span four phases with update probabilities 0.10, 0.88, 0.20, and 0.65. Some phases reuse a small set of query cells or edited inputs. Twenty percent of edits deliberately repeat the current value.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark incremental_spreadsheet_recalculation
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
