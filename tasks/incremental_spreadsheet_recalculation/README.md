# Incremental Spreadsheet Recalculation

The first `input_count` cells are mutable integers. Every later cell is an acyclic formula over earlier cells: integer addition, subtraction, or multiplication by a small integer constant. Apply the ordered `set` and `query` stream and return each queried value. Sets target input cells only. Python's exact integer arithmetic defines overflow behavior.

The problem's `initial_inputs` field supplies the input-cell values. The first `input_count` entries in `formulas` are `None`; subsequent entries are `("add", left, right)`, `("sub", left, right)`, or `("scale", left, constant)`. Cell operands are integer indexes, while the third field of `scale` is the multiplier itself. Operations are `("set", input_cell, value)` and `("query", cell)`; queries may target input or formula cells. Return a list or tuple containing one exact Python `int` per query, in order. Booleans are rejected.

The reference deliberately recalculates the entire sheet for every query. Candidates can build reverse dependencies, mark dirty cells, evaluate only queried ancestors, cache clean values, or batch invalidation. All edit handling and recalculation is timed. Some generated dependencies reach back to early input cells; scale factors stay small so values remain practical.

Return a plain list or tuple, not a subclass that could compute answers during untimed verification. Direct grading uses 650 cells; managed grading uses both 650 and 1300 cells. Scores are reference time divided by candidate time in host milliseconds. Problem generation and verification are outside the timer; all candidate preparation is inside it.

Edit `candidate.py` and run `python3 -m speedupmark incremental_spreadsheet_recalculation`.

## Workload distribution

- **Size:** n is total cell count; max(1,n//6) cells are inputs.
- **Selection:** One formula-DAG distribution combining local dependencies, input fan-out, and repeated updates.
- **Randomized:** Initial/input values -100..100, earlier-cell dependencies, add/subtract/scale formulas, query cells; scale factors are -2..2.
- **Fixed structure:** There are max(40,n) operations: every third is an update, and every ninth is deliberately a no-op update. Formula-selection rules also depend on cell index.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
