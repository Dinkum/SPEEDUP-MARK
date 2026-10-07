# Adaptive Query Engine

Execute a compact relational plan representation over in-memory integer tables:
filters, equality joins, grouped aggregates, top-k selection and projection.
There is no SQL text, no server, no disk format and no networking. What is timed
is planning, index construction and execution of a small engine — all inside the
candidate call.

## Input and submission

`problem["families"]` is a tuple of families. Each contains `tables`
(name → `{"columns": (names…), "rows": (row tuples…)}) and `queries`: plan trees
built from

- `("scan", table)`
- `("filter", input, column, op, value)` with `op` in `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- `("join", left, right, left_column, right_column)` — inner equality join,
  producing the left row followed by the right row
- `("group", input, key_columns, aggregations)` with aggregations of the form
  `("sum" | "min" | "max", column)` or `("count", None)`
- `("topk", input, column, k)`
- `("project", input, columns)`

The five workload families anchor selective filters, permissive filters, skew, top-k, and shared subplans. Each input varies join orientation, predicate nesting, aggregate functions and order, projection order, top-k limits, and two to four additional operator compositions per family. Table and source-column names are randomized. Shared subtrees remain shared after transformation. The grammar is bounded, with no SQL parsing or arbitrary SQL support.

A column name resolves to its first occurrence in the input schema; equality joins preserve both copies of a duplicated join-key column.

Implement `candidate.py::solve(problem)` and return, per family,
a tuple of answers, one per query. An answer is a tuple of row tuples sorted
ascending with duplicates preserved. `topk` selects the `k` rows under the total
order (*column* descending, then the whole row tuple ascending) and returns the
survivors ascending, so the boundary between rank `k` and rank `k+1` is decided
exactly even when values tie.

Answers and rows must use built-in tuples or lists; row values must be exact Python `int` values.

Library policy: this task is about building the engine. An embedded relational
engine (`sqlite3`, `duckdb`, `polars`, `pandas`, `pandasql`, `sqlalchemy`, `apsw`,
`datafusion`, `pyarrow`, `sqlglot`) is forbidden. Dictionaries, heaps, sorting
and `itertools` are allowed.

## Scoring and verification

The harness times `solve` and `candidate_solve` in host milliseconds and verifies
with a separate sort/merge relational interpreter. It uses merge joins and sorted group reduction rather than the reference hash joins and incremental accumulators. Every part of a candidate's work — planning, index construction,
materializing intermediates — is inside the timed call.

## Reference and verification

`_evaluate_node` walks each tree exactly as written: it materializes every scan,
hashes the *written* left input of every join and probes with the right, applies
each filter where the tree puts it, groups with a dictionary, and sorts fully
before slicing for `topk`.

The families put pressure on different decisions. Selective filters reward early
filtering; permissive filters pass nearly every row. Bounded top-k queries offer
an alternative to fully sorting all rows, while repeated subplans offer reuse.
Candidates can also choose join build sides, estimate intermediate sizes,
reorder joins, or aggregate early where the join key permits it. The relative
benefit depends on the generated workload and the machine running it.

The task implementation is original SPEEDUP-MARK code.

## Workload distribution

- **Size:** n scales table cardinalities (minimum 32).
- **Selection:** Every input contains selective_filter, permissive_filter, skew, topk, and shared_subplan.
- **Randomized:** Table values and keys, table/column names, join orientation, predicate thresholds and nesting, aggregate functions/order, projection order, top-k limits, query order, and additional filter/group/top-k compositions. Each family retains its table-size ratios and value ranges.
- **Fixed structure:** Five family anchors and the six-operator grammar are fixed. `_vary_plans` transforms and extends the anchor trees; this is a bounded grammar distribution, not arbitrary SQL. Family order stays fixed.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark adaptive_query_engine
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
