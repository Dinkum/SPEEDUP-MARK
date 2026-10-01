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

Filters are generated *above* the joins and joins in a fixed left-deep order, so
pushing a predicate into a scan, choosing which side to hash, and reordering a
three-way join are real decisions rather than decoration.

Implement `candidate.py::solve(problem, reference_solve)` and return, per family,
a tuple of answers, one per query. An answer is a tuple of row tuples sorted
ascending with duplicates preserved. `topk` selects the `k` rows under the total
order (*column* descending, then the whole row tuple ascending) and returns the
survivors ascending, so the boundary between rank `k` and rank `k+1` is decided
exactly even when values tie.

Submissions must be plain tuples or lists of exact integers (no container subclasses): a lazy sequence could otherwise do its work during untimed verification.

Library policy: this task is about building the engine. An embedded relational
engine (`sqlite3`, `duckdb`, `polars`, `pandas`, `pandasql`, `sqlalchemy`, `apsw`,
`datafusion`, `pyarrow`, `sqlglot`) is not an optimization here. `speedupmark.task
.forbidden_imports` combines the candidate's module attributes, forbidden names
inside its code objects, the roots `watch_imports` observes during the call, and
the `sys.modules` delta — the last two matter because `from sqlite3 import
connect` binds a function whose `__module__` is the C extension `_sqlite3`, and
because a function-local import binds nothing at module level. A submission that
trips the check fails verification. Dictionaries, heaps, sorting and `itertools`
are primitives and are allowed. The check is good-faith, not a sandbox.

## Scoring and verification

The harness times `solve` and `candidate_solve` in host milliseconds and verifies
with a re-derived answer, so candidate-reported timing and correctness are never
trusted. Every part of a candidate's work — planning, index construction,
materializing intermediates — is inside the timed call.

## Reference and families

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

This is an independently authored SPEEDUP-MARK task, not an AlgoTune port.

```console
python3 -m speedupmark adaptive_query_engine
```

Inside a managed run, use `python3 grade.py`. Edit only `candidate.py`.

## Workload distribution

- **Size:** n scales table cardinalities (minimum 32).
- **Selection:** Every input contains selective_filter, permissive_filter, skew, topk, and shared_subplan.
- **Randomized:** Customer, order, item, line-item, and event values and join keys; each family uses its own table-size ratios and value ranges in the generator helpers.
- **Fixed structure:** The query trees, predicates, top-k limits, and family order are fixed; rows vary. This is a distribution over data for these query templates, not arbitrary query programs.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
