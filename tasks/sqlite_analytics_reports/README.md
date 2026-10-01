# SQLite Analytics Report Workload

For every call, return the three fully materialized reports: category aggregates, duplicate `(user_id, category)` groups, and per-user value totals above a threshold. SQL `NULL` behavior is binding: `COUNT(value)` and `COUNT(DISTINCT note)` exclude nulls, `SUM` returns null for an all-null group, and null categories group together. Every report has an explicit deterministic order; duplicate input rows remain duplicates.

Input is `rows`, a sequence of `(user_id, category, value, note)`, plus `minimum_total`. Return `(categories, duplicates, totals)`:

- Category rows are `(category, row_count, nonnull_value_count, value_sum, distinct_nonnull_note_count)`, sorted by descending sum (NULL sums last), then category ascending with NULL first among ties.
- Duplicate rows are `(user_id, category, copies)` for counts greater than one, sorted by copies descending, user ID ascending, then category ascending with NULL first.
- Total rows are `(user_id, value_sum)` with sum at least `minimum_total`, sorted by sum descending then user ID ascending. All-null user sums do not qualify.

The reference performs conventional schema setup, bulk insert, index creation, and SQL aggregation. Candidates may change schema, indexes, queries, and transaction strategy, or bypass SQLite and compute equivalent reports directly. Any setup a candidate chooses is inside the timed call. Every opened connection must be closed, including on failure.

Return plain built-in lists or tuples for the result, each report, and each row. Subclasses and lazy sequences are rejected before their callbacks can run; scalar leaves must have the exact expected types and values.

Edit `candidate.py` and run `python -m speedupmark.harness tasks/sqlite_analytics_reports`.

## Workload distribution

- **Size:** n is base row count, plus scheduled duplicates.
- **Selection:** One relational-report distribution with nulls, ties, and duplicates.
- **Randomized:** Keys from max(1,n//25 + 1) values; categories/notes from six choices; non-null values -20..100.
- **Fixed structure:** Every thirteenth value is null and every 41st row is duplicated. Report queries and minimum_total=450 are fixed.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
