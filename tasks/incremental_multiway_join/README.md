# Incremental Weighted Multiway Join

## Input and submission

Maintain the exact weighted triangle aggregate `sum(R[a,b] * S[b,c] * T[c,a])` over all triples `(a,b,c)`. Missing relation entries have weight zero. The three coordinate domains are distinct even when their integer identifiers are equal. Signed integer weights may cancel; answers must use exact Python integers.

The input is `{"scenarios": (...)}`. Each independent scenario contains a descriptive `family`, initial `relations` mapping `R`, `S`, and `T` to sequences of unique `(source, target, weight)` rows, and an `operations` sequence. Operations are `("set", relation, source, target, weight)`, `("delete", relation, source, target)`, and `("query",)`. A set replaces the previous weight; setting zero is equivalent to deletion; deleting a missing pair is a no-op. `R` coordinates are `(a,b)`, `S` coordinates `(b,c)`, and `T` coordinates `(c,a)`. Return one list or tuple per scenario containing the aggregate at every query, in order. The outer result may also be a built-in list or tuple. Booleans and floating-point values are not valid answers.

Every generated problem contains uniform, hub-skewed, and dense-burst scenarios. Sizes 4,000 and 8,000 exercise all three families at domain sides of 504 and 712. Updates include replacements, insertions, deletions, signed weights, and zero weights.

Return built-in lists or tuples, including each scenario's answers. Scores are reference time divided by candidate time in host milliseconds.

Maintaining the join is the task. Existing relational engines (`sqlite3`, `_sqlite3`, `duckdb`, `_duckdb`, `sqlalchemy`, `apsw`, `polars`, `pandas`, `datafusion`, `pyarrow`, `sqlglot`, `pandasql`) are forbidden imports. Dictionaries, heaps, and sorting remain available.

## Reference and verification

The reference builds both directions of each relation and maintains the aggregate with a first-order delta. Changing one edge intersects its two neighboring adjacency maps. The grader compares timed outputs with the reference and checks a small seeded stream by independently enumerating two-edge paths. Candidates can maintain selected two-relation products, handle high-degree vertices separately, or combine repeated edits between queries. Candidate preparation and operation processing are both timed.

## Workload distribution

- **Size:** Each scenario has `n` operations plus an opening and closing query. Each key domain has side `max(8, 8 * isqrt(n))`.
- **Selection:** Every input contains uniform, hub_skew, and dense_bursts.
- **Randomized:** All relations start with sparse background edges. In each skewed scenario two relations also have high-degree rows with overlapping neighborhoods. The seed chooses the relation targeted by concentrated updates and permutes all three key domains.
- **Fixed structure:** Uniform and hub scenarios query every 9 and 11 operations. The hub scenario directs 95% of edits at edges joining busy rows. Dense bursts switch between query intervals of 113 and 3. Each 113-operation block allows edits to either supporting relation near its start and concentrates the remaining edits on up to six edges.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark incremental_multiway_join
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
