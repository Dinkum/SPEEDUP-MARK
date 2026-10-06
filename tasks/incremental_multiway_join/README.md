# Incremental Weighted Multiway Join

## Input and submission

Maintain the exact weighted triangle aggregate `sum(R[a,b] * S[b,c] * T[c,a])` over all triples `(a,b,c)`. Missing relation entries have weight zero. The three coordinate domains are distinct even when their integer identifiers are equal. Signed integer weights may cancel; answers must use exact Python integers.

The input is `{"scenarios": (...)}`. Each independent scenario contains a descriptive `family`, initial `relations` mapping `R`, `S`, and `T` to sequences of unique `(source, target, weight)` rows, and an `operations` sequence. Operations are `("set", relation, source, target, weight)`, `("delete", relation, source, target)`, and `("query",)`. A set replaces the previous weight; setting zero is equivalent to deletion; deleting a missing pair is a no-op. `R` coordinates are `(a,b)`, `S` coordinates `(b,c)`, and `T` coordinates `(c,a)`. Return one list or tuple per scenario containing the aggregate at every query, in order. The outer result may also be a built-in list or tuple. Booleans and floating-point values are not valid answers.

Every generated problem contains uniform, hub-skewed, and dense/bursty scenarios. Sizes 1,200 and 2,400 each run all families; `n` controls each scenario's operation count and coordinate-domain size. Updates include replacements, insertions, deletions, signed weights, and zero weights. Dense scenarios alternate update-heavy and query-heavy phases, including a phase concentrated on one relation.

Return built-in lists or tuples, including each scenario's answers. Scores are reference time divided by candidate time in host milliseconds.

Maintaining the join is the task. Existing relational engines (`sqlite3`, `_sqlite3`, `duckdb`, `_duckdb`, `sqlalchemy`, `apsw`, `polars`, `pandas`, `datafusion`, `pyarrow`, `sqlglot`, `pandasql`) are forbidden imports. Dictionaries, heaps, and sorting remain available.

## Reference and verification

The reference builds both directions of each relation and maintains the aggregate with a first-order delta: changing one edge intersects its two neighboring adjacency maps. It is already incremental and indexed. The independent checker reconstructs the aggregate by enumerating two-edge paths at each query. Opportunities include maintaining selected two-relation products, choosing intermediates around skew, handling high-degree vertices separately, batching changes between queries, and switching strategies as update/query ratios change. Each strategy has initialization, maintenance, and memory costs; all candidate input-dependent preparation and operations are timed. There is no separate enforced memory budget.

## Workload distribution

- **Size:** n is operation count per scenario; key-domain side is max(8,2*isqrt(n)).
- **Selection:** Every input contains uniform, hub_skew, and dense_bursts.
- **Randomized:** Weighted R/S/T edges and updates. Hub endpoints are concentrated with probability 0.8; updates choose existing edges with probability 0.65 and deletion with probability 0.23.
- **Fixed structure:** Query intervals are 9/23 for uniform/hubs, and alternate 97/3 for dense bursts. Every scenario begins and ends with a query; dense phases include one-relation bursts.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark incremental_multiway_join
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
