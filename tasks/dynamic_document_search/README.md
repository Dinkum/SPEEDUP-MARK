# Dynamic Ranked Document Search

## Input and submission

Apply a stream of `add`, `delete`, and `query` operations to an initially empty corpus. Adding an existing integer document ID replaces its previous text; deleting a missing ID is a no-op. Text is split on whitespace with no case folding. A query term may repeat. Its exact integer score is the sum, over distinct query terms, of `document_term_frequency * query_term_frequency`. Return only positive-score documents, ordered by descending score and then ascending document ID, truncated to the requested limit.

Operations are `("add", document_id, text)`, `("delete", document_id)`, and `("query", term_sequence, limit)`. Query terms are already split. Return one list or tuple of hits per query, in operation order; additions and deletions append nothing, while a query with no hits appends an empty sequence. Each hit is `(document_id, score)`, with both fields exact Python `int` values. The outer result, per-query results, and hit rows may be lists or tuples.

All answer containers must be built-in lists or tuples. Direct grading uses 800 operations; managed grading uses both 800 and 1600. Scores are reference time divided by candidate time in host milliseconds; candidate index construction is timed.

Building the search index is the task. Delegating to an existing SQL or search engine is forbidden: `sqlite3`, `_sqlite3`, `duckdb`, `_duckdb`, `sqlalchemy`, `apsw`, `polars`, `pandas`, `datafusion`, `pyarrow`, `whoosh`, and `tantivy` imports fail verification. Dictionaries, heaps, and sorting remain available.

## Reference and verification

The grader compares timed outputs with the reference. Each sample also runs an untimed stream with replacement, deletion, repeated terms, a tied ranking, and large document IDs. `_verify_search` checks that stream by counting terms in raw token lists.

The reference stores per-document term counts and scans every live document for every query. Candidates can maintain an inverted index, update postings incrementally, accumulate sparse scores, and select the top results without a full sort. The score includes every update and search in the timed stream.

## Workload distribution

- **Size:** n is exact operation count; normal grading uses 800 and 1600.
- **Selection:** The grading streams begin with 500 or 1,000 distinct documents. The remaining operations form four phases with query probabilities 0.85, 0.25, 0.90, and 0.45.
- **Randomized:** The 2,400- or 4,800-term vocabulary is assigned a seeded rank order. Document terms are sampled with Zipf weights proportional to inverse rank. Document lengths are 8–24 or 80–160. Long-document probabilities are 0.75 in phases two and four and 0.15 otherwise. Each query uses 2–6 terms, sometimes adds an absent term, and requests 3, 10, or 30 hits. Query terms come from the most common 5% with probabilities 0.15, 0.80, 0.85, and 0.25 across the phases. Other query terms come uniformly from the tail. Mutation steps choose deletion with probability 0.15. For remaining mutations, replacement has probability 0.65 when the corpus is nonempty.
- **Fixed structure:** Every normal-size stream spans different update/query pressures. Phase probabilities and monotonic IDs are prescribed; exact operation schedules, contents, and term frequencies vary. Tiny inputs have at least one warmup addition when nonempty.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark dynamic_document_search
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
