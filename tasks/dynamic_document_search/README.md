# Dynamic Ranked Document Search

Apply a stream of `add`, `delete`, and `query` operations to an initially empty corpus. Adding an existing integer document ID replaces its previous text; deleting a missing ID is a no-op. Text is split on whitespace with no case folding. A query term may repeat. Its exact integer score is the sum, over distinct query terms, of `document_term_frequency * query_term_frequency`. Return only positive-score documents, ordered by descending score and then ascending document ID, truncated to the requested limit.

Operations are `("add", document_id, text)`, `("delete", document_id)`, and `("query", term_sequence, limit)`. Query terms are already split. Return one list or tuple of hits per query, in operation order; additions and deletions append nothing, while a query with no hits appends an empty sequence. Each hit is `(document_id, score)`, with both fields exact Python `int` values. The outer result, per-query results, and hit rows may be lists or tuples.

The reference stores per-document term counts and scans every live document for every query. Candidates can maintain an inverted index, update postings incrementally, accumulate sparse scores, and select the top results without a full sort. All updates and searches are timed.

All answer containers must be plain lists or tuples, not subclasses that could do work during untimed verification. Direct grading uses 800 operations; managed grading uses both 800 and 1600. Scores are reference time divided by candidate time in host milliseconds; candidate index construction is timed.

Building the search index is the task. Delegating to an existing SQL or search engine is forbidden: `sqlite3`, `_sqlite3`, `duckdb`, `_duckdb`, `sqlalchemy`, `apsw`, `polars`, `pandas`, `datafusion`, `pyarrow`, `whoosh`, and `tantivy` imports fail verification. Dictionaries, heaps, and sorting remain available. The import check is a good-faith policy, not a sandbox.

Edit `candidate.py` and run `python3 -m speedupmark dynamic_document_search`.

## Workload distribution

- **Size:** n is exact operation count; normal grading uses 800 and 1600.
- **Selection:** Each stream begins with n//8 additions, then four equal-length phases with query probabilities 0.85, 0.25, 0.90, and 0.45.
- **Randomized:** Vocabulary size is max(160,min(768,n//3)), with eight sampled hot terms. Documents draw 65% of words from hot terms; lengths are 8–24 or 80–160, with long-document probability 0.75 in phases two/four and 0.15 otherwise. Queries draw hot terms with probabilities 0.15/0.80/0.85/0.25, use 2–6 terms plus an optional absent term, and request 3/10/30 hits. Mutation steps choose deletion with probability 0.15, otherwise replacement with probability 0.65 or insertion.
- **Fixed structure:** Every normal-size stream spans different update/query pressures. Phase probabilities and monotonic IDs are prescribed; exact operation schedules, contents, and term frequencies vary. Tiny inputs have at least one warmup addition when nonempty.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
