# Multi-Literal Replacement

## Input and submission

Treat all byte chunks as one logical stream, so a match may cross any number of chunk boundaries. At each input position, choose the longest matching nonempty needle. Equal-length ties use the earlier replacement-table entry. Emit its replacement and advance past the original needle; otherwise copy one byte. Replacements are simultaneous and nonrecursive: emitted bytes are never scanned again. Empty replacements are allowed, and inputs may contain arbitrary bytes.

All chunks are supplied together in `problem["chunks"]`. Matches must cross chunk boundaries correctly. Candidates may concatenate the chunks; bounded-memory processing is optional and memory usage is not graded.

Return an exact built-in `bytes` value. `bytearray`, other buffer types, text strings, and iterators are also rejected.

## Reference and verification

The verifier independently walks a pattern trie and selects the longest matching prefix, retaining the first rule for duplicate needles. The executable checker is `_verify_replacement` in `task_spec.py`.

The reference joins the stream and tests every pattern at every position. Candidates can stream with a bounded carry, index by first byte and length, build a trie, or use a multi-pattern automaton while preserving leftmost/longest precedence. Joining, matching, and output construction are timed.

## Workload distribution

- **Size:** n is byte-stream length.
- **Selection:** One overlapping-literal distribution with random chunk boundaries.
- **Randomized:** Chunks use a fixed motif with probability 0.7, otherwise 1–7 random bytes; a total of 32–128 patterns with target lengths 2–8, replacements of lengths 0–5, and stream chunks of lengths 1–18.
- **Fixed structure:** Seven overlap rules are fixed. Additional patterns are drawn from the input with probability 0.75, otherwise random bytes.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark multi_literal_replacement
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
