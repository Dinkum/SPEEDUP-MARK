# Compiled Streaming Pattern Matching

Compile a collection of patterns, then report which patterns each byte stream
matches. Patterns arrive as parsed structures, streams arrive as chunks, and
matches are allowed to span a chunk boundary. The answer is one integer bitmask
per stream, so a wrong "no match" or a wrong "match" fails the task.

## Input and submission

`problem["families"]` is a tuple of families, each with `patterns` and `streams`
(a tuple of tuples of byte chunks). Patterns use

- `("lit", bytes)`, `("class", frozenset of byte values)`
- `("cat", p1, p2, …)`, `("alt", p1, p2, …)`
- `("rep", p, low, high)` with `high = None` for unbounded repetition

Every complete pattern must have a minimum match length of at least one byte.
Subpatterns may match the empty string. For example, optional `a` followed by
mandatory `b` is legal, but optional `a` alone is not. The reference rejects
whole patterns that can match the empty string with `ValueError`.

Implement `candidate.py::solve(problem, reference_solve)` and return, per
family, one integer per stream: bit `i` set when pattern `i` matches anywhere in
that stream. A pattern matches when some non-empty substring of the stream
belongs to the pattern's language; non-matching positives and negatives are both
verified.
Empty streams match no legal pattern.

Submissions must be plain tuples or lists of exact integers (no container subclasses): a lazy sequence could otherwise do its work during untimed verification.

Library policy: building the matcher is the task. An existing regex engine
(`re`, `_sre`, `sre_compile`, `sre_parse`, `regex`, `hyperscan`, `re2`, `pyre2`,
`rure`, `oniguruma`) is a contract violation rather than an optimization.
`speedupmark.task.forbidden_imports` combines four signals, because each one alone
has a hole: the candidate's module attributes, import instructions inside its code
objects (a function-local `import re` binds nothing at module level), the roots
`speedupmark.task.watch_imports` sees imported during the call (`re` is already in
`sys.modules` at interpreter start, so a delta cannot see it), and the
`sys.modules` delta since the call began. `bytes.find`, sets, dictionaries and
heaps are primitives and are allowed. The check is good-faith, not a sandbox.

String literals, variable names, and attribute names alone are not import evidence.

## Scoring and verification

Wall-clock milliseconds for the whole call, so compilation and scanning are both
timed; the families vary how much stream data one compiled rule set is reused
for. Compiling one enormous combined automaton is not free, and the verifier
tests the exact bitmask.

## Reference and families

`solve` compiles one Thompson automaton per pattern and scans every byte of every
stream with an unanchored start state. It never prefilters with a literal, never
shares work between patterns, and never chooses a representation per pattern
family.

| Family | Regime | What it rewards |
| --- | --- | --- |
| `literal_heavy` | Long mandatory literals | Literal scanning instead of automaton execution |
| `automata_heavy` | Classes and repetitions, at least six atoms, no useful literal | Precomputed transitions, shared automata, bit-parallel state sets |
| `many_patterns_few_streams` | 22 patterns, little data | Compile cost dominates; a heavy combined automaton is the wrong purchase |
| `few_patterns_many_streams` | A few patterns, eight streams | Scan cost dominates; compile once and amortize |
| `boundary_crossing` | Every witness sits across a chunk boundary | Streaming across chunks rather than restarting per chunk |

The two amortization families pull in opposite directions, which is the intended
conflict: the same representation is not the right purchase for both. A
submission that matches each chunk independently is genuinely wrong on
`boundary_crossing` — the tests assert that the resetting answer differs from the
correct one on that family — and every generated stream misses at least one
pattern, so a constant "everything matched" answer is never correct.

Independent verification of the oracle: the repository's test suite translates
the pattern IR into bytes regexes and checks the task's own matcher against
Python's `re` engine for every `(pattern, stream)` pair of a generated problem.
That cross-check found two real automaton bugs during development.

```console
python3 -m speedupmark compiled_streaming_pattern_matching
```

Inside a managed run, use `python3 grade.py`. Edit only `candidate.py`.

## Workload distribution

- **Size:** n scales stream length (minimum 64).
- **Selection:** Every input contains literal_heavy, automata_heavy, many_patterns_few_streams, few_patterns_many_streams, and boundary_crossing.
- **Randomized:** Patterns, alphabets, streams, and chunk boundaries within family-specific ranges; generated streams are retried up to 16 times to avoid uninformative all-pattern matches.
- **Fixed structure:** Family-specific pattern/stream counts and compile-versus-scan ratios are fixed functions of n. See _family and its builders for rejection and construction rules.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
