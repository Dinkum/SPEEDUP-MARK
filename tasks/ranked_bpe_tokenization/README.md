# Ranked Byte-Pair Tokenization

## Input and submission

Start with one token per input byte, using IDs 0 through 255. Merge-table rank `r` contains one ordered token pair and creates token ID `256 + r`. Pair members may be byte IDs or token IDs created by earlier ranks. Repeatedly choose the lowest-ranked pair currently present, then merge every non-overlapping occurrence of that one pair from left to right. Stop when no ranked pair remains. This rule applies to arbitrary bytes, including NUL and `0xff`, and completely specifies overlap and rank ties.

Input is `{"data": bytes, "merges": ((left_id, right_id), ...)}`. The position of a pair in `merges` is its rank, and `n` controls the length of `data` in generated problems.

If a pair appears at multiple ranks, its earliest rank wins and determines the emitted token ID.

Return the final tokens as a built-in list or tuple of nonnegative exact Python `int` values.

Direct grading uses 6,000 input bytes. Managed grading uses 6,000 and 12,000. Each timing sample sums two independent inputs from the same merge family. Scores compare reference and candidate times in host milliseconds. Candidate preparation is timed.

Implement the ranked merge rule yourself. Existing tokenizer engines (`tokenizers`, `tiktoken`, `sentencepiece`, `transformers`, `subword_nmt`, `youtokentome`) are forbidden imports. Ordinary Python containers, heaps, and byte operations are allowed.

## Reference and verification

The grader compares timed outputs with the reference. `_verify_encoding` independently checks a small seeded input using a linked-token queue that selects the lowest rank and then the leftmost occurrence.

The reference repeatedly scans the full token list to find the next rank and rebuilds it to apply merges. Candidates can use linked neighbors, occurrence indexes, heaps with stale-entry checks, or compact arrays. All tokenization work is timed.

## Workload distribution

- **Size:** `n` is byte-string length. Merge-table size is `min(768, max(32, n // 12 + 16))`, giving 516 and 768 rules at the grading sizes.
- **Selection:** seed % 3 selects shallow byte-pair rules, deep chains, or balanced merge construction.
- **Randomized:** The generator samples 32–64 byte values and builds 12–96 motifs with shared prefixes and varied tails. It adds repeated-byte, alternating-pair, and overlapping motifs. Rules derive from actual motif tokenizations. Shallow rules use byte pairs, chains merge leftmost, and balanced rules merge adjacent tokens with the shortest combined expansion. Legal filler rules bring the table to its target size. Sampled data mutate one byte in a motif with probability 0.20 for shallow and 0.08 otherwise.
- **Fixed structure:** All three families obey the same ranked BPE semantics. Structural motif classes remain deliberate, but their bytes, lengths, tables, and data vary. Three consecutive grading seeds cover all three merge families.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark ranked_bpe_tokenization
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
