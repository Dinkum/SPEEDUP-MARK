# Ranked Byte-Pair Tokenization

Start with one token per input byte, using IDs 0 through 255. Merge-table rank `r` contains one ordered token pair and creates token ID `256 + r`. Pair members may be byte IDs or token IDs created by earlier ranks. Repeatedly choose the lowest-ranked pair currently present, then merge every non-overlapping occurrence of that one pair from left to right. Stop when no ranked pair remains. This rule applies to arbitrary bytes, including NUL and `0xff`, and completely specifies overlap and rank ties.

Input is `{"data": bytes, "merges": ((left_id, right_id), ...)}`. The position of a pair in `merges` is its rank, and `n` controls the length of `data` in generated problems.

If a pair appears at multiple ranks, its earliest rank wins and determines the emitted token ID.

Return the final tokens as a plain list or tuple of nonnegative exact Python `int` values. Booleans, lazy iterators, and list or tuple subclasses are rejected.

The reference repeatedly scans the full token list to find the next rank and rebuilds it to apply merges. Candidates can use linked neighbors, occurrence indexes, heaps with stale-entry checks, or compact arrays. All tokenization work is timed.

Direct grading uses 3000 input bytes; managed grading uses 3000 and 6000. Scores are reference time divided by candidate time in host milliseconds. Problem generation and verification are outside the timer; candidate preparation is inside it.

Implement the ranked merge rule yourself. Existing tokenizer engines (`tokenizers`, `tiktoken`, `sentencepiece`, `transformers`, `subword_nmt`, `youtokentome`) are forbidden imports. Ordinary Python containers, heaps, and byte operations are allowed. This is a good-faith policy check, not a sandbox.

Edit `candidate.py` and run `python3 -m speedupmark ranked_bpe_tokenization`.

## Workload distribution

- **Size:** n is byte-string length; merge-table target is min(128,max(32,n//20+16)).
- **Selection:** seed % 3 selects shallow byte-pair rules, deep chains, or balanced merge construction.
- **Randomized:** Sample 8–24 byte values; six random motifs of lengths 8–24 plus repeated-byte, alternating-pair, and overlapping motifs. Rules are built from the motifs' actual current tokenization: shallow uses byte-only pairs, chains merge leftmost, balanced merges adjacent tokens with shortest combined expansion. Additional legal rules fill the target. Data sample these motifs with a byte mutation probability of 0.20 for shallow or 0.08 otherwise.
- **Fixed structure:** All three families obey the same ranked BPE semantics. Structural motif classes remain deliberate, but their bytes, lengths, tables, and data vary. Three consecutive grading seeds cover all three merge families.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
