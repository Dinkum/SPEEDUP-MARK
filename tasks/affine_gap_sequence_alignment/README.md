# Exact Affine Gap Sequence Alignment

## Input and submission

Find the minimum cost of globally aligning each pair of byte sequences. All bytes in both sequences must be consumed, including leading and trailing gaps. Equal byte pairs cost zero; unequal pairs cost `mismatch`. A maximal gap run of length `k >= 1` costs `gap_open + (k - 1) * gap_extend`. Insertion and deletion use the same gap costs. Consecutive gaps in opposite directions are legal and start separate runs; a column containing two gaps is not legal.

Input is `{"alignments": ((left, right, mismatch, gap_open, gap_extend), ...)}`. Sequences are exact built in `bytes`, including arbitrary byte values and empty sequences. All three penalties are positive Python integers; `gap_open` may be smaller than `gap_extend`. Return one exact nonnegative Python integer minimum cost per pair, in order, in a built-in list or tuple. Booleans and floats are invalid. No alignment traceback is required.

Every generated input contains nearly identical sequences, sequences with long inserted/deleted blocks, repetitive shifted sequences, and unrelated sequences with unequal lengths. This makes short edit distance shortcuts compete with methods that handle dense differences, long gaps, and repeated content. Exactness is required in every regime; a fixed narrow band or heuristic alignment that misses the optimum fails.

Only the Python standard library is required. Implement the alignment algorithm yourself: `Bio`, `edlib`, `parasail`, `pywfa`, `wavefront`, `Levenshtein`, `rapidfuzz`, and `skbio` are forbidden engine imports. Standard-library containers, heaps, byte operations, and integer arithmetic remain available. Do not install dependencies or modify benchmark files.

## Reference and verification

The reference uses conventional three state, rolling row dynamic programming over every alignment cell. Verification independently finds a shortest path through the three state edit graph using A* with a consistent remaining length lower bound. Small exhaustive path enumeration checks both implementations, including empty sequences, expensive mismatches, and opening costs below extension costs. The checker never calls the reference. All input dependent candidate preparation and alignment work is timed in host milliseconds.

## Workload distribution

- **Size:** n is the left sequence length, from 1 through 512. Direct grading uses 256; managed grading uses 256 and 384.
- **Selection:** Every input includes all four regimes in the order nearly identical, long gaps, repetitive shifts, and unrelated unequal lengths.
- **Randomized:** DNA byte values, substitution positions, inserted/deleted blocks, repeat motifs, phase shifts, right sequence lengths, and positive mismatch/open/extension penalties vary with the seed.
- **Fixed structure:** Generated sequences use `A`, `C`, `G`, and `T`. Matches cost zero, scoring is global, and gap run costs follow the same affine rule in every family. The legal contract accepts arbitrary bytes, not just generated DNA values.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark affine_gap_sequence_alignment
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

This is an independently authored SPEEDUP-MARK contract for the standard affine gap alignment problem; no upstream implementation was copied. [Wavefront alignment research](https://pmc.ncbi.nlm.nih.gov/articles/PMC8355039/) illustrates exact alternatives to conventional dynamic programming. SPEEDUP-MARK's generator, gap cost convention, allowed penalties, and verification define this task's results.
