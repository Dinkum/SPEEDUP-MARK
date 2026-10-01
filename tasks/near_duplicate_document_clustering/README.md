# Exact Near-Duplicate Document Clustering

Given whitespace-tokenized, case-sensitive documents, form an undirected edge for every pair whose sets of contiguous `shingle_width` word shingles meet the exact rational Jaccard threshold. A nonempty document shorter than the width contributes one whole-document shingle. Empty documents always remain singleton components, including when compared with another empty document. Return one component label per input document; every label is the minimum input index in that connected component. Similarity is transitive only through connected components, and approximate candidate generation may not miss qualifying edges.

`threshold` is `(numerator, denominator)`. Shingle sets `A` and `B` qualify exactly when `len(A & B) * denominator >= len(A | B) * numerator`; for example, `(4, 5)` means Jaccard similarity of at least `4/5`. Return a list or tuple with exactly one exact Python `int` label per input document. Booleans are rejected.

The reference compares every pair and materializes shingle sets. Faster exact methods can cache compact representations, use length bounds or inverted indexes, and avoid comparisons that cannot reach the threshold. Generation, verification, and candidate import are outside the timer; all clustering work is inside `solve`.

Return a plain list or tuple, not a subclass that could defer clustering to untimed verification. Direct grading uses 360 documents; managed grading uses 360 and 720. The seed modulo three selects `(shingle_width, threshold)` from `(2, 1/2)`, `(3, 2/3)`, and `(4, 4/5)`. Three consecutive samples cover all three profiles. Documents vary from short to long, mix common boilerplate with distinct content, and include overlapping text windows near the threshold plus interior edits. Input order is shuffled and contains no group labels. These distributions stress exact candidate-pair filtering and connected-component closure; a cluster representative need not be similar to every member. The verifier independently builds integer shingle bitsets and traverses the resulting similarity graph. Scores are reference time divided by candidate time in host milliseconds.

Edit `candidate.py` and run `python3 -m speedupmark near_duplicate_document_clustering`.

This is an original lightweight adaptation inspired by Terminal-Bench 4's [`distributed-dedup`](https://github.com/harbor-framework/terminal-bench/blob/v4.0.0/tasks/distributed-dedup/instruction.md), not a faithful port. No source code was copied.

## Workload distribution

- **Size:** n is document count.
- **Selection:** seed % 3 selects shingle width/Jaccard threshold: (2,1/2), (3,2/3), or (4,4/5).
- **Randomized:** Groups of 3–8 shifted documents, lengths 24/48/96/192, optional 24-word shared boilerplate, occasional interior edits, and shuffled document order. Vocabulary size is max(1024,3*n).
- **Fixed structure:** Shift lengths deliberately create threshold-neighbor similarities. The legal input contract includes documents outside these generated templates.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
