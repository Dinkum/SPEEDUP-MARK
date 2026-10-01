# Matrix Multiplication

Given `{"A": [[int,...],...], "B": [[int,...],...]}` with compatible nonempty rectangular shapes, return the complete matrix product as nested lists of exact integers. No modulo reduction or floating tolerance is permitted. Defaults are size parameters 45 and 75; seeds choose square, wide-inner and wide-output shapes plus dense/sparse signed coefficients. The baseline computes scalar dot products; a separately organized row-accumulation verifier checks every entry. Unlike upstream's NumPy floating arrays, this stdlib adaptation uses exact integer arithmetic and includes all conversion costs.

Optimize `candidate.py:solve(problem, reference_solve)`. Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

Provenance: independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/matrix_multiplication/description.txt` (https://github.com/ScalingIntelligence/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract below are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.

## Workload distribution

- **Size:** n scales (rows,inner,columns).
- **Selection:** seed % 3 selects (n,n,n), (max(1,n//3),2*n,n), or (n,max(1,n//2),2*n); seed % 4 == 0 selects sparse values. Joint cycle: 12.
- **Randomized:** Entries are integers -128..128; sparse mode first sets each entry to zero with probability 0.9.
- **Fixed structure:** Shape and sparsity choices follow the joint cycle, rather than independent family draws.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
