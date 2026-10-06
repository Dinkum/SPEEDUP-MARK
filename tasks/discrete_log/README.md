# Prime-Field Discrete Logarithm

## Input and submission

Given a prime `p`, a primitive root `g`, and `h` in its multiplicative group, return exactly `{"x": integer}` where `0 <= x < p-1` and `pow(g, x, p) == h`. The exponent is unique in that interval. Zero is a valid answer when `h == 1`; booleans are not integers in the output contract.

`n` is the modulus bit length, from 12 through 62; managed cases use 50 and 60. The seeded generator chooses primes and primitive roots without external libraries. Group orders have bounded prime factors; their factorization is not supplied.

Standard library only. Return fully materialized output and do not modify benchmark files.

## Reference and verification

The reference factors p-1, uses Pohlig–Hellman prime-power lifting, solves each prime-order digit with baby-step/giant-step, and combines residues with CRT. Verification is one modular exponentiation. Smooth, medium-subgroup, and large-subgroup cases change the useful search strategy and table budget; candidates can use small-group tables, rho subgroup search, batched arithmetic, or different memory/time tradeoffs. All order factoring and tables are timed. The 50/60-bit sizes describe the modulus, not a generic 50/60-bit prime-order search. Index-calculus performance is not claimed.

## Workload distribution

- **Size:** n is modulus bit length, 12–62; managed cases are 50 and 60.
- **Selection:** seed % 3 selects smooth, medium_subgroup, or large_subgroup.
- **Randomized:** Odd group-order factors, the optional larger prime factor, the primitive root, and an exponent uniformly drawn from [0,p-1). Primes are selected by bounded rejection sampling.
- **Fixed structure:** The smooth order uses odd primes 3–31 and a power of two. Medium and large orders also include a prime of min(17,n-6) or min(29,n-6) bits, respectively. The power of two fills the modulus bit budget. These are structured prime fields, not arbitrary composite moduli or generic large-prime-order groups.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark discrete_log
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

This is an independently authored, lightweight adaptation of [AlgoTune discrete_log](https://github.com/oripress/AlgoTune/blob/dff9914c10800c7a031c9e8c3d4d1c8cd1b38906/AlgoTuneTasks/discrete_log/discrete_log.py). The source uses SymPy and a different generator/reference. SPEEDUP-MARK scores are not AlgoTune scores; no upstream implementation code is copied.
