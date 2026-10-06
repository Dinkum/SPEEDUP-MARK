# Integer Factorization

## Input and submission

Factor `{"composite": N}` into `{"p": p, "q": q}` with distinct prime integers `p < q` and `p*q == N`. The generator guarantees such a semiprime. `n` is a composite bit budget, from 24 through 80; managed cases use 64 and 80. Products usually have n or n-1 bits (close factors at odd n can give n-2). Every factor is below `2**64`.

Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The reference tries 128 Fermat steps for nearby factors, then uses replayable Pollard rho with Floyd cycle detection. Verification checks the exact product and each factor with deterministic 64-bit Miller–Rabin; it never reruns factorization. Families reward different choices: close factors favor Fermat, smooth p-1 favors order-based methods, and balanced or unbalanced factors change rho costs. Brent cycle detection, batched GCDs, and algorithm selection remain available. These bounded families do not establish performance on arbitrary balanced 120-bit semiprimes or an ECM/quadratic-sieve ladder.

## Workload distribution

- **Size:** n is a composite bit budget, 24–80; managed cases are 64 and 80.
- **Selection:** seed % 4 selects balanced, close, smooth_factor, or unbalanced. Four consecutive seeds cover all four.
- **Randomized:** Prime starts, smooth-order products, and close-factor gaps vary. Balanced and smooth_factor use a smaller-factor bit length min(32,n//2); unbalanced uses min(20,n//3). The other factor uses the remaining bit budget. Close uses two n//2-bit factors separated by a random even start gap 2–1024, advanced to a prime.
- **Fixed structure:** Distinct-prime semiprimes; the smaller generic factor is capped at 32 bits. Smooth-factor orders use small odd primes and a power of two. Prime advancement and primality conditioning do not sample primes uniformly. No factor or family label is supplied in the problem.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark integer_factorization
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/integer_factorization/description.txt` (https://github.com/oripress/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.
