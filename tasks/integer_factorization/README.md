# Integer Factorization

Factor `{"composite": N}` into `{"p": p, "q": q}` with distinct prime integers `p < q` and `p*q == N`. The generator guarantees such a semiprime. `n` controls prime magnitude rather than the upstream byte count: defaults are 400,000 and 800,000, giving bounded roughly 40-bit composites. Seeded cases include balanced and unbalanced factors. The baseline uses odd trial division; verification independently checks primality and product. Pollard-style factorization and arithmetic optimizations have useful room here, without treating cryptographic-size factoring as a laptop workload.

Optimize `candidate.py:solve(problem, reference_solve)`. Only Python standard-library dependencies are required. All output containers must be materialized built-in types; integers must be exact `int`, never `bool`. All construction and preprocessing belongs inside the timed solve call. Inputs are read-only.

Provenance: independently authored SPEEDUP-MARK adaptation of the task described in AlgoTune, `AlgoTuneTasks/integer_factorization/description.txt` (https://github.com/ScalingIntelligence/AlgoTune). No upstream implementation was copied. The bounded generator, Python data representations and exact verification contract below are SPEEDUP-MARK-specific; scores are not equivalent to upstream AlgoTune scores.

## Workload distribution

- **Size:** n scales prime factors, clamped to at least 20.
- **Selection:** seed % 3 == 0 permits a small second factor; other residues start it at scale//2.
- **Randomized:** First starting factor is drawn from [scale//4,scale); second from [3,2*scale) or [scale//2,2*scale). Each is advanced to a prime, with distinct factors enforced.
- **Fixed structure:** Instances are semiprimes. Advancing to the next prime means primes are not sampled uniformly.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
