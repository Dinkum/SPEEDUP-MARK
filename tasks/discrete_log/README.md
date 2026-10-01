# Prime-Field Discrete Logarithm

Given a prime `p`, a primitive root `g`, and `h` in its multiplicative group, return exactly `{"x": integer}` where `0 <= x < p-1` and `pow(g, x, p) == h`. The exponent is unique in that interval. Zero is a valid answer when `h == 1`; booleans are not integers in the output contract.

`n` controls the approximate prime size, not its bit length. The seeded generator chooses primes and primitive roots without external libraries. The reference walks the group by repeated modular multiplication. Baby-step/giant-step and other number-theoretic algorithms offer algorithmic headroom. All candidate tables and preprocessing are timed.

Edit only `candidate.py`; keep the `solve(problem, reference_solve)` entrypoint. Standard library only. Return fully materialized output and do not modify benchmark files. Grade from the repository root with `python3 -m speedupmark discrete_log`; inside a managed run use `python3 grade.py` to record progress.

This is an independently authored, lightweight adaptation of [AlgoTune discrete_log](https://github.com/oripress/AlgoTune/blob/dff9914c10800c7a031c9e8c3d4d1c8cd1b38906/AlgoTuneTasks/discrete_log/discrete_log.py). The source uses SymPy and a different generator/reference. SPEEDUP-MARK scores are not AlgoTune scores; no upstream implementation code is copied.

## Workload distribution

- **Size:** n scales the prime modulus, with a minimum of 7.
- **Selection:** One distribution over prime-field discrete logarithms.
- **Randomized:** A starting modulus near n is advanced to the next prime; the generator chooses a primitive root and a hidden exponent.
- **Fixed structure:** The returned target is a power of that root. This does not sample arbitrary composite-modulus instances.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
