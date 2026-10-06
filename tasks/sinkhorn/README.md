# Entropy-Regularized Transport

## Input and submission

Compute an entropy-regularized transport plan at the accuracy stored in the problem.

The objective is `<M, P> - reg * H(P)` with `H(P) = -sum P (log P - 1)` and positive marginals `source_weights` and `target_weights`. Input also contains `cost_matrix`, `reg`, and `accuracy`. Return `transport_plan`. The plan must be finite and strictly positive, its marginal L1 error must be at most `accuracy`, and `log P + M / reg` must be within `1e-4` of an additive table `u_i + v_j`. An exact but unregularized transport plan fails that Gibbs residual. A scaled Gibbs kernel with the wrong marginals fails the marginal test.

Seeds rotate through `weak_reg`, `strong_reg`, and `uneven` masses and shapes.

NumPy is the optional numerical extra. One BLAS thread is set before it loads.

## Reference and verification

The reference is stabilized log-domain Sinkhorn. Upstream calls `ot.sinkhorn` and requires one numerical matrix. The fixed accuracy test is intentional.

## Workload distribution

- **Size:** n is source-support size; destinations number n or max(2,n//2). Managed cases use 128 and 384.
- **Selection:** seed % 3 selects weak_reg, strong_reg, or uneven.
- **Randomized:** Random 2D support coordinates and normalized masses (uniform in weak_reg, random in strong_reg, fourth powers of random draws in uneven).
- **Fixed structure:** Regularization is 0.02/1/0.05; cost scales 1/8/3; target accuracy is 1e-5.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark sinkhorn
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/sinkhorn` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
