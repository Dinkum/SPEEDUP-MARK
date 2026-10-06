# 1D Viscous Burgers Equation

## Input and submission

Integrate the semi-discrete viscous Burgers equation and return the interior state at `t1`.

The stencil is fixed. With zero Dirichlet values outside the interior nodes, spacing `dx`, and viscosity `nu`:

- diffusion is the second central difference divided by `dx²`
- advection is `u * u_x` with the upwind difference: backward where `u ≥ 0` and forward where `u < 0`

Input contains `t0`, `t1`, `y0`, `params` (`nu`, `dx`, `num_points`), and `x_grid`. `n` is the number of interior points. Return a list of Python floats of that length. The list must match an independent Radau integration of this same right-hand side within relative `1e-4` and absolute `1e-5`. Matching the mean, or only the conserved mass, is not enough.

Seeds rotate through `smooth` (viscosity 0.08), `steep` (a narrow bump, viscosity 0.01), and `two_wave`.

NumPy and SciPy are the optional numerical extra. One BLAS thread is set before they load.

## Reference and verification

The reference is SciPy `solve_ivp` with BDF at relative `1e-6` and absolute `1e-8`, without a hand-written Jacobian. Upstream used RK45 and compared a solution with itself. The second integrator is intentional.

## Workload distribution

- **Size:** n is interior grid-point count (minimum 8).
- **Selection:** seed % 3 selects smooth, steep, or two_wave.
- **Randomized:** Independent uniform perturbations in [-0.01,0.01) added to the family's initial profile.
- **Fixed structure:** Grid spans (-1,1); viscosity/final-time pairs are (0.08,0.25), (0.01,0.12), (0.04,0.2). Base sine/Gaussian profiles are fixed.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark pde_burgers1d
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

AlgoTune `AlgoTuneTasks/pde_burgers1d` at `dff9914c10800c7a031c9e8c3d4d1c8cd1b38906`. No upstream source was copied.
