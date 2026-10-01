# Exact Integer Signal Convolution

Given signed integer sequences `signal_x` and `signal_y`, return exactly `{"convolution": sequence_of_integers}`. The full convolution has coefficient `z[k] = sum(x[i] * y[k-i])` over valid indices. Empty inputs produce an empty output.

Modes:

- `full`: all `len(x) + len(y) - 1` coefficients.
- `same`: `len(x)` coefficients starting at full-output index `(len(y)-1)//2`.
- `valid`: full-output slice `[min(len(x),len(y))-1 : max(len(x),len(y))]`.

The `convolution` value must be a list or tuple of exact Python integer coefficients; other sequence types, floats, booleans, and lazy iterators are rejected. An FFT-based candidate may round internally but must return the exact integers. Inputs include dense signed signals, uneven lengths, and sparse kernels. `n` controls the first signal length. The reference uses conventional direct convolution; FFT, divide-and-conquer multiplication, sparse methods, and packing are possible optimization approaches. Preprocessing is timed.

Edit only `candidate.py`; preserve `solve(problem, reference_solve)`. Standard library only. Grade from the repository root with `python3 -m speedupmark fft_convolution`; inside a managed run use `python3 grade.py` to record progress.

This is an independently authored, lightweight adaptation of [AlgoTune fft_convolution](https://github.com/oripress/AlgoTune/blob/dff9914c10800c7a031c9e8c3d4d1c8cd1b38906/AlgoTuneTasks/fft_convolution/fft_convolution.py). The original uses floating-point signals and SciPy's FFT reference. This task deliberately uses exact integers and a direct reference to remain dependency-free; scores are not comparable with AlgoTune. No upstream implementation code is copied.

## Workload distribution

- **Size:** n is the first signal length; the second length is drawn from max(1,n//2) through n+1.
- **Selection:** seed % 3 selects full/same/valid; seed % 5 == 4 sparsifies the second signal. Joint cycle: 15.
- **Randomized:** Signal values are integers -32 through 32; sparse mode independently keeps about 10% of second-signal entries.
- **Fixed structure:** Mode and sparsity combinations are coupled through the seed cycle.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
