# Layout-Aware Pipeline Compiler

Compile a generated tensor pipeline into a program for a vector scratchpad
machine. The algorithm is given; the *organization* is the submission. A
candidate chooses which intermediates exist, where they live and in which
layout, how the work is tiled, what it recomputes instead of storing, and how
transfers, elementwise arithmetic, multiplies, shuffles and reductions are
scheduled against each other. This is the algorithm/schedule separation Halide
exposes, with the schedule chosen by the candidate instead of searched by a
compiler.

## Input and submission

`problem["workloads"]` is a tuple of workloads. Each contains:

- `inputs`: `(name, (row_count, column_count), address)` per input tensor.
  These are shape and address descriptors, not tensor values. The runtime
  tensor image starts at `address`.
- `pipeline`: `(name, expression)` stages in evaluation order, where an
  expression is one of
  - `("ew", "add" | "mul", left, right)` — elementwise on equal shapes,
  - `("stencil", source, kernel)` — 1-D kernel of one to five signed taps along
    the last axis with zero padding,
  - `("transpose", source)` — 2-D transpose,
  - `("reduce", source)` — sum along the last axis, producing a column.
  Arithmetic is unsigned 32-bit wrapping. Every stage is either consumed by a
  later stage or listed in `outputs`; the generator emits no dead stages.
- `outputs`: stage names that must be present in slow memory when the program
  ends.
- `machine`: `lanes` (vector width), `scratch` (vector slots, all zero at
  cycle 0), `banks`, per-cycle issue capacities for `alu`, `mul`, `perm`,
  `load` and `store`, and `memory` (cell capacity). A vector transfer in
  address `a` occupies bank `(a // lanes) % banks`, and one transfer per bank is
  allowed per cycle.
- `family`: a descriptive label (`chain`, `shared_layout`, `pressure`,
  `bandwidth`). It documents the regime; nothing in verification depends on it.

Storage convention: a 2-D tensor is row-major with `4` zero cells after each
row, so a stencil tap that runs past the end of a row reads a zero. Memory
starts zeroed, which is why the reference's pad cells stay zero.

Implement `candidate.py::solve(problem, reference_solve)` and return one
submission per workload: `{"program": schedule, "placements": {output:
(base, row_stride, column_stride)}}`. Cell `(row, column)` of an output lives at
`base + row * row_stride + column * column_stride`; output regions must be
disjoint and inside memory. Submissions contain only exact built-in dictionaries,
lists/tuples, strings, and integers; subclasses and executable objects are rejected.

Placements are checked by their actual cell addresses, so column-major and
reversed layouts are legal when every cell is in bounds and no cells overlap.
Reduction results have shape `(source_rows, 1)` and can feed later pipeline
stages like any other 2-D tensor.

The candidate compiles once from this description. Only after the submission is
copied into plain program data does the verifier draw runtime tensor values.
The same unchanged program must work on three fresh sets of values per workload.
Paired grading freezes both reference and candidate submissions before drawing
an independent 256-bit verification seed, then checks both on the same values.
Input values are never provided to `solve` or its `reference_solve` callback.
Pipeline constants, such as stencil coefficients, remain available to compilation.

## Instructions

| Instruction | Meaning | Issue | Latency |
| --- | --- | --- | --- |
| `("vload", dst, addr)` | `lanes` consecutive cells into scratch slot `dst` | load | 4 |
| `("vstore", src, addr)` | slot `src` into `lanes` consecutive cells | store | 4 |
| `("sload", dst, lane, addr)` | one cell into one lane | load | 8 |
| `("sstore", src, lane, addr)` | one lane into one cell | store | 8 |
| `("vconst", dst, value)` | every lane set to an unsigned 32-bit constant | alu | 1 |
| `("vadd", dst, a, b)` | elementwise wrapping add | alu | 1 |
| `("vmul", dst, a, b)` | elementwise wrapping multiply | mul | 3 |
| `("vmadd", dst, a, b, c)` | `a * b + c`, elementwise | mul | 3 |
| `("vperm", dst, a, b, mask)` | lane `i` takes `a[mask[i]]` if `mask[i] < lanes`, else `b[mask[i] - lanes]` | perm | 1 |
| `("vred", dst, src)` | lane 0 becomes the wrapping sum of `src`; other lanes zero | alu | 5 |

A packet is a tuple of instructions issued in one cycle, ordered. Reads observe
the pre-write state of the cycle. An earlier instruction in the packet reserves its destination immediately, so a later instruction cannot read that slot in the same packet, even to request its old value. A producer's value is readable at the start of
`issue + latency`. A slot may not be read or rewritten before its producer
completes. Stores commit at their completion cycle; a load issued before that cycle reads the earlier memory contents. The machine issues in order: an instruction may share a cycle with its
predecessor, never precede it. Slots are zero-initialized, so only value
correctness matters.

## Scoring and verification

The score is the total simulated cycles across the workloads; a workload costs
the larger of its packet count and its last completion cycle. Each workload
is counted once, not once per correctness replay: the straight-line instruction
schedule and opcode latencies make its cost independent of input values. Programs are
bounded by `32 * cells + 32` packets and the same number of instructions.

For each of three runtime inputs, the verifier resets the machine, loads the
input values, executes the fixed program, then independently evaluates the
pipeline from the algorithm description and those same values. It requires every
declared output cell to hold the evaluated value. Only outputs are checked, so
fusion, recomputation, in-place reuse, discarding intermediates, arbitrary
output layouts and inputs being overwritten are all legal. Correctness is a
property of values, not of structure. Compilation host time is not scored and is
bounded by the worker timeout.

Verification receipts record the private seed, generator version, problem hash,
frozen program hashes and tensor digest before simulation, including for failed
programs. The harness binds receipts to candidate source bytes and the task
revision. Replay validates these bindings before executing the program; changing
candidate code requires a fresh verification challenge. The task exposes this
through `evaluate_pair(problem, outputs, replay=receipt, record=callback)`; the
single-submission `evaluate_solution` uses the same verification logic.

## Reference and improvements

`_compile` is deliberately unsophisticated but never absurd: it allocates a
padded row-major region per stage, materializes every intermediate, reloads a
shifted window for each stencil tap, and moves single cells for a transpose.
Its own schedule is issued through the same scoreboarding a candidate faces, so
its cycle count is a real program, not a formula.

Each generated program presents one of four optimization regimes:

| Family | Optimization decisions |
| --- | --- |
| `chain` | Fuse variable-length elementwise chains and reuse stencil windows. |
| `shared_layout` | Choose layouts across row-wise and transposed consumers with varying branch depth. |
| `pressure` | Choose reuse, recomputation, and spills across 2–4 branches under 4–6 scratch slots. |
| `bandwidth` | Reuse shifted windows and select reduction placement under a tight memory capacity. |

`tests/test_layout_pipeline_compiler.py` includes an explicit canonical chain fixture where hand-written fusion uses less than half the reference cycles. That verifies an accessible optimization, not an aggregate speedup across the generated distribution. The broader program distribution requires choosing schedules from the actual pipeline.

This is an independently authored SPEEDUP-MARK task, not an Anthropic port.

```console
python3 -m speedupmark layout_aware_pipeline_compiler
```

Inside a managed run, use `python3 grade.py`. Edit only `candidate.py`.

## Workload distribution

- **Size:** n determines rows rounded down to a multiple of eight; columns are randomly 8, 16, or 24.
- **Selection:** Every input contains chain, shared_layout, pressure, and bandwidth.
- **Randomized:** The public seed selects pipeline structure and machine parameters.
  Runtime tensor cells come from the versioned `shake256-uint32be-v1` stream
  expanded from an independent 256-bit seed drawn after both programs are frozen.
  The three streams per workload are shared by reference and candidate. They
  cannot be reconstructed from the public seed. `--seed` replays the compilation
  problem; the separate verification receipt also replays correctness values
  for the same candidate bytes, frozen programs, problem and task revision. Chain has 3–6 elementwise stages plus a stencil and final combine. Shared-layout has row and transposed branches of 1–3 stages. Pressure has 2–4 stencil branches with varying source reuse and a join tree. Bandwidth has 2–4 stencils and a reduction from a random stage. Stencil lengths are 2–5 with nonzero coefficients in {-2,-1,1,2,3}; elementwise operators and reuse choices vary.
- **Fixed structure:** The instruction set and four structural regimes are fixed. Scratch slots are 8/12 for chain, 8/16 for shared_layout, 4/6 for pressure, and 4 for bandwidth; pressure banks vary between 2/4, bandwidth has 2, others 4. All stages feed an output, directly or through later stages.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.
