# SIMD Traversal Kernel

Optimize a traversal kernel for a small simulated SIMD processor.
Optimize vectorization, instruction order, table reuse, and the number of
traversals in flight. Exact outputs are required; lower simulated cycles win.
The kernel, machine, reference, and generator are original SPEEDUP-MARK code.

## Kernel

Each state has a node index `node` and an unsigned 32-bit `value`. Repeat
the following for the workload's declared number of rounds:

```python
rotated = ((value << 7) | (value >> 25)) & 0xffffffff
value = ((value ^ payload[node]) * 2654435761 + rotated) & 0xffffffff
node = edges[2 * node + (value >> 31)]
```

Write **both** the final node indices and values to their declared output
regions. States are independent, but each state's next lookup depends on its
previous step. The table and initial states vary at runtime.

## Input and submission

Implement `candidate.py::solve(problem, reference_solve)`. `problem["workloads"]`
contains four descriptor dictionaries. Return one instruction stream per workload,
in that order: a built-in list/tuple of built-in lists/tuples of instruction
lists/tuples. Each instruction contains an opcode string followed by exact Python
integers.

Each descriptor gives `family`, `nodes`, `rounds`, `batch`, total `memory` cells,
and base addresses for `payload`, `edges`, `positions`, `values`, `out_positions`,
and `out_values`. The payload has `nodes` words, edges has `2 * nodes` words,
and each state/output array has `batch` words. Every other memory cell begins at
zero and can be used for temporary storage. All in-bounds memory is writable;
only the two final output regions are checked against the original inputs.

Compilation receives **no runtime words**. The reference callback also receives
only descriptors. Both compiled programs are copied into plain data before a
private verification seed is drawn. The same frozen programs run on three fresh
memories, shared between reference and candidate. All three must be correct.

## Machine

There are 20 registers, each containing eight unsigned 32-bit lanes, initially
zero. Every instruction specifies an active width `w` from 1 through 8. A write
changes only those leading lanes; the rest retain their values. Register readiness
is tracked for the whole register. Arithmetic wraps modulo 2**32.

The instruction stream issues in order. Consecutive independent instructions may
issue in the same cycle when their engines and memory banks have capacity. The
simulator automatically waits for source readiness, destination readiness,
memory dependencies, bank availability, and engine capacity. Reordering the
stream is the candidate's scheduling decision. There are no branches or jumps;
the compiler can unroll the public round count and batch dimensions.

| Instruction | Meaning | Engine | Latency |
| --- | --- | --- | --- |
| `("const", w, d, immediate)` | Broadcast a uint32 constant | alu | 1 |
| `("load", w, d, address)` | Load contiguous words | load | 6 + bank span - 1 |
| `("store", w, s, address)` | Store contiguous words | store | 6 + bank span - 1 |
| `("gather", w, d, base, indexes)` | Load `memory[base + indexes[lane]]` | load | 6 + bank span - 1 |
| `("add", w, d, a, b)` | Add registers | alu | 1 |
| `("xor", w, d, a, b)` | XOR registers | alu | 1 |
| `("muli", w, d, a, immediate)` | Multiply by a uint32 constant | mul | 3 |
| `("andi", w, d, a, immediate)` | AND with a uint32 constant | alu | 1 |
| `("shri", w, d, a, shift)` | Logical right shift by 0–31 | alu | 1 |
| `("rotli", w, d, a, shift)` | Rotate left by 0–31 | alu | 1 |
| `("pick", w, d, a, b, indexes)` | Select lanes from concatenated full registers `a,b`, indices 0–15 | perm | 2 |

Per-cycle issue capacities are two `alu` instructions and one each of `mul`,
`load`, `store`, and `perm`. Scalars consume the same instruction slots as vectors.
A result issued at cycle `c` is available at the start of `c + latency`.
Readers and subsequent writers wait for outstanding writes. Operands are read
at issue, so an in-place operation reads its previous value correctly.

Memory has four banks: word address `a` belongs to bank `a % 4`. A bank transfers
one unique word per cycle. A gather coalesces repeated addresses within that
instruction. Its bank span is the maximum number of unique addresses assigned
to any one bank. Each participating bank is reserved for its own count of cycles.
An instruction waits until all its banks are available, then reserves them
together. Loads and stores share the banks. Aliasing memory operations wait for
prior stores to complete. Memory bounds, register bounds, and instruction arities
are enforced; invalid instructions fail the submission.

The cost includes final completion of outstanding instructions, with a minimum
of one cycle. This is an explicit synthetic machine, not a prediction of native
CPU or GPU performance.

## Scoring and replay

The task's cost is the sum of workload cycle counts, averaged across the three
runtime trials. Bank conflicts depend on runtime addresses, so all correctness
trials contribute to the cost. SPEEDUP-MARK reports `reference_cost / candidate_cost`.
Host compilation time is excluded from that ratio and bounded by the normal
worker timeout. Each program is limited to `batch * (16 * rounds + 32) + 256`
instructions. Any incorrect output, invalid instruction, or resource violation
fails the task.

Verification receipts record the private seed, generator version, descriptor
hash, both frozen program hashes, and memory digest before execution, including
failed evaluations. The shared harness binds the receipt to task revision and
candidate bytes. `--seed` reproduces descriptors; the verification receipt is
also required to reproduce runtime memories and exact cycle counts. Replay
requires the same descriptors, programs, generator, and source identities.
A changed candidate gets a fresh challenge. Same-process hostile tampering with
the grader is outside this interface's protection.

## Reference and verification

`task_spec.py::_compile` is a scalar reference that retains each state's node and
value in registers across rounds. It has no artificial sleeps or redundant
round trips through memory. `task_spec.py::_oracle` computes the kernel directly,
independently of the simulator's instruction decoding and scheduling.

Vectorization reduces instruction overhead, while moving independent work ahead
of consumers and interleaving streams hide dependencies. Twenty registers hold
four straightforward five-register streams. Caching the shared table consumes
three registers and reduces that straightforward implementation to three streams.
More compact allocation, alternative instruction sequences, partial table
caching, and schedules adapted to bank pressure remain possible.

The calibration also reuses dead registers to fit five four-register streams,
or four streams plus the cached table. Extra streams can improve crowded batches
while losing on batches that leave the last tile partially filled. Register
allocation and tile size therefore require choosing together.

`tests/simd_traversal_kernel_schedules.py` contains hand-written calibration schedules.
`python3 tests/calibrate_simd_traversal_kernel.py` checks them against the independent oracle
and emits per-family cycles and diagnostics. These are engineering fixtures,
not model benchmark results. `Machine.stats` exposes instructions, unique loaded
and stored words, registers touched, and sequentially attributed dependency,
bank, and issue stall cycles. Registers touched is not a liveness analysis.

## Workload distribution

**Selection:** Every problem contains every family, even when grading uses one sample:

| Family | Nodes | Rounds | States | Optimization pressure |
| --- | --- | --- | --- | --- |
| `shared` | 8 | 6, 8, or 10 | n | Cache the three table vectors in scarce registers |
| `scattered` | 64, 128, or 256 | 6, 8, or 10 | n | Bank conflicts and dependent gathers |
| `deep` | 64, 128, or 256 | 16, 20, or 24 | n | Long chains reward interleaving independent states |
| `crowded` | 64, 128, or 256 | 6, 8, or 10 | 2n + 3 | Bounded tiles, register reuse, and partial-vector tails |

**Size:** `n` must be an integer in 1–4096. The default is 64; managed development
and final grading both use 32 and 128. The family table defines the batch sizes.

**Randomized:** The public seed chooses each listed size/round option and independent
0–3-word gaps before each memory region, varying their relative bank alignment.
Private runtime data use
`simd_traversal_kernel-shake256-v1`: SHAKE-256 expands a fresh
256-bit seed with workload/trial domain separation into big-endian uint32 words.
Payloads and state values are uniform uint32. Each edge and initial node is
a word modulo `nodes` (all generated node counts are powers of two). Tables,
initial positions, and values are independent draws. A shared table means a
small working set; states are not forced to follow identical paths.

**Fixed structure:** The machine, traversal update, four families, memory layout
rules, and three runtime trials are fixed. `task_spec.py::generate_problem` and
`_runtime_memory` are the executable distribution definitions.

## Grading

Edit only `candidate.py`, preserving `solve(problem, reference_solve)`.

From the repository root:

```console
python3 -m speedupmark simd_traversal_kernel
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
