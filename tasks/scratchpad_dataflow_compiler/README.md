# Scratchpad Dataflow Compiler

Compile four integer computation graphs into schedules for a pipelined machine.
Every generated problem includes shared subexpressions, parallel chains, scratch
pressure, and multiply-add chains. Scheduling, spilling, recomputation, fusion,
and execution-unit selection interact; no external compiler or GPU is required.

## Input and submission

`problem["workloads"]` is a tuple of workloads. Each contains:

- `inputs`: unsigned 32-bit integers, initially in slow memory at node IDs
  `0..len(inputs)-1`.
- `nodes`: a tuple beginning with one `None` per input, followed by
  `(opcode, left_node, right_node)` records. Operands always precede their node.
  Opcodes are `add`, `xor`, and `mul`; arithmetic wraps modulo `2**32`.
- `outputs`: node IDs that must exist in slow memory at completion.
- `machine`: scratch slot count, memory bank count, and per-cycle issue capacities
  for `alu`, `mul`, `load`, and `store`. Memory address `node` uses bank
  `node % banks`, with one total transfer per bank per cycle.
- `family`: descriptive workload label; correctness is defined by the graph.

Implement `candidate.py::solve(problem)`. Return one schedule per
workload, in order. A schedule is a built-in list or tuple of cycle packets, each a built-in list or
tuple of instructions. Empty packets are idle cycles. All instruction indexes
must be exact Python integers.

## Instructions

| Instruction | Meaning | Issue resource | Latency |
| --- | --- | --- | --- |
| `("load", node, dst)` | Load an input or previously stored node into scratch slot `dst` | load | 2 |
| `("store", src, node)` | Store scratch slot `src` to its own node's memory address | store | 2 |
| `("add", node, dst, left, right)` | Implement that graph node from scratch operands | alu | 1 |
| `("xor", node, dst, left, right)` | Implement that graph node from scratch operands | alu | 1 |
| `("mul", node, dst, left, right)` | Implement that graph node on the multiplier | mul | 3 |
| `("mul_alu", node, dst, left, right)` | Implement a multiplication on the ALU | alu | 5 |
| `("madd", node, dst, a, b, c)` | Fuse `add(mul(a,b),c)` without materializing its multiplication node | mul | 3 |

For `madd`, the destination graph node must be an `add` whose **left** operand is
a `mul`. Scratch operands must have exactly the node identities of that
multiplication's left/right operands and the addition's right operand, in that
order. The multiplication node can still be computed independently if another
consumer needs it. No commutative operand swaps or other algebraic rewrites are
implicitly accepted.

Every scratch operand must be ready at the start of its issue cycle. An
instruction issued at cycle `t` completes at `t + latency`; its result is usable
at the start of that cycle. All instructions in a packet read the pre-write
scratch state. A ready source slot may also be a destination in that packet,
but two writes to one slot are forbidden. A pending destination cannot be read
or overwritten. Stores capture their source value at issue; loads may access
only completed memory stores. Scratch starts empty. Slow memory is keyed by
node identity and retains stored values, so reloading, duplicate stores, and
recomputation are allowed. There are no arbitrary immediate constants or memory
addresses.

## Scoring and verification

The score is the reference's total simulated cycles divided by the candidate's
total simulated cycles across the four workloads. Each schedule costs the larger
of its packet count and its last instruction completion time. Final transfers
are included; trailing idle packets cost cycles. Schedules must contain between
1 and `32 * len(nodes) + 32` packets, and at most that many instructions.

The frozen task spec independently executes instructions, enforces hardware
limits and node provenance, evaluates the original graph directly, and checks
all output stores. Equal values with different node identities cannot substitute
for required operands. Scratch and simulated instruction budgets are enforced;
host compiler memory is not separately limited. Compilation host time is not
part of the cycle score, but the harness's 60-second default worker timeout
bounds the entire evaluation, including compilation and simulation.

## Reference and verification

The reference uses topological scheduling, register caching, farthest-next-use
spills, dead-subgraph elimination, and a resource scoreboard. It does not fuse
multiply-adds, select alternate execution units, or reorder ready graph nodes.
`n` is the number of computation nodes per workload; declared managed grading
sizes are 240 and 480, with all four families at both sizes.

The task implementation is original SPEEDUP-MARK code.

## Workload distribution

- **Size:** n is DAG operation count (minimum 8); input count is max(8,n//8).
- **Selection:** Every input contains shared, chains, pressure, and multiply_add.
- **Randomized:** DAG operand choices and uint32 input values.
- **Fixed structure:** Operation mix and machine resources depend on family; pressure has 8 slots versus 16 otherwise, shared has 2 banks versus 4. Outputs are the last max(4,n//6) nodes.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark scratchpad_dataflow_compiler
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.
