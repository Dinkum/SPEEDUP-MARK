# Register VM Program Execution

## Input and submission

Execute every program in a batch on a small register machine and return each program's final state. The input is `{"programs": [...]}`. Each program is `{"code": [[op, a, b, c], ...], "registers": [16 integers], "memory": [integers]}`. Return `{"states": [[registers, memory], ...]}`: one pair of integer lists per program, in input order, holding the sixteen registers and the whole memory when the program halts.

The machine has sixteen registers `r0`–`r15` holding 32-bit unsigned values, a memory of `len(memory)` words, a program counter starting at 0, and an initially empty call stack. Each instruction is four integers; unused operands are 0. After an instruction executes, control continues at the next instruction unless it jumps. Arithmetic wraps modulo 2**32, comparisons are unsigned, and a memory address is taken modulo the memory length.

| Op | Name | Effect |
| --- | --- | --- |
| 0 | `HALT` | stop; the state is final |
| 1 | `LI a b` | `r[a] = b` |
| 2 | `MOV a b` | `r[a] = r[b]` |
| 3 | `ADD a b c` | `r[a] = (r[b] + r[c]) mod 2**32` |
| 4 | `SUB a b c` | `r[a] = (r[b] - r[c]) mod 2**32` |
| 5 | `MUL a b c` | `r[a] = (r[b] * r[c]) mod 2**32` |
| 6 | `XOR a b c` | `r[a] = r[b] ^ r[c]` |
| 7 | `AND a b c` | `r[a] = r[b] & r[c]` |
| 8 | `OR a b c` | `r[a] = r[b] \| r[c]` |
| 9 | `ADDI a b c` | `r[a] = (r[b] + c) mod 2**32` |
| 10 | `SHL a b c` | `r[a] = (r[b] << c) mod 2**32`, `0 <= c <= 31` |
| 11 | `SHR a b c` | `r[a] = r[b] >> c`, `0 <= c <= 31` |
| 12 | `LT a b c` | `r[a] = 1` if `r[b] < r[c]` else `0` |
| 13 | `EQ a b c` | `r[a] = 1` if `r[b] == r[c]` else `0` |
| 14 | `LOAD a b c` | `r[a] = memory[(r[b] + c) mod len(memory)]` |
| 15 | `STORE a b c` | `memory[(r[b] + c) mod len(memory)] = r[a]` |
| 16 | `JMP a` | continue at instruction `a` |
| 17 | `JZ a b` | continue at instruction `b` if `r[a] == 0` |
| 18 | `JNZ a b` | continue at instruction `b` if `r[a] != 0` |
| 19 | `CALL a` | push the next instruction's index, continue at `a` |
| 20 | `RET` | pop an index and continue there |

Every supplied program is well formed: operands and jump targets are in range, immediates and initial values are in `0..2**32-1`, `RET` never runs on an empty stack, and the program reaches `HALT`. Programs are independent; no state carries from one to the next. Building Python source or code objects from a program with `compile` or `exec` is permitted.

Only Python standard-library dependencies are required. All output containers must be materialized built-in lists; integers must be exact `int`, never `bool`. All decoding, translation and execution belongs inside the timed solve call. Inputs are read-only.

## Reference and verification

The starting reference decodes and executes one instruction at a time through a chain of opcode comparisons. An independently structured table-dispatch interpreter in the task specification recomputes every final state and compares it exactly. Candidate improvements include cheaper dispatch, pre-decoded or fused instructions, translating a program into Python code, recovering loops and calls as native control flow, removing redundant masking, and deciding per program whether translation is worth its cost.

## Workload distribution

- **Size:** n is the approximate number of executed instructions in thousands; declared sizes are 1500 and 3000.
- **Selection:** Every input contains all five program kinds, shuffled: two long loop nests (about 30% of executed instructions), two data-dependent branching loops over memory (20%), two call-heavy programs with leaf routines and binary recursion (15%), warm programs of 1,000–12,000 instructions each (15%), and thousands of cold programs that execute about 80 instructions once (20%).
- **Randomized:** Instruction mixes, operands, immediates, loop trip counts, block lengths, routine counts, recursion depth, initial registers and memory contents.
- **Fixed structure:** Loop counters, branch conditions and the recursion depth live in registers that generated data instructions never write, so every program terminates. About half of the cold programs carry a large block behind an unconditional jump that never executes. Memory holds 16 words in cold programs and 256 or 1024 elsewhere.

The executable definition is [`task_spec.py`](task_spec.py), `generate_problem` and its helpers. See [sampling and coverage](../../GUIDE.md#sampling) for how the grader chooses and records seeds.

## Grading

Inside a managed run, edit only `candidate.py`, preserving `solve(problem)`.

From the repository root:

```console
python3 -m speedupmark register_vm_execution
```

Inside a managed run, use `python3 grade.py` to record progress.
See the [submission rules](../../GUIDE.md#submission-rules) and
[scoring guide](../../GUIDE.md#scoring) for shared requirements.

## Provenance

Independently authored for SPEEDUP-MARK. The instruction set, program generator and verifier are original to this repository; no external implementation was copied.
