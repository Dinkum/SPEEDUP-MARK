# Exact Circuit Fault Simulation

## Input and submission

Implement `solve(problem)`. A gate fault forces the output of one gate to zero or one for the entire circuit evaluation. For every gate, find the input patterns on which that fault changes at least one observed wire.

| Field | Meaning |
| --- | --- |
| `pattern_count` | Positive number of input patterns. Bit 0 represents the first pattern. |
| `inputs` | One nonnegative integer per primary input wire. Its low `pattern_count` bits hold that wire’s values and its higher bits are zero. |
| `gates` | Ordered triples `(op, a, b)`. The operation is `"and"`, `"or"`, or `"xor"`. Gate index `i` writes wire `len(inputs) + i`. Each operand names an earlier wire and both may name the same wire. |
| `outputs` | IDs of observed wires. An observed wire may also feed other gates. |

Return one pair `(stuck_zero, stuck_one)` per gate in the given order, using built-in lists or tuples and exact Python integers. Bit `p` in each mask is one exactly when the corresponding fault changes an observed value on pattern `p`. All higher bits must be zero. Evaluate each fault separately against the fault-free circuit. A fault affects every use of the gate’s output. A gate with no observable effect returns `(0, 0)`.

For example, consider one input carrying mask `0b10` for two patterns. Gate 0 computes `or(input, input)` and gate 1 computes `xor(gate0, gate0)`. Observing only gate 1 gives `[(0, 0), (0, 3)]`. Forcing gate 0 changes both XOR operands together, so the observed XOR stays zero. Forcing gate 1 to one is visible on both patterns.

## Reference and verification

The standard-library starter packs patterns into integers, skips gates outside the observed outputs’ ancestry, and follows gates with a single live consumer. It simulates the remaining fault sites in batches of 16. All preprocessing belongs inside the timed call. Scoring uses the ratio of reference time to candidate time, including output freezing.

The grader checks the materialized pairs against the measured reference answer. Tests separately evaluate small circuits one fault and one pattern at a time and check selected bits on generated circuits.

## Workload distribution

- **Size:** `n` is the minimum gate count. The generator completes its last stage. Direct grading uses 2400 and managed grading uses 2400 and 4200.
- **Selection:** At each size, the seed modulo three selects 64, 256, or 512 patterns. Every circuit mixes addition, comparison, conditional selection, and parity.
- **Randomized:** Bus widths range from four to eight wires. Each stage shuffles the previous wires, reconnects blocks, and draws side inputs. Gate IDs are shuffled subject to dependency order. Input masks and observed wires vary with the seed.
- **Fixed structure:** Each stage uses all four block types. Some final wires and one sampled gate per stage are observed, with duplicate observations removed. The solver sees the flattened gate list. [Generator](task_spec.py)

## Grading

In a managed workspace, edit `candidate.py` and run `python3 grade.py`. From the checkout, grade a submitted candidate with `python3 -m speedupmark circuit_fault_simulation`. The task belongs to `all`.

## Provenance

The generator, reference, and tests were written for SPEEDUP-MARK. The task simulates single stuck-at faults in combinational logic.
