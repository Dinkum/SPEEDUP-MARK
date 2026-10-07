"""Compile exact integer DAGs for a bounded, pipelined scratchpad machine."""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import SolutionEvaluation, load_reference


_reference = load_reference(__file__)
_latency = _reference._latency
_resource = _reference._resource


MASK = (1 << 32) - 1


def _operation(op, left, right):
    if op == "add":
        return (left + right) & MASK
    if op == "xor":
        return left ^ right
    return (left * right) & MASK


def _operands(nodes, instruction):
    """Return required node identities, independently of submitted slot values."""
    op, node = instruction[:2]
    kind, left, right = nodes[node]
    if op == "madd":
        if kind != "add" or nodes[left] is None or nodes[left][0] != "mul":
            raise ValueError("madd requires an add with a multiplication on its left")
        return (*nodes[left][1:], right)
    if op != kind and not (op == "mul_alu" and kind == "mul"):
        raise ValueError("operation does not implement the declared node")
    return left, right


def _simulate(workload, schedule):
    nodes, inputs = workload["nodes"], workload["inputs"]
    machine = workload["machine"]
    # Bounds make malformed submissions cheap to reject before simulation.
    bound = 32 * len(nodes) + 32
    if type(schedule) not in (tuple, list) or not 0 < len(schedule) <= bound:
        raise ValueError("invalid schedule length")
    registers = [None] * machine["scratch_slots"]
    ready = [0] * len(registers)
    memory = dict(enumerate(inputs))
    pending = {}
    last_completion = 0
    instruction_count = 0

    def slot(index, cycle):
        if not 0 <= index < len(registers) or ready[index] > cycle or registers[index] is None:
            raise ValueError("unavailable scratch slot")
        return registers[index]

    for cycle, packet in enumerate(schedule):
        for target, index, value in pending.pop(cycle, ()):
            if target == "memory":
                memory[index] = value
            else:
                registers[index] = value
        if type(packet) not in (tuple, list) or len(packet) > sum(machine[k] for k in ("alu", "mul", "load", "store")):
            raise ValueError("invalid packet")
        used, banks, writes, completions = {}, set(), set(), []
        for instruction in packet:
            instruction_count += 1
            if instruction_count > bound or type(instruction) not in (tuple, list) or len(instruction) < 3:
                raise ValueError("invalid instruction")
            op = instruction[0]
            lengths = {"load": 3, "store": 3, "add": 5, "xor": 5, "mul": 5, "mul_alu": 5, "madd": 6}
            if type(op) is not str or op not in lengths or len(instruction) != lengths[op]:
                raise ValueError("invalid opcode or arity")
            if any(type(index) is not int for index in instruction[1:]):
                raise ValueError("instruction indexes must be exact integers")
            resource = _resource(op)
            used[resource] = used.get(resource, 0) + 1
            if used[resource] > machine[resource]:
                raise ValueError("issue capacity exceeded")
            completion = cycle + _latency(op)
            last_completion = max(last_completion, completion)
            if op in ("load", "store"):
                node = instruction[1] if op == "load" else instruction[2]
                if not 0 <= node < len(nodes):
                    raise ValueError("invalid memory address")
                bank = node % machine["banks"]
                if bank in banks:
                    raise ValueError("memory bank conflict")
                banks.add(bank)
            if op == "store":
                identity, value = slot(instruction[1], cycle)
                if identity != node:
                    raise ValueError("store provenance mismatch")
                completions.append((completion, "memory", node, value))
                continue
            node, destination = instruction[1:3]
            if not 0 <= destination < len(registers) or ready[destination] > cycle or destination in writes:
                raise ValueError("destination is busy or written twice")
            writes.add(destination)
            if op == "load":
                if node not in memory:
                    raise ValueError("load before spill completion")
                value = memory[node]
            else:
                if not len(inputs) <= node < len(nodes):
                    raise ValueError("invalid computation node")
                operands = _operands(nodes, instruction)
                sources = [slot(index, cycle) for index in instruction[3:]]
                if tuple(source[0] for source in sources) != tuple(operands):
                    raise ValueError("operand provenance mismatch")
                values = [source[1] for source in sources]
                value = ((values[0] * values[1] + values[2]) & MASK if op == "madd"
                         else _operation(nodes[node][0], *values))
            completions.append((completion, "register", destination, (node, value)))
        # Every instruction reads the cycle's initial state. Writes become
        # pending only after the entire packet has been checked.
        for completion, target, index, value in completions:
            pending.setdefault(completion, []).append((target, index, value))
            if target == "register":
                ready[index] = completion
                registers[index] = None
    for cycle in sorted(pending):
        for target, index, value in pending[cycle]:
            if target == "memory":
                memory[index] = value
            else:
                registers[index] = value
    expected = list(inputs)
    for op, left, right in nodes[len(inputs):]:
        expected.append(_operation(op, expected[left], expected[right]))
    if any(node not in memory or memory[node] != expected[node] for node in workload["outputs"]):
        raise ValueError("missing or incorrect output store")
    return max(len(schedule), last_completion)


class ScratchpadDataflowCompilerTask:
    name = "scratchpad_dataflow_compiler"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 240
    grading_cases = (240, 480)
    metric_unit = "cycles"

    def generate_problem(self, n=240, random_seed=0):
        if n < 8:
            raise ValueError("n must be at least 8")
        rng = random.Random(random_seed)
        workloads = []
        for family in ("shared", "chains", "pressure", "multiply_add"):
            input_count = max(8, n // 8)
            nodes = [None] * input_count
            for index in range(n):
                node = len(nodes)
                if family == "chains":
                    left = max(0, node - 4)
                    right = rng.randrange(input_count)
                elif family == "shared":
                    left = rng.randrange(max(1, node // 3))
                    right = rng.randrange(max(input_count, node - 12), node) if node > input_count else rng.randrange(input_count)
                elif family == "pressure":
                    left, right = rng.randrange(node), rng.randrange(node)
                else:
                    left = node - 1 if index % 2 else rng.randrange(node)
                    right = rng.randrange(input_count)
                op = ("add" if index % 2 else "mul") if family == "multiply_add" else rng.choice(("add", "xor", "mul"))
                nodes.append((op, left, right))
            machine = {"scratch_slots": 8 if family == "pressure" else 16,
                       "banks": 2 if family == "shared" else 4,
                       "alu": 2, "mul": 1, "load": 2, "store": 1}
            workloads.append({"family": family, "nodes": tuple(nodes),
                              "inputs": tuple(rng.randrange(1 << 32) for _ in range(input_count)),
                              "outputs": tuple(range(len(nodes) - max(4, n // 6), len(nodes))),
                              "machine": machine})
        return {"workloads": tuple(workloads)}

    solve = staticmethod(_reference.solve)


    def evaluate_solution(self, problem, proposed):
        try:
            if type(proposed) not in (tuple, list) or len(proposed) != len(problem["workloads"]):
                raise ValueError("one schedule per workload is required")
            cycles = sum(_simulate(workload, schedule) for workload, schedule in zip(problem["workloads"], proposed))
            return SolutionEvaluation(float(cycles), True)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            return SolutionEvaluation(0.0, False)

    def is_solution(self, problem, proposed):
        return self.evaluate_solution(problem, proposed).correct


TASK = ScratchpadDataflowCompilerTask()
