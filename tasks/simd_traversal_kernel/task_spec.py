"""SIMD Traversal Kernel: compile a dependent traversal for a small banked SIMD machine.

This machine, kernel, and generator are original SPEEDUP-MARK implementations.
Compilation sees descriptors only. Programs are frozen before runtime data exist.
"""

import hashlib
import json
import random
import secrets
import struct

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import SolutionEvaluation, load_candidate


_candidate = load_candidate(__file__)
MASK = (1 << 32) - 1
LANES = 8
REGISTERS = 20
BANKS = 4
MULTIPLIER = 2654435761
FAMILIES = ("shared", "scattered", "deep", "crowded")
GENERATOR = "simd_traversal_kernel-shake256-v1"
TRIALS = 3
# Arithmetic issue capacity is shared by scalar and vector instructions.
ENGINES = {"alu": 2, "mul": 1, "load": 1, "store": 1, "perm": 1}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("ascii")).hexdigest()


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"expected an integer in [{low}, {high}]")
    return value


def _freeze(programs, workloads):
    """Bound and copy exact containers before sampling any private input."""
    if type(programs) not in (tuple, list) or len(programs) != len(workloads):
        raise ValueError("one instruction stream per workload is required")
    frozen = []
    for program, workload in zip(programs, workloads):
        limit = workload["batch"] * (16 * workload["rounds"] + 32) + 256
        if type(program) not in (tuple, list) or not 1 <= len(program) <= limit:
            raise ValueError("invalid instruction count")
        instructions = []
        for instruction in program:
            if (type(instruction) not in (tuple, list) or not 4 <= len(instruction) <= 6
                    or type(instruction[0]) is not str
                    or len(instruction[0]) > 6
                    or any(type(item) is not int or not 0 <= item <= MASK
                           for item in instruction[1:])):
                raise ValueError("instructions must contain only an opcode and plain integers")
            instructions.append(tuple(instruction))
        frozen.append(tuple(instructions))
    return tuple(frozen)


def _runtime_memory(workload, seed, index, trial):
    """Portable private input stream, independent of the public descriptor seed."""
    nodes, batch = workload["nodes"], workload["batch"]
    count = 3 * nodes + 2 * batch
    stream = hashlib.shake_256(
        b"speedmark/simd_traversal_kernel/v1\0" + bytes.fromhex(seed)
        + struct.pack(">II", index, trial)).digest(4 * count)
    words = iter(struct.unpack(f">{count}I", stream))
    payload = [next(words) for _ in range(nodes)]
    edges = [next(words) % nodes for _ in range(2 * nodes)]
    # Shared paths use the same eight-node table; every table entry is still
    # private, so caching the table is legal while compiling its answers is not.
    positions = [next(words) % nodes for _ in range(batch)]
    values = [next(words) for _ in range(batch)]
    memory = [0] * workload["memory"]
    for name, data in (("payload", payload), ("edges", edges),
                       ("positions", positions), ("values", values)):
        base = workload[name]
        memory[base:base + len(data)] = data
    return tuple(memory)


def _oracle(workload, memory):
    """Direct scalar semantics, independent of instruction execution/scheduling."""
    positions, values = [], []
    for item in range(workload["batch"]):
        node = memory[workload["positions"] + item]
        value = memory[workload["values"] + item]
        for _ in range(workload["rounds"]):
            rotated = ((value << 7) | (value >> 25)) & MASK
            value = ((value ^ memory[workload["payload"] + node])
                     * MULTIPLIER + rotated) & MASK
            node = memory[workload["edges"] + 2 * node + (value >> 31)]
        positions.append(node)
        values.append(value)
    return tuple(positions), tuple(values)


class Machine:
    """In-order multiple-issue simulator with register and memory scoreboards.

    Values are computed at issue and tagged with their availability cycle.
    Every later reader and writer waits for that tag. In-order issue makes
    this equivalent to delayed writes without a separate event queue.
    """

    def __init__(self, workload, memory):
        self.memory = list(memory)
        if len(self.memory) != workload["memory"]:
            raise ValueError("wrong memory size")
        self.registers = [[0] * LANES for _ in range(REGISTERS)]
        self.ready = [0] * REGISTERS
        self.memory_ready = [0] * len(memory)
        self.banks = [0] * BANKS
        self.cycle = 0
        self.finish = 0
        self.issued = {}
        self.used_registers = set()
        self.stats = {"instructions": 0, "load_words": 0, "store_words": 0,
                      "dependency_stalls": 0, "bank_stalls": 0,
                      "issue_stalls": 0}

    def _reg(self, register):
        _integer(register, 0, REGISTERS - 1)
        self.used_registers.add(register)
        return register

    def _advance(self, cycle):
        if cycle > self.cycle:
            self.cycle = cycle
            self.issued = {}

    def run(self, program):
        for instruction in program:
            op, width, target, *args = instruction
            _integer(width, 1, LANES)
            target = self._reg(target)
            sources, addresses = [], []
            destination = op != "store"
            if op in ("load", "store", "const"):
                if len(args) != 1:
                    raise ValueError("wrong instruction arity")
                engine, latency = ("alu", 1) if op == "const" else (op, 6)
                if op == "const":
                    _integer(args[0], 0, MASK)
                else:
                    _integer(args[0], 0, len(self.memory) - width)
                    addresses = list(range(args[0], args[0] + width))
                if op == "store":
                    sources = [target]
            elif op == "gather":
                if len(args) != 2:
                    raise ValueError("wrong gather arity")
                _integer(args[0], 0, len(self.memory) - 1)
                sources = [self._reg(args[1])]
                engine, latency = "load", 6
            elif op == "pick":
                if len(args) != 3:
                    raise ValueError("wrong pick arity")
                sources = [self._reg(arg) for arg in args]
                engine, latency = "perm", 2
            elif op in ("add", "xor", "muli", "andi", "shri", "rotli"):
                if len(args) != 2:
                    raise ValueError("wrong arithmetic arity")
                sources = [self._reg(args[0])]
                if op in ("add", "xor"):
                    sources.append(self._reg(args[1]))
                else:
                    _integer(args[1], 0, 31 if op in ("shri", "rotli") else MASK)
                engine, latency = ("mul", 3) if op == "muli" else ("alu", 1)
            else:
                raise ValueError(f"unknown opcode {op!r}")

            # WAW waits matter even when an instruction has no source operands.
            needed = sources + ([target] if destination else [])
            ready = max((self.ready[reg] for reg in needed), default=0)
            self.stats["dependency_stalls"] += max(0, ready - self.cycle)
            self._advance(ready)
            if op == "gather":
                addresses = [args[0] + value for value in self.registers[sources[0]][:width]]
                for address in addresses:
                    _integer(address, 0, len(self.memory) - 1)
            unique = set(addresses)
            if unique:
                ready = max(self.memory_ready[address] for address in unique)
                self.stats["dependency_stalls"] += max(0, ready - self.cycle)
                self._advance(ready)
                bank_counts = [sum(address % BANKS == bank for address in unique)
                               for bank in range(BANKS)]
                ready = max(self.banks[bank] for bank, count in enumerate(bank_counts) if count)
                self.stats["bank_stalls"] += max(0, ready - self.cycle)
                self._advance(ready)
                latency += max(bank_counts) - 1
            if self.issued.get(engine, 0) == ENGINES[engine]:
                self.stats["issue_stalls"] += 1
                self._advance(self.cycle + 1)
            self.issued[engine] = self.issued.get(engine, 0) + 1
            completion = self.cycle + latency
            self.finish = max(self.finish, completion)
            if unique:
                for bank, count in enumerate(bank_counts):
                    if count:
                        self.banks[bank] = self.cycle + count
                if op == "store":
                    for address, value in zip(addresses, self.registers[target][:width]):
                        self.memory[address] = value
                        self.memory_ready[address] = completion
                    self.stats["store_words"] += len(unique)
                else:
                    values = [self.memory[address] for address in addresses]
                    self.stats["load_words"] += len(unique)
            elif op == "const":
                values = [args[0]] * width
            elif op == "pick":
                table = self.registers[sources[0]] + self.registers[sources[1]]
                indexes = self.registers[sources[2]][:width]
                values = [table[_integer(index, 0, 2 * LANES - 1)] for index in indexes]
            else:
                left = self.registers[sources[0]][:width]
                right = (self.registers[sources[1]][:width] if len(sources) == 2
                         else [args[1]] * width)
                if op == "add":
                    values = [(a + b) & MASK for a, b in zip(left, right)]
                elif op == "xor":
                    values = [a ^ b for a, b in zip(left, right)]
                elif op == "muli":
                    values = [(a * b) & MASK for a, b in zip(left, right)]
                elif op == "andi":
                    values = [a & b for a, b in zip(left, right)]
                elif op == "shri":
                    values = [a >> b for a, b in zip(left, right)]
                else:
                    values = [((a << b) | (a >> ((32 - b) % 32))) & MASK
                              for a, b in zip(left, right)]
            if destination:
                self.registers[target][:width] = values
                self.ready[target] = completion
            self.stats["instructions"] += 1
        self.stats.update(cycles=max(self.cycle + 1, self.finish),
                          scratch_registers=len(self.used_registers))
        return self.stats["cycles"]


def _compile(workload):
    """Readable scalar baseline retaining each state in scratch across rounds."""
    program = []
    for item in range(workload["batch"]):
        program.extend((("load", 1, 0, workload["positions"] + item),
                        ("load", 1, 1, workload["values"] + item)))
        for _ in range(workload["rounds"]):
            program.extend((
                ("gather", 1, 2, workload["payload"], 0),
                ("rotli", 1, 3, 1, 7),
                ("xor", 1, 2, 1, 2),
                ("muli", 1, 2, 2, MULTIPLIER),
                ("add", 1, 1, 2, 3),
                ("shri", 1, 3, 1, 31),
                ("muli", 1, 4, 0, 2),
                ("add", 1, 4, 4, 3),
                ("gather", 1, 0, workload["edges"], 4),
            ))
        program.extend((("store", 1, 0, workload["out_positions"] + item),
                        ("store", 1, 1, workload["out_values"] + item)))
    return tuple(program)


def _verify(workloads, programs, memories):
    if programs is None:
        return SolutionEvaluation(0.0, False)
    total = 0
    try:
        for workload, program, trials in zip(workloads, programs, memories):
            for memory in trials:
                expected = _oracle(workload, memory)
                machine = Machine(workload, memory)
                total += machine.run(program)
                actual = tuple(tuple(machine.memory[workload[name]:workload[name] + workload["batch"]])
                               for name in ("out_positions", "out_values"))
                if actual != expected:
                    return SolutionEvaluation(0.0, False)
        # Gather bank occupancy depends on runtime addresses. Every correctness
        # trial therefore also contributes its measured cycles to the mean.
        return SolutionEvaluation(total / TRIALS, True)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return SolutionEvaluation(0.0, False)


def _evaluate_submissions(problem, outputs, *, replay=None, record=None):
    workloads = problem["workloads"]
    frozen = {}
    for role, output in outputs.items():
        try:
            frozen[role] = _freeze(output, workloads)
        except ValueError:
            frozen[role] = None
    receipt = {
        "generator_version": GENERATOR,
        "problem_sha256": _digest(problem),
        "program_sha256": {role: _digest(program) for role, program in frozen.items()},
    }
    if replay is None:
        seed = secrets.token_hex(32)
    else:
        if type(replay) is not dict or set(replay) != set(receipt) | {"seed", "memory_sha256"}:
            raise ValueError("invalid SIMD Traversal Kernel verification receipt")
        if any(replay[key] != value for key, value in receipt.items()):
            raise ValueError("replay requires identical descriptors, programs and generator")
        seed = replay["seed"]
        if (type(seed) is not str or len(seed) != 64
                or any(char not in "0123456789abcdef" for char in seed)):
            raise ValueError("invalid verification seed")
    memories = tuple(tuple(_runtime_memory(workload, seed, index, trial)
                            for trial in range(TRIALS))
                     for index, workload in enumerate(workloads))
    receipt.update(seed=seed, memory_sha256=_digest(memories))
    if replay is not None and replay["memory_sha256"] != receipt["memory_sha256"]:
        raise ValueError("replay memory digest mismatch")
    if record is not None:
        record(receipt)
    return {role: _verify(workloads, program, memories) for role, program in frozen.items()}


class SIMDTraversalKernelTask:
    name = "simd_traversal_kernel"
    task_version = "1.1.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 64
    grading_cases = (32, 128)
    metric_unit = "cycles"

    def generate_problem(self, n=64, random_seed=0):
        _integer(n, 1, 4096)
        rng = random.Random(random_seed)
        workloads = []
        for family in FAMILIES:
            nodes = 8 if family == "shared" else rng.choice((64, 128, 256))
            rounds = rng.choice((16, 20, 24)) if family == "deep" else rng.choice((6, 8, 10))
            batch = 2 * n + 3 if family == "crowded" else n
            # Independent gaps vary relative bank alignment. A common shift
            # would merely rename all four banks and preserve their conflicts.
            payload = rng.randrange(BANKS)
            edges = payload + nodes + rng.randrange(BANKS)
            positions = edges + 2 * nodes + rng.randrange(BANKS)
            values = positions + batch + rng.randrange(BANKS)
            out_positions = values + batch + rng.randrange(BANKS)
            out_values = out_positions + batch + rng.randrange(BANKS)
            workloads.append(dict(family=family, nodes=nodes, rounds=rounds, batch=batch,
                                  payload=payload, edges=edges, positions=positions, values=values,
                                  out_positions=out_positions, out_values=out_values,
                                  memory=out_values + 3 * batch + 64))
        return {"workloads": tuple(workloads)}

    def solve(self, problem):
        return tuple(_compile(workload) for workload in problem["workloads"])

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def evaluate_pair(self, problem, outputs, *, replay=None, record=None):
        if type(outputs) is not dict or set(outputs) != {"reference", "candidate"}:
            raise ValueError("paired evaluation requires reference and candidate")
        return _evaluate_submissions(problem, outputs, replay=replay, record=record)

    def evaluate_solution(self, problem, proposed):
        return _evaluate_submissions(problem, {"candidate": proposed})["candidate"]

    def is_solution(self, problem, proposed):
        return self.evaluate_solution(problem, proposed).correct


TASK = SIMDTraversalKernelTask()
