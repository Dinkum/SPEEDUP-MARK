"""Compile a generated tensor pipeline into a schedule for a vector scratchpad.

The algorithm is fixed; the *organization* is not. A submission decides which
intermediates exist, where each one lives and in which layout, how the work is
tiled, what it recomputes instead of storing, and how transfers, elementwise
arithmetic, multiplies and shuffles are scheduled against each other -- the same
separation of algorithm from schedule that Halide exposes, with the choices made
by the candidate instead of by a compiler.

Correctness is defined by values, not by structure. The verifier executes the
submitted machine program, independently evaluates the pipeline from the
algorithm description, and then requires the declared output cells to hold
exactly the right values. Fusion, recomputation, in-place reuse, arbitrary
intermediate layouts and discarding intermediates are all legal, and the score
is the simulated cycle count. Compilation receives only shapes and addresses;
runtime tensor values are drawn after the submitted program is fixed.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
import struct

from speedupmark.task import SolutionEvaluation, load_candidate


_candidate = load_candidate(__file__)
MASK = (1 << 32) - 1
PAD = 4
MAX_KERNEL = 5

# Issue resource and latency per opcode. Every instruction completes
# ``latency`` cycles after it issues and is readable at the start of that cycle.
LATENCY = {"vload": 4, "vstore": 4, "sload": 8, "sstore": 8, "vconst": 1,
           "vadd": 1, "vmul": 3, "vmadd": 3, "vperm": 1, "vred": 5}
RESOURCE = {"vload": "load", "sload": "load", "vstore": "store", "sstore": "store",
            "vconst": "alu", "vadd": "alu", "vred": "alu", "vperm": "perm",
            "vmul": "mul", "vmadd": "mul"}
READS = {"vstore": (1,), "sstore": (1,), "vadd": (2, 3), "vmul": (2, 3),
         "vmadd": (2, 3, 4), "vperm": (2, 3), "vred": (2,)}
WRITES = {"vload": 1, "sload": 1, "vconst": 1, "vadd": 1, "vmul": 1, "vmadd": 1,
          "vperm": 1, "vred": 1}
ARITY = {"vload": 3, "vstore": 3, "sload": 4, "sstore": 4, "vconst": 3, "vadd": 4,
         "vmul": 4, "vmadd": 5, "vperm": 5, "vred": 3}


def _tensor_shape(tensor):
    if tensor and isinstance(tensor[0], (tuple, list)):
        return len(tensor), len(tensor[0])
    return 1, len(tensor)


def _shapes(workload):
    """Shape of every named tensor, derived from inputs and the pipeline order."""
    shapes = {name: shape for name, shape, _ in workload["inputs"]}
    for name, expression in workload["pipeline"]:
        kind = expression[0]
        if kind == "ew":
            shapes[name] = shapes[expression[2]]
        elif kind == "stencil":
            shapes[name] = shapes[expression[1]]
        elif kind == "transpose":
            rows, columns = shapes[expression[1]]
            shapes[name] = (columns, rows)
        elif kind == "reduce":
            rows, _ = shapes[expression[1]]
            shapes[name] = (rows, 1)
        else:
            raise ValueError(f"unsupported pipeline stage {kind!r}")
    return shapes


def _row_stride(shape):
    """Storage convention: every 2-D row is followed by ``PAD`` zero cells.

    The zero padding is what makes a stencil tap that runs past the end of a row
    read a zero instead of the next row, exactly as the algorithm specifies.
    """
    rows, columns = shape
    return columns + PAD if columns > 1 else 1


def _cell_address(base, row, column, row_stride, column_stride):
    """The one address formula for every declared output cell."""
    return base + row * row_stride + column * column_stride


def _evaluate(workload, inputs):
    """The algorithm itself: full tensors, in order, unsigned 32-bit wrapping."""
    values = dict(inputs)
    for name, expression in workload["pipeline"]:
        kind = expression[0]
        if kind == "ew":
            _, operator, left_name, right_name = expression
            left, right = values[left_name], values[right_name]
            values[name] = tuple(
                tuple(
                    (left[row][column] + right[row][column]) & MASK if operator == "add"
                    else (left[row][column] * right[row][column]) & MASK
                    for column in range(len(left[row]))
                )
                for row in range(len(left))
            )
        elif kind == "stencil":
            _, source_name, kernel = expression
            source = values[source_name]
            width = len(kernel)
            values[name] = tuple(
                tuple(
                    sum(kernel[tap] * (row[column + tap] if column + tap < len(row) else 0)
                        for tap in range(width)) & MASK
                    for column in range(len(row))
                )
                for row in source
            )
        elif kind == "transpose":
            source = values[expression[1]]
            values[name] = tuple(
                tuple(source[row][column] for row in range(len(source)))
                for column in range(len(source[0]))
            )
        elif kind == "reduce":
            source = values[expression[1]]
            values[name] = tuple((sum(row) & MASK,) for row in source)
        else:
            raise ValueError(f"unsupported pipeline stage {kind!r}")
    return values


def _flatten_padded(tensor):
    """Flat memory image of one tensor in the padded row-major convention."""
    rows, columns = _tensor_shape(tensor)
    stride = _row_stride((rows, columns))
    image = [0] * (rows * stride)
    for row in range(rows):
        for column in range(columns):
            image[row * stride + column] = (tensor[row][column]
                                            if isinstance(tensor[0], (tuple, list))
                                            else (tensor[row] if column == 0 else 0))
    return image


class _Simulator:
    """Executes a submitted program; ``run`` returns the final cycle count."""

    def __init__(self, workload, inputs):
        machine = workload["machine"]
        self.lanes = machine["lanes"]
        self.slots = machine["scratch"]
        self.banks = machine["banks"]
        self.machine = machine
        self.capacity = machine["memory"]
        self.registers = [[0] * self.lanes for _ in range(self.slots)]
        self.busy = [0] * self.slots
        self.memory = [0] * self.capacity
        self.pending = {}
        self.last = 0
        for name, _, address in workload["inputs"]:
            image = _flatten_padded(inputs[name])
            if address < 0 or address + len(image) > self.capacity:
                raise ValueError("declared input region is out of range")
            self.memory[address:address + len(image)] = image
        self.shapes = _shapes(workload)

    def run(self, program):
        total_cells = sum(_row_stride(shape) * shape[0] for shape in self.shapes.values())
        bound = 32 * total_cells + 32
        if type(program) not in (tuple, list) or not 0 < len(program) <= bound:
            raise ValueError("invalid program length")
        issued = 0
        for cycle, packet in enumerate(program):
            for target, index, lane, value in self.pending.pop(cycle, ()):
                if target == "scratch":
                    self.registers[index][lane] = value
                else:
                    self.memory[index] = value
            if type(packet) not in (tuple, list):
                raise ValueError("a packet must be a sequence of instructions")
            used = {}
            banks_used = set()
            written = set()
            for instruction in packet:
                issued += 1
                if issued > bound:
                    raise ValueError("program exceeds the instruction budget")
                completion = self._issue(instruction, cycle, used, banks_used, written)
                self.last = max(self.last, completion)
        # Writes still in flight when the program ends are part of the machine
        # state: a store scheduled past the last packet still lands. Dropping them
        # would silently reject a correct program whose tail is unrolled.
        for cycle in sorted(self.pending):
            for target, index, lane, value in self.pending[cycle]:
                if target == "scratch":
                    self.registers[index][lane] = value
                else:
                    self.memory[index] = value
        return max(len(program), self.last)

    def _issue(self, instruction, cycle, used, banks_used, written):
        if type(instruction) not in (tuple, list) or not instruction:
            raise ValueError("an instruction must be a non-empty tuple")
        opcode = instruction[0]
        if not isinstance(opcode, str) or opcode not in ARITY:
            raise ValueError(f"unknown opcode {opcode!r}")
        if len(instruction) != ARITY[opcode]:
            raise ValueError(f"{opcode} has the wrong number of fields")
        if any(type(field) is not int for field in instruction[1:]
               if not isinstance(field, (tuple, list))):
            raise ValueError("instruction fields must be exact integers")
        resource = RESOURCE[opcode]
        used[resource] = used.get(resource, 0) + 1
        if used[resource] > self.machine[resource]:
            raise ValueError(f"{resource} issue capacity exceeded")
        for slot in READS.get(opcode, ()):
            self._readable(instruction[slot], cycle)
        bank = self._bank(opcode, instruction)
        if bank is not None:
            if bank in banks_used:
                raise ValueError("memory bank conflict in one packet")
            banks_used.add(bank)
        value = self._compute(opcode, instruction, cycle)
        completion = cycle + LATENCY[opcode]
        destination = WRITES.get(opcode)
        if destination is not None:
            slot = instruction[WRITES[opcode]]
            if slot in written:
                raise ValueError("two instructions write one slot in a packet")
            if self.busy[slot] > cycle:
                raise ValueError("destination slot is still pending")
            written.add(slot)
            self.busy[slot] = completion
            self.pending.setdefault(completion, []).extend(
                ("scratch", slot, lane, value[lane]) for lane in range(self.lanes))
        if opcode == "vstore":
            address = instruction[2]
            for lane in range(self.lanes):
                self.pending.setdefault(completion, []).append(
                    ("memory", address + lane, 0, value[lane]))
        elif opcode == "sstore":
            self.pending.setdefault(completion, []).append(
                ("memory", instruction[3], 0, value[instruction[2]]))
        return completion

    def _readable(self, slot, cycle):
        if type(slot) is not int or not 0 <= slot < self.slots:
            raise ValueError("scratch slot out of range")
        if self.busy[slot] > cycle:
            raise ValueError("slot read before its producer completed")
        return self.registers[slot]

    def _bank(self, opcode, instruction):
        if opcode in ("vload", "vstore"):
            address = instruction[2]
            if address < 0 or address + self.lanes > self.capacity:
                raise ValueError("vector transfer outside memory")
            return (address // self.lanes) % self.banks
        if opcode in ("sload", "sstore"):
            address = instruction[3]
            if not 0 <= address < self.capacity:
                raise ValueError("scalar transfer outside memory")
            return (address // self.lanes) % self.banks
        return None

    def _compute(self, opcode, instruction, cycle):
        if opcode == "vload":
            address = instruction[2]
            return tuple(self.memory[address + lane] for lane in range(self.lanes))
        if opcode == "vstore":
            return tuple(self._readable(instruction[1], cycle))
        if opcode == "sload":
            lane = instruction[2]
            if type(lane) is not int or not 0 <= lane < self.lanes:
                raise ValueError("lane index out of range")
            value = self.memory[instruction[3]]
            current = self._readable(instruction[1], cycle)
            return [value if position == lane else current[position]
                    for position in range(self.lanes)]
        if opcode == "sstore":
            lane = instruction[2]
            if type(lane) is not int or not 0 <= lane < self.lanes:
                raise ValueError("lane index out of range")
            return self._readable(instruction[1], cycle)
        if opcode == "vconst":
            value = instruction[2]
            if not 0 <= value <= MASK:
                raise ValueError("constant must fit in an unsigned 32-bit cell")
            return tuple([value] * self.lanes)
        if opcode in ("vadd", "vmul", "vmadd"):
            operands = [self._readable(instruction[index], cycle)
                        for index in READS[opcode]]
            if opcode == "vadd":
                return tuple((operands[0][lane] + operands[1][lane]) & MASK
                             for lane in range(self.lanes))
            if opcode == "vmul":
                return tuple((operands[0][lane] * operands[1][lane]) & MASK
                             for lane in range(self.lanes))
            return tuple((operands[0][lane] * operands[1][lane] + operands[2][lane]) & MASK
                         for lane in range(self.lanes))
        if opcode == "vperm":
            left = self._readable(instruction[2], cycle)
            right = self._readable(instruction[3], cycle)
            mask = instruction[4]
            if type(mask) not in (tuple, list) or len(mask) != self.lanes:
                raise ValueError("vperm mask must have one entry per lane")
            if any(type(index) is not int or not 0 <= index < 2 * self.lanes
                   for index in mask):
                raise ValueError("vperm mask entries must be lane indexes")
            return tuple(left[index] if index < self.lanes else right[index - self.lanes]
                         for index in mask)
        if opcode == "vred":
            source = self._readable(instruction[2], cycle)
            total = 0
            for lane in range(self.lanes):
                total = (total + source[lane]) & MASK
            return tuple([total] + [0] * (self.lanes - 1))
        raise ValueError(f"unknown opcode {opcode!r}")


def _placements(workload, proposed):
    """Validate declared output layouts and return them as plain tuples."""
    outputs = tuple(workload["outputs"])
    if type(proposed) is not dict or set(proposed) != set(outputs):
        raise ValueError("placements must cover exactly the declared outputs")
    shapes = _shapes(workload)
    capacity = workload["machine"]["memory"]
    claimed = {}
    cleaned = {}
    for name in outputs:
        entry = proposed[name]
        if type(entry) not in (tuple, list) or len(entry) != 3:
            raise ValueError("a placement is (base, row_stride, column_stride)")
        base, row_stride, column_stride = entry
        if (type(base) is not int or type(row_stride) is not int
                or type(column_stride) is not int):
            raise ValueError("placement fields must be exact integers")
        rows, columns = shapes[name]
        # Actual cell addresses decide validity: stride comparisons would reject
        # non-overlapping column-major or reversed layouts.
        for row in range(rows):
            for column in range(columns):
                address = _cell_address(base, row, column, row_stride, column_stride)
                if not 0 <= address < capacity:
                    raise ValueError("declared output cell is outside memory")
                if address in claimed:
                    raise ValueError("output regions overlap")
                claimed[address] = name
        cleaned[name] = (base, row_stride, column_stride)
    return cleaned


def _check_outputs(workload, inputs, placements, simulator):
    """Compare every declared output cell against the independently evaluated pipeline."""
    expected = _evaluate(workload, inputs)
    shapes = _shapes(workload)
    for name in workload["outputs"]:
        base, row_stride, column_stride = placements[name]
        tensor = expected[name]
        rows, columns = shapes[name]
        nested = isinstance(tensor[0], (tuple, list))
        for row in range(rows):
            for column in range(columns):
                address = _cell_address(base, row, column, row_stride, column_stride)
                value = tensor[row][column] if nested else tensor[row]
                if simulator.memory[address] != value:
                    raise ValueError(f"output {name!r} is wrong at {row},{column}")


class _Emitter:
    """Scoreboarding emitter used by the reference compiler.

    Every instruction is placed at the earliest cycle where its operands are
    ready and its issue resource and memory bank are free, mirroring what the
    verifier will execute.
    """

    def __init__(self, machine):
        self.machine = machine
        self.lanes = machine["lanes"]
        self.banks = machine["banks"]
        self.capacity = machine["memory"]
        self.packets = []
        self.resource_use = []
        self.bank_use = []
        self.slot_ready = [0] * machine["scratch"]
        self.memory_ready = [0] * self.capacity
        # The machine issues in order: an instruction may share a cycle with the
        # one before it, but never issue earlier. Without this the emitter could
        # place a write to a slot in a cycle the verifier reaches only after an
        # earlier-emitted reader of that same slot, which the hardware rejects.
        self.cursor = 0

    def issue(self, instruction):
        opcode = instruction[0]
        dependencies = []
        bank = None
        if opcode in ("vload", "vstore"):
            address = instruction[2]
            if address < 0 or address + self.lanes > self.capacity:
                raise ValueError("reference program left memory")
            dependencies.append(max(self.memory_ready[address:address + self.lanes]))
            bank = (address // self.lanes) % self.banks
        elif opcode in ("sload", "sstore"):
            address = instruction[3]
            if not 0 <= address < self.capacity:
                raise ValueError("reference program left memory")
            dependencies.append(self.memory_ready[address])
            bank = (address // self.lanes) % self.banks
        for slot in READS.get(opcode, ()):
            dependencies.append(self.slot_ready[instruction[slot]])
        writes = WRITES.get(opcode)
        if writes is not None:
            dependencies.append(self.slot_ready[instruction[writes]])
        cursor = max(dependencies) if dependencies else 0
        cursor = max(cursor, self.cursor)
        resource = RESOURCE[opcode]
        while True:
            while len(self.packets) <= cursor:
                self.packets.append([])
                self.resource_use.append({})
                self.bank_use.append(set())
            if (self.resource_use[cursor].get(resource, 0) < self.machine[resource]
                    and (bank is None or bank not in self.bank_use[cursor])):
                break
            cursor += 1
        self.cursor = cursor
        self.packets[cursor].append(instruction)
        self.resource_use[cursor][resource] = self.resource_use[cursor].get(resource, 0) + 1
        if bank is not None:
            self.bank_use[cursor].add(bank)
        completion = cursor + LATENCY[opcode]
        if writes is not None:
            self.slot_ready[instruction[writes]] = completion
        if opcode == "vstore":
            for lane in range(self.lanes):
                self.memory_ready[instruction[2] + lane] = completion
        elif opcode == "sstore":
            self.memory_ready[instruction[3]] = completion
        return cursor

    def program(self):
        return tuple(tuple(packet) for packet in self.packets) or ((),)


def _compile(workload):
    """Reference compiler: one stage at a time, every intermediate materialized.

    It is deliberately unsophisticated but never absurd. It allocates a padded
    row-major region per stage, materializes every intermediate even when nobody
    consumes it, reloads a shifted window for every stencil tap, and moves single
    cells for a transpose. Fusion, recomputation, layout changes, in-place reuse
    and shuffle networks are all left on the table.
    """
    machine = workload["machine"]
    lanes = machine["lanes"]
    shapes = _shapes(workload)
    emitter = _Emitter(machine)
    cursor = 0
    addresses = {}
    placements = {}
    for name, shape, address in workload["inputs"]:
        addresses[name] = address
        cursor = max(cursor, address + shape[0] * _row_stride(shape))

    def allocate(name):
        nonlocal cursor
        rows, columns = shapes[name]
        cells = rows * _row_stride((rows, columns))
        base, cursor = cursor, cursor + cells
        addresses[name] = base
        return base, rows, columns

    for name, expression in workload["pipeline"]:
        base, rows, columns = allocate(name)
        stride = _row_stride((rows, columns))
        kind = expression[0]
        if kind == "ew":
            _, operator, left_name, right_name = expression
            opcode = "vadd" if operator == "add" else "vmul"
            for row in range(rows):
                if columns == 1:
                    emitter.issue(("sload", 0, 0, addresses[left_name] + row))
                    emitter.issue(("sload", 1, 0, addresses[right_name] + row))
                    emitter.issue((opcode, 2, 0, 1))
                    emitter.issue(("sstore", 2, 0, base + row))
                    continue
                for offset in range(0, columns, lanes):
                    emitter.issue(("vload", 0, addresses[left_name] + row * _row_stride(shapes[left_name]) + offset))
                    emitter.issue(("vload", 1, addresses[right_name] + row * _row_stride(shapes[right_name]) + offset))
                    emitter.issue((opcode, 2, 0, 1))
                    emitter.issue(("vstore", 2, base + row * stride + offset))
        elif kind == "stencil":
            _, source_name, kernel = expression
            width = len(kernel)
            source_stride = _row_stride(shapes[source_name])
            for row in range(rows):
                if columns == 1:
                    # Every later tap is outside the one-cell row and reads zero.
                    emitter.issue(("sload", 0, 0, addresses[source_name] + row))
                    emitter.issue(("vconst", 1, kernel[0] & MASK))
                    emitter.issue(("vmul", 2, 0, 1))
                    emitter.issue(("sstore", 2, 0, base + row))
                    continue
                for offset in range(0, columns, lanes):
                    # One reload per tap: the window slides but the reference
                    # never reuses it, so each tap costs a full vector transfer.
                    for tap in range(width):
                        emitter.issue(("vload", 0, addresses[source_name] + row * source_stride + offset + tap))
                        emitter.issue(("vconst", 1, kernel[tap] & MASK))
                        if tap == 0:
                            emitter.issue(("vmul", 2, 0, 1))
                        else:
                            emitter.issue(("vmadd", 2, 0, 1, 2))
                    emitter.issue(("vstore", 2, base + row * stride + offset))
        elif kind == "transpose":
            source_name = expression[1]
            source_stride = _row_stride(shapes[source_name])
            for row in range(rows):
                for column in range(columns):
                    emitter.issue(("sload", 0, 0, addresses[source_name] + row + column * source_stride))
                    emitter.issue(("sstore", 0, 0, base + row * stride + column))
        elif kind == "reduce":
            source_name = expression[1]
            source_columns = shapes[source_name][1]
            source_stride = _row_stride(shapes[source_name])
            for row in range(rows):
                if source_columns == 1:
                    emitter.issue(("sload", 0, 0, addresses[source_name] + row))
                    emitter.issue(("sstore", 0, 0, base + row))
                    continue
                emitter.issue(("vload", 0, addresses[source_name] + row * source_stride))
                for vector in range(1, (source_columns + lanes - 1) // lanes):
                    emitter.issue(("vload", 1, addresses[source_name] + row * source_stride + vector * lanes))
                    emitter.issue(("vadd", 0, 0, 1))
                emitter.issue(("vred", 2, 0))
                emitter.issue(("sstore", 2, 0, base + row * stride))
        else:
            raise ValueError(f"unsupported pipeline stage {kind!r}")
        if name in workload["outputs"]:
            placements[name] = (base, stride, 1)
    return {"program": emitter.program(), "placements": placements}


def _check_pipeline(workload):
    """Fail loudly if a generated workload breaks its own documented contract."""
    lanes = workload["machine"]["lanes"]
    shapes = _shapes(workload)
    consumed = set()
    for _, expression in workload["pipeline"]:
        kind = expression[0]
        if kind == "ew":
            consumed.update((expression[2], expression[3]))
        elif kind in ("stencil", "transpose", "reduce"):
            consumed.add(expression[1])
        elif kind != "ew":
            raise ValueError(f"unsupported pipeline stage {kind!r}")
    for name, expression in workload["pipeline"]:
        if name not in consumed and name not in workload["outputs"]:
            raise ValueError(f"stage {name!r} is neither an output nor consumed")
        if expression[0] == "stencil":
            kernel = expression[2]
            if not kernel or len(kernel) > MAX_KERNEL:
                raise ValueError("stencil kernels must hold one to five taps")
    for name, shape in shapes.items():
        rows, columns = shape
        # A single-column tensor is a stack of scalars and needs no vector tiling.
        if rows < 1 or columns < 1 or (columns > 1 and columns % lanes):
            raise ValueError(f"tensor {name!r} does not tile by the vector width")


def _freeze_submission(value):
    """Copy plain program data; never run candidate callbacks on runtime inputs."""
    if type(value) in (int, str):
        return value
    if type(value) in (tuple, list):
        return tuple(_freeze_submission(item) for item in value)
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _freeze_submission(item) for key, item in value.items()}
    raise ValueError("submissions must contain only plain containers, strings and integers")


VERIFICATION_GENERATOR = "shake256-uint32be-v1"
VERIFICATION_TRIALS = 3


def _digest(value):
    """Hash canonical plain program/problem data, including output placements."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_inputs(workload, seed, workload_index, trial):
    """Expand a private 256-bit seed into a portable, versioned uint32 stream."""
    cells = sum(rows * columns for _, (rows, columns), _ in workload["inputs"])
    domain = b"speedmark/compiler/shake256-uint32be-v1\0"
    stream = hashlib.shake_256(domain + bytes.fromhex(seed) +
                              struct.pack(">II", workload_index, trial)).digest(cells * 4)
    values = iter(struct.unpack(f">{cells}I", stream))
    return {name: tuple(tuple(next(values) for _ in range(columns))
                        for _ in range(rows))
            for name, (rows, columns), _ in workload["inputs"]}


def _verify_frozen(problem, proposed, runtime_inputs):
    """Check one fixed program on the shared challenge, counting its schedule once."""
    try:
        if type(proposed) is not tuple or len(proposed) != len(problem["workloads"]):
            raise ValueError("one submission per workload is required")
        cycles = 0
        for workload, submission, trials in zip(problem["workloads"], proposed, runtime_inputs):
            if type(submission) is not dict or set(submission) != {"program", "placements"}:
                raise ValueError("a submission is {program, placements}")
            placements = _placements(workload, submission["placements"])
            for trial, inputs in enumerate(trials):
                simulator = _Simulator(workload, inputs)
                cost = simulator.run(submission["program"])
                _check_outputs(workload, inputs, placements, simulator)
                # The ISA has no data-dependent branches or latencies.
                if trial == 0:
                    cycles += cost
        return SolutionEvaluation(float(cycles), True)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError):
        return SolutionEvaluation(0.0, False)


def _evaluate_submissions(problem, outputs, *, replay=None, record=None):
    frozen = {}
    for role, output in outputs.items():
        try:
            frozen[role] = _freeze_submission(output)
        except (ValueError, TypeError, RecursionError):
            frozen[role] = None
    # A malformed single submission needs no correctness challenge. In paired
    # grading a malformed candidate must still leave reference verification intact.
    if not any(output is not None for output in frozen.values()) and replay is None and record is None:
        return {role: SolutionEvaluation(0.0, False) for role in frozen}
    receipt = {
        "generator_version": VERIFICATION_GENERATOR,
        "problem_sha256": _digest(problem),
        "program_sha256": {role: _digest(output) if output is not None else None
                           for role, output in frozen.items()},
    }
    if replay is not None:
        if type(replay) is not dict or set(replay) != set(receipt) | {"seed", "tensor_sha256"}:
            raise ValueError("invalid compiler verification receipt")
        if any(replay[key] != value for key, value in receipt.items()):
            raise ValueError("compiler replay requires the same problem, frozen programs and generator")
        seed = replay["seed"]
        if (type(seed) is not str or len(seed) != 64 or
                any(character not in "0123456789abcdef" for character in seed)):
            raise ValueError("compiler verification seed must be 256-bit lowercase hexadecimal")
    else:
        # Both outputs have been fully consumed before entropy is sampled. The
        # public workload seed never determines this independent challenge.
        seed = secrets.token_hex(32)
    runtime_inputs = tuple(tuple(_runtime_inputs(workload, seed, index, trial)
                                for trial in range(VERIFICATION_TRIALS))
                           for index, workload in enumerate(problem["workloads"]))
    receipt.update(seed=seed, tensor_sha256=_digest(runtime_inputs))
    if replay is not None and replay["tensor_sha256"] != receipt["tensor_sha256"]:
        raise ValueError("compiler replay tensor digest does not match")
    # Persist evidence before simulation, including a failed program's challenge.
    # The harness binds this receipt to candidate bytes and the task revision.
    if record is not None:
        record(receipt)
    return {role: _verify_frozen(problem, output, runtime_inputs)
            for role, output in frozen.items()}


class LayoutAwarePipelineCompilerTask:
    name = "layout_aware_pipeline_compiler"
    task_version = "2.0.2"
    display_name = "Layout-Aware Pipeline Compiler"
    default_n = 32
    grading_cases = (32, 48)
    metric_unit = "cycles"
    lanes = 8

    def generate_problem(self, n=32, random_seed=0):
        if n < self.lanes:
            raise ValueError(f"n must be at least {self.lanes}")
        rng = random.Random(random_seed)
        columns = rng.choice((1, 2, 3)) * self.lanes
        rows = (n // self.lanes) * self.lanes
        workloads = (
            self._chain(rng, rows, columns),
            self._shared_layout(rng, rows, columns),
            self._pressure(rng, rows, columns),
            self._bandwidth(rng, rows, columns),
        )
        for workload in workloads:
            _check_pipeline(workload)
        return {"workloads": workloads}

    def _workload(self, family, scratch, banks, slack, inputs, pipeline, outputs):
        """Assemble one workload, laying inputs out padded from address zero."""
        addresses = []
        cursor = 0
        for name, shape in inputs:
            addresses.append((name, shape, cursor))
            cursor += shape[0] * _row_stride(shape)
        shapes = _shapes({"inputs": addresses, "pipeline": pipeline})
        for stage, _ in pipeline:
            rows, columns = shapes[stage]
            cursor += rows * _row_stride((rows, columns))
        machine = {"lanes": self.lanes, "banks": banks, "scratch": scratch,
                   "alu": 2, "mul": 1, "perm": 1, "load": 2, "store": 1,
                   "memory": cursor + int(cursor * slack)}
        return {"family": family, "inputs": tuple(addresses), "pipeline": pipeline,
                "outputs": outputs, "machine": machine}

    def _chain(self, rng, rows, columns):
        pipeline = []
        source = "x"
        for index in range(rng.randrange(3, 7)):
            name = f"chain{index}"
            pipeline.append((name, ("ew", rng.choice(("add", "mul")), source, rng.choice(("x", "y")))))
            source = name
        pipeline.extend((("window", ("stencil", source, self._kernel(rng))),
                         ("result", ("ew", rng.choice(("add", "mul")), "window", "window"))))
        return self._workload(
            "chain", rng.choice((8, 12)), 4, 0.25,
            (("x", (rows, columns)),
             ("y", (rows, columns))),
            tuple(pipeline), ("result",))

    def _shared_layout(self, rng, rows, columns):
        pipeline = [("shared", ("stencil", "x", self._kernel(rng)))]
        source = "shared"
        for index in range(rng.randrange(1, 4)):
            name = f"row{index}"
            pipeline.append((name, ("ew", rng.choice(("add", "mul")), source,
                                   rng.choice(("x", "shared")))))
            source = name
        row_output = source
        pipeline.append(("flipped", ("transpose", "shared")))
        source = "flipped"
        for index in range(rng.randrange(1, 4)):
            name = f"column{index}"
            expression = (("stencil", source, self._kernel(rng)) if rng.random() < 0.5
                          else ("ew", rng.choice(("add", "mul")), source, "flipped"))
            pipeline.append((name, expression))
            source = name
        return self._workload(
            "shared_layout", rng.choice((8, 16)), 4, 0.25,
            (("x", (rows, columns)),),
            tuple(pipeline), (row_output, source))

    def _pressure(self, rng, rows, columns):
        pipeline = [("root", ("ew", "mul", "x", "y")),
                    ("shared", ("stencil", "root", self._kernel(rng)))]
        branches = []
        for index in range(rng.randrange(2, 5)):
            name = f"branch{index}"
            source = rng.choice(("shared", "root")) if index else "shared"
            pipeline.append((name, ("stencil", source, self._kernel(rng))))
            branches.append(name)
        source = branches[0]
        for index, branch in enumerate(branches[1:]):
            name = f"join{index}"
            pipeline.append((name, ("ew", rng.choice(("add", "mul")), source, branch)))
            source = name
        return self._workload(
            "pressure", rng.choice((4, 6)), rng.choice((2, 4)), 0.05,
            (("x", (rows, columns)),
             ("y", (rows, columns))),
            tuple(pipeline), (source,))

    def _bandwidth(self, rng, rows, columns):
        pipeline = []
        source = "x"
        stages = []
        for index in range(rng.randrange(2, 5)):
            name = f"window{index}"
            pipeline.append((name, ("stencil", source, self._kernel(rng))))
            stages.append(name)
            source = name
        pipeline.extend((("total", ("reduce", rng.choice(stages))),
                         ("result", ("ew", rng.choice(("add", "mul")), source, "y"))))
        return self._workload(
            "bandwidth", 4, 2, 0.0,
            (("x", (rows, columns)),
             ("y", (rows, columns))),
            tuple(pipeline), ("result", "total"))

    @staticmethod
    def _kernel(rng):
        return tuple(rng.choice((-2, -1, 1, 2, 3)) for _ in range(rng.randrange(2, 6)))

    def solve(self, problem):
        return tuple(_compile(workload) for workload in problem["workloads"])

    def candidate_solve(self, problem):
        return _candidate.solve(problem, self.solve)

    def evaluate_pair(self, problem, outputs, *, replay=None, record=None):
        """Freeze both compiled programs, then verify on one recorded challenge."""
        if type(outputs) is not dict or set(outputs) != {"reference", "candidate"}:
            raise ValueError("paired verification requires reference and candidate outputs")
        return _evaluate_submissions(problem, outputs, replay=replay, record=record)

    def evaluate_solution(self, problem, proposed):
        return _evaluate_submissions(problem, {"candidate": proposed})["candidate"]

    def is_solution(self, problem, proposed):
        return self.evaluate_solution(problem, proposed).correct


TASK = LayoutAwarePipelineCompilerTask()
