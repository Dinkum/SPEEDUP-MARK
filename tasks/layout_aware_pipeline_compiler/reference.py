"""Reference implementation for layout_aware_pipeline_compiler; copied into fresh run candidates."""

from __future__ import annotations

MASK = (1 << 32) - 1


PAD = 4


LATENCY = {"vload": 4, "vstore": 4, "sload": 8, "sstore": 8, "vconst": 1,
           "vadd": 1, "vmul": 3, "vmadd": 3, "vperm": 1, "vred": 5}


RESOURCE = {"vload": "load", "sload": "load", "vstore": "store", "sstore": "store",
            "vconst": "alu", "vadd": "alu", "vred": "alu", "vperm": "perm",
            "vmul": "mul", "vmadd": "mul"}


READS = {"vstore": (1,), "sstore": (1,), "vadd": (2, 3), "vmul": (2, 3),
         "vmadd": (2, 3, 4), "vperm": (2, 3), "vred": (2,)}


WRITES = {"vload": 1, "sload": 1, "vconst": 1, "vadd": 1, "vmul": 1, "vmadd": 1,
          "vperm": 1, "vred": 1}


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


def solve(problem):
    return tuple(_compile(workload) for workload in problem["workloads"])
