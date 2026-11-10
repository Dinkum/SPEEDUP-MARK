"""Execute batches of register-machine programs; hot loops and cold code disagree.

The machine has sixteen 32-bit registers, a word-addressed memory, and a call
stack. One problem mixes long-running loop nests, data-dependent branching over
memory, call-heavy code, and thousands of short programs that each run once. A
translator that pays off on the first three loses its compile time on the last.
See README.md.
"""

import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)

MASK = 0xFFFFFFFF
REGISTERS = 16
(HALT, LI, MOV, ADD, SUB, MUL, XOR, AND, OR, ADDI, SHL, SHR, LT, EQ,
 LOAD, STORE, JMP, JZ, JNZ, CALL, RET) = range(21)

# Generated code keeps data in r0..r7. r8..r11 count loops by nesting depth,
# r12 holds branch conditions, and r13 is the recursion depth argument, so a
# random body can never clobber the state that guarantees termination.
DATA = 8
COUNTERS = (8, 9, 10, 11)
CONDITION, DEPTH = 12, 13


def _body(rng, code, length, memory_share):
    """Straight-line data instructions; returns the number emitted."""
    for _ in range(length):
        d, s, t = rng.randrange(DATA), rng.randrange(DATA), rng.randrange(DATA)
        roll = rng.random()
        if roll < memory_share / 2:
            code.append([LOAD, d, s, rng.randrange(64)])
        elif roll < memory_share:
            code.append([STORE, s, t, rng.randrange(64)])
        elif roll < memory_share + 0.2:
            # Fresh constants keep values from collapsing to zero under AND,
            # SHR and even multipliers over a long run.
            code.append([ADDI, d, s, rng.randrange(1, MASK + 1)])
        else:
            op = rng.choice((ADD, ADD, SUB, MUL, XOR, XOR, AND, OR, SHL, SHR, LT, EQ, MOV, LI))
            if op in (SHL, SHR):
                code.append([op, d, s, rng.randrange(1, 9)])
            elif op == MOV:
                code.append([MOV, d, s, 0])
            elif op == LI:
                code.append([LI, d, rng.randrange(MASK + 1), 0])
            else:
                code.append([op, d, s, t])
    return length


def _branch(rng, code, length, memory_share):
    """A block skipped when an unsigned comparison of two data registers fails."""
    code.append([LT, CONDITION, rng.randrange(DATA), rng.randrange(DATA)])
    jump = len(code)
    code.append([JZ, CONDITION, 0, 0])
    _body(rng, code, length, memory_share)
    code[jump][2] = len(code)


def _loop(code, depth, trips, emit):
    """Counted loop: the counter register is private to its nesting depth."""
    counter = COUNTERS[depth]
    code.append([LI, counter, trips, 0])
    top = len(code)
    emit()
    code.append([ADDI, counter, counter, MASK])
    code.append([JNZ, counter, top, 0])


def _program(rng, code, memory_words):
    registers = [rng.randrange(MASK + 1) for _ in range(DATA)] + [0] * (REGISTERS - DATA)
    memory = [rng.randrange(MASK + 1) for _ in range(memory_words)]
    return {"code": code, "registers": registers, "memory": memory}


def _hot_loops(rng, steps):
    """A three-deep loop nest of straight-line arithmetic with rare branches."""
    code = []
    inner = rng.randrange(8, 15)
    middle, outer = rng.randrange(6, 12), rng.randrange(3, 6)
    trips = max(2, steps // (outer * middle * (inner + 4)))

    def innermost():
        _body(rng, code, inner, 0.08)
        if rng.random() < 0.5:
            _branch(rng, code, 2, 0.0)

    def second():
        _body(rng, code, 3, 0.2)
        _loop(code, 2, trips, innermost)

    def first():
        _body(rng, code, 2, 0.2)
        _loop(code, 1, middle, second)

    _loop(code, 0, outer, first)
    code.append([HALT, 0, 0, 0])
    return _program(rng, code, 256)


def _branchy_memory(rng, steps):
    """One long loop whose path through several guarded blocks depends on data."""
    code = []
    blocks = rng.randrange(5, 9)
    # Roughly half of each guarded block runs, plus its two-instruction guard.
    per_trip = 6 + blocks * 6

    def body():
        _body(rng, code, 4, 0.5)
        for _ in range(blocks):
            _branch(rng, code, rng.randrange(5, 10), 0.4)
        _body(rng, code, 2, 0.5)

    _loop(code, 0, max(2, steps // per_trip), body)
    code.append([HALT, 0, 0, 0])
    return _program(rng, code, 1024)


def _call_heavy(rng, steps):
    """A driver loop calling leaf routines and one binary-recursive routine."""
    code, calls = [], []
    leaves = rng.randrange(3, 7)
    depth = rng.randrange(4, 7)
    # Each recursion node costs about ten instructions; leaves about eight.
    per_trip = leaves * 10 + (2 ** (depth + 1)) * 10

    def body():
        for leaf in range(leaves):
            _body(rng, code, 1, 0.0)
            calls.append((len(code), leaf))
            code.append([CALL, 0, 0, 0])
        code.append([LI, DEPTH, depth, 0])
        calls.append((len(code), leaves))
        code.append([CALL, 0, 0, 0])

    _loop(code, 0, max(2, steps // per_trip), body)
    code.append([HALT, 0, 0, 0])
    entries = []
    for _ in range(leaves):
        entries.append(len(code))
        _body(rng, code, rng.randrange(4, 9), 0.25)
        code.append([RET, 0, 0, 0])
    # recurse(depth): run a base block at depth zero, else recurse twice.
    entries.append(len(code))
    guard = len(code)
    code.append([JZ, DEPTH, 0, 0])
    code.append([ADDI, DEPTH, DEPTH, MASK])
    _body(rng, code, 2, 0.2)
    code.append([CALL, entries[-1], 0, 0])
    _body(rng, code, 2, 0.2)
    code.append([CALL, entries[-1], 0, 0])
    code.append([ADDI, DEPTH, DEPTH, 1])
    code.append([RET, 0, 0, 0])
    code[guard][2] = len(code)
    _body(rng, code, 3, 0.3)
    code.append([RET, 0, 0, 0])
    for position, target in calls:
        code[position][1] = entries[target]
    return _program(rng, code, 256)


def _cold(rng):
    """A short program run once: mostly straight-line, often with unreached code."""
    code = []
    _body(rng, code, rng.randrange(10, 40), 0.25)
    _loop(code, 0, rng.randrange(2, 6), lambda: _body(rng, code, rng.randrange(4, 12), 0.25))
    if rng.random() < 0.5:
        # A large block behind an unconditional jump is never executed.
        jump = len(code)
        code.append([JMP, 0, 0, 0])
        _body(rng, code, rng.randrange(60, 240), 0.25)
        code[jump][1] = len(code)
    _branch(rng, code, rng.randrange(4, 30), 0.25)
    _body(rng, code, rng.randrange(5, 20), 0.25)
    code.append([HALT, 0, 0, 0])
    return _program(rng, code, 16)


def _verify_run(program):
    """Table-dispatched interpreter, structured independently of the reference."""
    code = program["code"]
    registers, memory = list(program["registers"]), list(program["memory"])
    size, stack = len(memory), []
    state = [0]  # program counter, shared with the handlers

    def binary(function):
        def handler(a, b, c):
            registers[a] = function(registers[b], registers[c]) & MASK
        return handler

    def li(a, b, c): registers[a] = b
    def mov(a, b, c): registers[a] = registers[b]
    def addi(a, b, c): registers[a] = (registers[b] + c) % (MASK + 1)
    def shl(a, b, c): registers[a] = (registers[b] * 2 ** c) % (MASK + 1)
    def shr(a, b, c): registers[a] = registers[b] // 2 ** c
    def load(a, b, c): registers[a] = memory[(registers[b] + c) % size]
    def store(a, b, c): memory[(registers[b] + c) % size] = registers[a]
    def jmp(a, b, c): state[0] = a
    def jz(a, b, c): state[0] = b if registers[a] == 0 else state[0]
    def jnz(a, b, c): state[0] = b if registers[a] else state[0]
    def call(a, b, c): stack.append(state[0]); state[0] = a
    def ret(a, b, c): state[0] = stack.pop()

    table = {
        LI: li, MOV: mov, ADD: binary(lambda x, y: x + y), SUB: binary(lambda x, y: x - y),
        MUL: binary(lambda x, y: x * y), XOR: binary(lambda x, y: x ^ y),
        AND: binary(lambda x, y: x & y), OR: binary(lambda x, y: x | y), ADDI: addi,
        SHL: shl, SHR: shr, LT: binary(lambda x, y: int(x < y)),
        EQ: binary(lambda x, y: int(x == y)), LOAD: load, STORE: store, JMP: jmp,
        JZ: jz, JNZ: jnz, CALL: call, RET: ret,
    }
    while True:
        op, a, b, c = code[state[0]]
        if op == HALT:
            return [registers, memory]
        state[0] += 1
        table[op](a, b, c)


class RegisterVmExecutionTask:
    name = "register_vm_execution"
    task_version = "1.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 1500
    grading_cases = (1500, 3000)

    def generate_problem(self, n=1500, random_seed=0):
        """``n`` is the approximate number of executed instructions, in thousands."""
        if type(n) is not int or n < 8:
            raise ValueError("n must be an integer of at least 8 thousand instructions")
        rng = random.Random(random_seed)
        steps = n * 1000
        programs = []
        for build, share, count in ((_hot_loops, 0.30, 2), (_branchy_memory, 0.20, 2),
                                    (_call_heavy, 0.15, 2)):
            for _ in range(count):
                programs.append(build(rng, int(steps * share / count)))
        # Warm programs run a few thousand instructions each: near the point
        # where translating a program costs about what it saves.
        warm = int(steps * 0.15)
        while warm > 0:
            budget = rng.randrange(1000, 12000)
            programs.append(rng.choice((_hot_loops, _branchy_memory, _call_heavy))(rng, budget))
            warm -= budget
        # A cold program executes about 80 instructions; together they take
        # the remaining fifth of the work across thousands of distinct programs.
        programs.extend(_cold(rng) for _ in range(max(4, steps // 400)))
        rng.shuffle(programs)
        return {"programs": programs}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if (type(proposed) is not dict or set(proposed) != {"states"}
                or not plain_containers(proposed["states"])
                or type(proposed["states"]) is not list
                or len(proposed["states"]) != len(problem["programs"])):
            return False
        for program, state in zip(problem["programs"], proposed["states"]):
            if (type(state) is not list or len(state) != 2
                    or any(type(part) is not list for part in state)
                    or any(type(word) is not int for part in state for word in part)
                    or state != _verify_run(program)):
                return False
        return True


TASK = RegisterVmExecutionTask()
