"""Reference implementation for register_vm_execution; copied into fresh run candidates."""

MASK = 0xFFFFFFFF

(HALT, LI, MOV, ADD, SUB, MUL, XOR, AND, OR, ADDI, SHL, SHR, LT, EQ,
 LOAD, STORE, JMP, JZ, JNZ, CALL, RET) = range(21)


def _run(program):
    """Decode and execute one instruction at a time until HALT."""
    code = program["code"]
    registers = list(program["registers"])
    memory = list(program["memory"])
    size = len(memory)
    stack = []
    pc = 0
    while True:
        op, a, b, c = code[pc]
        pc += 1
        if op == HALT:
            return [registers, memory]
        elif op == LI:
            registers[a] = b
        elif op == MOV:
            registers[a] = registers[b]
        elif op == ADD:
            registers[a] = (registers[b] + registers[c]) & MASK
        elif op == SUB:
            registers[a] = (registers[b] - registers[c]) & MASK
        elif op == MUL:
            registers[a] = (registers[b] * registers[c]) & MASK
        elif op == XOR:
            registers[a] = registers[b] ^ registers[c]
        elif op == AND:
            registers[a] = registers[b] & registers[c]
        elif op == OR:
            registers[a] = registers[b] | registers[c]
        elif op == ADDI:
            registers[a] = (registers[b] + c) & MASK
        elif op == SHL:
            registers[a] = (registers[b] << c) & MASK
        elif op == SHR:
            registers[a] = registers[b] >> c
        elif op == LT:
            registers[a] = 1 if registers[b] < registers[c] else 0
        elif op == EQ:
            registers[a] = 1 if registers[b] == registers[c] else 0
        elif op == LOAD:
            registers[a] = memory[(registers[b] + c) % size]
        elif op == STORE:
            memory[(registers[b] + c) % size] = registers[a]
        elif op == JMP:
            pc = a
        elif op == JZ:
            if registers[a] == 0:
                pc = b
        elif op == JNZ:
            if registers[a] != 0:
                pc = b
        elif op == CALL:
            stack.append(pc)
            pc = a
        elif op == RET:
            pc = stack.pop()
        else:
            raise ValueError("unknown opcode")


def solve(problem):
    return {"states": [_run(program) for program in problem["programs"]]}
