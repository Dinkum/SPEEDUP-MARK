"""Reference implementation for simd_traversal_kernel; copied into fresh run candidates."""

MULTIPLIER = 2654435761


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


def solve(problem):
    return tuple(_compile(workload) for workload in problem["workloads"])
