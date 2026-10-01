"""Hand-written calibration schedules, independent of the task's compiler."""


def compile_workload(workload, *, lanes=8, streams=1, cached=False, scheduled=False,
                     compact=False):
    """Vectorize, interleave a bounded tile, and optionally cache the small table."""
    if lanes not in (1, 8) or not 1 <= streams <= (5 if compact else 4):
        raise ValueError("calibration stream count exceeds the register budget")
    cached = cached and workload["nodes"] == 8
    # Three cached table vectors leave room for three ordinary or four compact streams.
    if cached:
        streams = min(streams, 4 if compact else 3)
    program = []
    if cached:
        program.extend((("load", 8, 17, workload["payload"]),
                        ("load", 8, 18, workload["edges"]),
                        ("load", 8, 19, workload["edges"] + 8)))
    for tile in range(0, workload["batch"], lanes * streams):
        groups = []
        for stream in range(streams):
            item = tile + stream * lanes
            if item >= workload["batch"]:
                break
            width = min(lanes, workload["batch"] - item)
            if compact:
                node, value, mixed, edge = range(4 * stream, 4 * stream + 4)
                rotated = node
            else:
                node, value, mixed, rotated, edge = range(5 * stream, 5 * stream + 5)
            program.extend((("load", width, node, workload["positions"] + item),
                            ("load", width, value, workload["values"] + item)))
            steps = [
                (("pick", width, mixed, 17, 17, node) if cached
                 else ("gather", width, mixed, workload["payload"], node)),
                ("rotli", width, rotated, value, 7),
                ("xor", width, mixed, value, mixed),
                ("muli", width, mixed, mixed, 2654435761),
                ("add", width, value, mixed, rotated),
                ("shri", width, rotated, value, 31),
                ("muli", width, edge, node, 2),
                ("add", width, edge, edge, rotated),
                (("pick", width, node, 18, 19, edge) if cached
                 else ("gather", width, node, workload["edges"], edge)),
            ]
            if compact:
                # Preserve 2*old_node before recycling node for the rotation,
                # then the branch bit, and finally the next node.
                steps.insert(1, steps.pop(6))
            elif scheduled:
                steps.insert(2, steps.pop(6))
            groups.append((item, width, node, value, steps))
        for _ in range(workload["rounds"]):
            for stage in range(9):
                for _, _, _, _, steps in groups:
                    program.append(steps[stage])
        for item, width, node, value, _ in groups:
            program.extend((("store", width, node, workload["out_positions"] + item),
                            ("store", width, value, workload["out_values"] + item)))
    return tuple(program)
