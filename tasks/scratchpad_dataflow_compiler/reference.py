"""Reference implementation for scratchpad_dataflow_compiler; copied into fresh run candidates."""

def _resource(op):
    if op in ("load", "store"):
        return op
    return "mul" if op in ("mul", "madd") else "alu"


def _latency(op):
    return {"load": 2, "store": 2, "mul": 3, "madd": 3, "mul_alu": 5}.get(op, 1)


def _schedule(workload):
    """Topological register caching with farthest-next-use spills and scoreboarding."""
    nodes, inputs = workload["nodes"], workload["inputs"]
    machine = workload["machine"]
    needed = set(workload["outputs"])
    stack = list(needed)
    while stack:
        node = stack.pop()
        if nodes[node] is not None:
            for source in nodes[node][1:]:
                if source not in needed:
                    needed.add(source)
                    stack.append(source)
    order = sorted(node for node in needed if node >= len(inputs))
    uses = {node: [] for node in needed}
    for step, node in enumerate(order):
        for source in nodes[node][1:]:
            uses[source].append(step)
    positions = dict.fromkeys(needed, 0)
    residents = {}
    occupants = [None] * machine["scratch_slots"]
    slot_ready = [0] * len(occupants)
    memory_ready = dict.fromkeys(range(len(inputs)), 0)
    packets, resources, banks = [], [], []
    cursor = 0

    def issue(instruction):
        nonlocal cursor
        op = instruction[0]
        if op == "load":
            dependencies = [memory_ready[instruction[1]], slot_ready[instruction[2]]]
        elif op == "store":
            dependencies = [slot_ready[instruction[1]]]
        else:
            dependencies = [slot_ready[index] for index in instruction[2:]]
        cursor = max(cursor, *dependencies)
        resource = _resource(op)
        address = instruction[1] if op == "load" else instruction[2]
        bank = address % machine["banks"] if op in ("load", "store") else None
        while True:
            while len(packets) <= cursor:
                packets.append([])
                resources.append({})
                banks.append(set())
            if resources[cursor].get(resource, 0) < machine[resource] and (bank is None or bank not in banks[cursor]):
                break
            cursor += 1
        packets[cursor].append(instruction)
        resources[cursor][resource] = resources[cursor].get(resource, 0) + 1
        if bank is not None:
            banks[cursor].add(bank)
        completion = cursor + _latency(op)
        if op == "store":
            memory_ready[instruction[2]] = completion
        else:
            slot_ready[instruction[2]] = completion

    def allocate(excluded):
        free = next((slot for slot, node in enumerate(occupants) if node is None), None)
        if free is not None:
            return free
        choices = [node for node in residents if node not in excluded]

        def next_use(node):
            position = positions[node]
            return uses[node][position] if position < len(uses[node]) else float("inf")

        victim = max(choices, key=next_use)
        location = residents.pop(victim)
        if next_use(victim) != float("inf") and victim not in memory_ready:
            issue(("store", location, victim))
        occupants[location] = None
        return location

    for node in order:
        op, left, right = nodes[node]
        for source in (left, right):
            if source not in residents:
                location = allocate({left, right})
                issue(("load", source, location))
                residents[source] = location
                occupants[location] = source
        destination = allocate({left, right})
        issue((op, node, destination, residents[left], residents[right]))
        residents[node] = destination
        occupants[destination] = node
        if node in workload["outputs"]:
            issue(("store", destination, node))
        for source in (left, right):
            positions[source] += 1
    # Input outputs already reside in memory and need no instructions.
    return tuple(tuple(packet) for packet in packets) or ((),)


def solve(problem):
    return tuple(_schedule(workload) for workload in problem["workloads"])
