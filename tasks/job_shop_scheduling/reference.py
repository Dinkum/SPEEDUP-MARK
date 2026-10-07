"""Reference implementation for job_shop_scheduling; copied into fresh run candidates."""

import itertools


def _operation_data(problem):
    jobs = problem["jobs"]
    machine_count = problem["machines"]
    durations = []
    machines = []
    operation_for_job_machine = [[None] * machine_count for _ in jobs]
    machine_operations = [[] for _ in range(machine_count)]
    for job_index, job in enumerate(jobs):
        for step, (machine, duration) in enumerate(job):
            operation = job_index * machine_count + step
            durations.append(duration)
            machines.append(machine)
            operation_for_job_machine[job_index][machine] = operation
            machine_operations[machine].append(operation)
    job_edges = []
    for job_index in range(len(jobs)):
        first = job_index * machine_count
        job_edges.extend((first + step, first + step + 1) for step in range(machine_count - 1))
    return durations, machines, operation_for_job_machine, machine_operations, job_edges


def _schedule_with_topological_pass(durations, job_edges, machine_orders):
    """Build earliest starts for a partial or complete set of machine orders."""
    count = len(durations)
    adjacency = [[] for _ in range(count)]
    indegree = [0] * count
    edges = list(job_edges)
    for order in machine_orders:
        if order is not None:
            edges.extend(zip(order, order[1:]))
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    starts = [0] * count
    position = 0
    while position < len(ready):
        source = ready[position]
        position += 1
        finish = starts[source] + durations[source]
        for target in adjacency[source]:
            if finish > starts[target]:
                starts[target] = finish
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(ready) != count:
        return None
    return starts, max((starts[i] + durations[i] for i in range(count)), default=0)


def _reference_schedule(problem):
    """Enumerate machine sequences with critical-path bounds."""
    durations, _, operation_for_job_machine, machine_operations, job_edges = _operation_data(problem)
    jobs, machine_count = len(problem["jobs"]), problem["machines"]
    orders_by_machine = [tuple(itertools.permutations(ops)) for ops in machine_operations]

    # A shared job order on every machine is always acyclic and gives an initial
    # feasible incumbent before the exact machine-order search starts.
    best_starts = None
    best_makespan = float("inf")
    for job_order in itertools.permutations(range(jobs)):
        seed_orders = tuple(
            tuple(operation_for_job_machine[job][machine] for job in job_order)
            for machine in range(machine_count)
        )
        schedule = _schedule_with_topological_pass(durations, job_edges, seed_orders)
        if schedule and schedule[1] < best_makespan:
            best_starts, best_makespan = schedule

    chosen = [None] * machine_count

    def search(machine):
        nonlocal best_starts, best_makespan
        partial = _schedule_with_topological_pass(durations, job_edges, chosen)
        if partial is None or partial[1] >= best_makespan:
            return
        if machine == machine_count:
            best_starts, best_makespan = partial
            return
        for order in orders_by_machine[machine]:
            chosen[machine] = order
            search(machine + 1)
        chosen[machine] = None

    search(0)
    start_times = [
        [best_starts[job * machine_count + step] for step in range(machine_count)]
        for job in range(jobs)
    ]
    return {"start_times": start_times, "makespan": int(best_makespan)}


def solve(problem):
    return _reference_schedule(problem)
