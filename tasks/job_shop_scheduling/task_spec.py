"""Exact makespan scheduling for small fixed-route job shops."""

import itertools
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference


_reference = load_reference(__file__)
_operation_data = _reference._operation_data


def _relaxation_makespan(durations, edges):
    """Independent longest-path check by repeated constraint relaxation."""
    starts = [0] * len(durations)
    # One extra pass distinguishes a longest DAG path (at most N-1 arcs) from
    # a positive cycle even when edge order delays propagation by one node.
    for _ in range(len(durations) + 1):
        changed = False
        for source, target in edges:
            required = starts[source] + durations[source]
            if starts[target] < required:
                starts[target] = required
                changed = True
        if not changed:
            return max((starts[i] + durations[i] for i in range(len(durations))), default=0)
    # Positive durations make any directed cycle infeasible: each relaxation
    # pass would continue increasing at least one start time.
    return None


def _optimal_makespan(problem):
    durations, _, _, machine_operations, job_edges = _operation_data(problem)
    orders_by_machine = [tuple(itertools.permutations(ops)) for ops in machine_operations]
    best = float("inf")
    for machine_orders in itertools.product(*orders_by_machine):
        edges = list(job_edges)
        for order in machine_orders:
            edges.extend(zip(order, order[1:]))
        makespan = _relaxation_makespan(durations, edges)
        if makespan is not None and makespan < best:
            best = makespan
    return best


class Task:
    name = "job_shop_scheduling"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 4
    grading_cases = (3, 4)

    def generate_problem(self, n=4, random_seed=0):
        if type(n) is not int or not 2 <= n <= 4:
            raise ValueError("n must be an integer from 2 through 4 jobs")
        rng = random.Random(random_seed)
        machine_count = 3
        jobs = []
        for job in range(n):
            route = list(range(machine_count))
            rng.shuffle(route)
            operations = []
            for machine in route:
                # Each machine receives a long bottleneck operation in a
                # different job; the other operations create route conflicts.
                bottleneck_job = (machine + random_seed) % n
                duration = rng.randint(8, 13) if job == bottleneck_job else rng.randint(2, 7)
                operations.append((machine, duration))
            jobs.append(tuple(operations))
        return {"machines": machine_count, "jobs": tuple(jobs)}

    solve = staticmethod(_reference.solve)

    def is_solution(self, problem, proposed):
        if type(proposed) is not dict or set(proposed) != {"start_times", "makespan"}:
            return False
        starts, reported = proposed["start_times"], proposed["makespan"]
        jobs, machine_count = problem["jobs"], problem["machines"]
        if type(starts) is not list or len(starts) != len(jobs) or type(reported) is not int:
            return False
        if any(type(row) is not list or len(row) != machine_count for row in starts):
            return False
        if any(type(t) is not int or t < 0 for row in starts for t in row):
            return False
        machine_intervals = [[] for _ in range(machine_count)]
        actual_makespan = 0
        for job_index, job in enumerate(jobs):
            for step, (machine, duration) in enumerate(job):
                start = starts[job_index][step]
                finish = start + duration
                actual_makespan = max(actual_makespan, finish)
                if step and start < starts[job_index][step - 1] + job[step - 1][1]:
                    return False
                machine_intervals[machine].append((start, finish))
        if reported != actual_makespan:
            return False
        for intervals in machine_intervals:
            intervals.sort()
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                return False
        return reported == _optimal_makespan(problem)


TASK = Task()
