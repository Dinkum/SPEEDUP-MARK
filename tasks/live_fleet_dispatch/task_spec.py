"""Exact battery-limited fleet tours on a changing directed road graph."""

from __future__ import annotations

import bisect
import functools
import heapq
import math
import random

from speedupmark.catalog import TASK_CATALOG
from speedupmark.task import load_reference, plain_containers


_reference = load_reference(__file__)
_INF = _reference._INF
_add_label = _reference._add_label


def _reverse_labels(adjacency, destination, budget):
    """Independent sparse label-setting search over the reversed roads."""
    labels = [[] for _ in adjacency]
    labels[destination].append((0, 0))
    queue = [(0, 0, destination)]
    while queue:
        elapsed, used, vertex = heapq.heappop(queue)
        if (used, elapsed) not in labels[vertex]:
            continue
        for neighbor, travel_time, energy in adjacency[vertex].values():
            next_energy = used + energy
            next_time = elapsed + travel_time
            if next_energy <= budget and _add_label(
                    labels[neighbor], next_energy, next_time):
                heapq.heappush(queue, (next_time, next_energy, neighbor))
    return labels


def _checked_route_options(vehicle, jobs, profiles):
    """Enumerate ordered service sequences instead of merging by subset and endpoint."""
    origin, capacity, skills, budget = vehicle
    options = {0: 0}

    def visit(mask, location, labels, depth):
        if mask:
            for used, elapsed in labels:
                for energy, travel_time in profiles[location][origin]:
                    if used + energy <= budget:
                        options[mask] = min(options.get(mask, _INF),
                                            elapsed + travel_time)
        if depth == capacity:
            return
        for index, (destination, required, _) in enumerate(jobs):
            bit = 1 << index
            if mask & bit or not skills & required:
                continue
            next_labels = []
            for used, elapsed in labels:
                for energy, travel_time in profiles[location][destination]:
                    if used + energy <= budget:
                        _add_label(next_labels, used + energy,
                                   elapsed + travel_time)
            if next_labels:
                visit(mask | bit, destination, next_labels, depth + 1)

    visit(0, origin, [(0, 0)], 0)
    return options


def _checked_allocation(options_by_vehicle, jobs):
    options_by_job = []
    for options in options_by_vehicle:
        buckets = [[] for _ in jobs]
        for mask, cost in options.items():
            if mask:
                for index in range(len(jobs)):
                    if mask & (1 << index):
                        buckets[index].append((mask, cost))
        options_by_job.append(buckets)

    @functools.lru_cache(None)
    def best(remaining, available):
        if not remaining:
            return 0
        bit = remaining & -remaining
        index = bit.bit_length() - 1
        result = jobs[index][2] + best(remaining ^ bit, available)
        for vehicle in range(len(options_by_vehicle)):
            if not available & (1 << vehicle):
                continue
            for mask, cost in options_by_job[vehicle][index]:
                if mask & remaining == mask:
                    result = min(result, cost + best(remaining ^ mask,
                                                     available ^ (1 << vehicle)))
        return result

    return best((1 << len(jobs)) - 1, (1 << len(options_by_vehicle)) - 1)


def _checked_dispatch(reverse, vehicles, jobs):
    if not jobs:
        return 0
    budget = max((vehicle[3] for vehicle in vehicles), default=0)
    waypoints = {location for location, _, _, _ in vehicles}
    waypoints.update(job[0] for job in jobs)
    backward = {target: _reverse_labels(reverse, target, budget)
                for target in waypoints}
    profiles = {source: {target: tuple(backward[target][source])
                         for target in waypoints}
                for source in waypoints}
    options = [_checked_route_options(vehicle, jobs, profiles)
               for vehicle in vehicles]
    return _checked_allocation(options, jobs)


def _checked_visible_vehicles(events, cutoff, identifiers):
    vehicles = []
    for identifier in identifiers:
        ordered = sorted(events.get(identifier, ()))
        index = bisect.bisect_right(ordered, (cutoff, math.inf)) - 1
        if index >= 0:
            vehicles.append(ordered[index][2:])
    return tuple(vehicles)


def _checked_trace(problem):
    answers = []
    for scenario in problem['scenarios']:
        size = scenario['node_count']
        forward = [{} for _ in range(size)]
        reverse = [{} for _ in range(size)]
        for start, end, travel_time, energy in scenario['edges']:
            forward[start][end] = (end, travel_time, energy)
            reverse[end][start] = (start, travel_time, energy)
        rounds = []
        events = {}
        for sequence, operation in enumerate(scenario['operations']):
            kind = operation[0]
            if kind == 'set':
                _, start, end, travel_time, energy = operation
                forward[start][end] = (end, travel_time, energy)
                reverse[end][start] = (start, travel_time, energy)
            elif kind == 'delete':
                _, start, end = operation
                forward[start].pop(end, None)
                reverse[end].pop(start, None)
            elif kind == 'telemetry':
                _, identifier, timestamp, location, capacity, skills, battery = operation
                events.setdefault(identifier, []).append((timestamp, sequence, location, capacity, skills, battery))
            else:
                _, cutoff, identifiers, jobs = operation
                vehicles = _checked_visible_vehicles(events, cutoff, identifiers)
                rounds.append(_checked_dispatch(reverse, vehicles, jobs))
        answers.append(tuple(rounds))
    return tuple(answers)


class LiveFleetDispatchTask:
    name = "live_fleet_dispatch"
    task_version = "2.0.0"
    display_name = TASK_CATALOG[name].display_name
    default_n = 60
    grading_cases = (60, 96)

    def generate_problem(self, n=60, random_seed=0):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        size = max(20, n)
        rng = random.Random(random_seed)
        scenarios = []
        for family in ("grid", "hubs", "bottleneck"):
            edges = {}

            def add(start, end, express=False):
                if start != end:
                    if express:
                        edges[start, end] = (rng.randrange(1, 4), rng.randrange(4, 7))
                    else:
                        edges[start, end] = (rng.randrange(5, 10), 1)

            if family == "grid":
                width = math.isqrt(size)
                for vertex in range(size):
                    if vertex + 1 < size and vertex // width == (vertex + 1) // width:
                        add(vertex, vertex + 1)
                        add(vertex + 1, vertex)
                    if vertex + width < size:
                        add(vertex, vertex + width)
                        add(vertex + width, vertex)
                    if vertex + width + 1 < size and vertex % width + 1 < width:
                        add(vertex, vertex + width + 1, express=True)
                        add(vertex + width + 1, vertex, express=True)
            elif family == "hubs":
                hubs = max(2, size // 24)
                for vertex in range(size):
                    add(vertex, (vertex + 1) % size)
                    add((vertex + 1) % size, vertex)
                    for _ in range(2):
                        hub = rng.randrange(hubs)
                        add(vertex, hub, express=True)
                        add(hub, vertex, express=True)
            else:
                half = size // 2
                for start, stop in ((0, half), (half, size)):
                    for vertex in range(start, stop):
                        neighbor = start + (vertex - start + 1) % (stop - start)
                        add(vertex, neighbor)
                        add(neighbor, vertex)
                        add(vertex, rng.randrange(start, stop), express=True)
                add(half - 1, half)
                add(half, half - 1)
                add(half - 2, half + 1, express=True)
                add(half + 1, half - 2, express=True)

            initial = tuple((start, end, *value)
                            for (start, end), value in sorted(edges.items()))
            bridges = ((size // 2 - 1, size // 2), (size // 2, size // 2 - 1),
                       (size // 2 - 2, size // 2 + 1), (size // 2 + 1, size // 2 - 2))
            road_pairs = tuple(pair for pair in sorted(edges)
                               if family != "bottleneck" or pair not in bridges)
            depots = tuple(rng.randrange(size) for _ in range(4))
            job_count = 10 if size <= 64 else 12
            vehicle_count = 3 if size <= 64 else 4
            # Concentrated stops make longer tours feasible; dispersed jobs
            # retain the resource-constrained path problem. Every job remains
            # a separate service choice even when two share a road vertex.
            stops = rng.sample(range(size), min(size, job_count))
            capacities = tuple(3 + identifier % 2 for identifier in range(vehicle_count))
            jobs = [(rng.choice(stops), rng.choice((1, 2, 4)),
                     rng.randrange(25, 91)) for _ in range(job_count)]
            operations = []
            for identifier in range(vehicle_count):
                operations.append(("telemetry", identifier, 0, rng.choice(depots),
                                   capacities[identifier], rng.choice((3, 5, 6, 7)),
                                   rng.randrange(30, 45)))
            round_count = 8 if size <= 64 else 10
            for round_index in range(round_count):
                phase = 3 * round_index // round_count
                cutoff = 10 * (round_index + 1)
                for _ in range(1 if phase == 0 else 2):
                    identifier = rng.randrange(vehicle_count)
                    operations.append((
                        "telemetry", identifier,
                        max(0, cutoff - rng.choice((0, 5, 15, 25))),
                        rng.choice(depots) if rng.random() < 0.8 else rng.randrange(size),
                        capacities[identifier], rng.choice((3, 5, 6, 7)),
                        rng.randrange(30, 45),
                    ))
                if phase == 1:
                    start, end = rng.choice(road_pairs)
                    operations.append(("set", start, end, rng.randrange(1, 10),
                                       rng.randrange(1, 7)))
                elif phase == 2:
                    for _ in range(max(2, size // 24)):
                        start, end = rng.choice(road_pairs)
                        if rng.random() < 0.2:
                            operations.append(("delete", start, end))
                        else:
                            operations.append(("set", start, end, rng.randrange(1, 10),
                                               rng.randrange(1, 7)))
                if family == "bottleneck" and round_index in (
                        round_count // 3, 2 * round_count // 3):
                    for start, end in bridges:
                        if round_index == round_count // 3:
                            operations.append(("delete", start, end))
                        else:
                            operations.append(("set", start, end,
                                               rng.randrange(1, 10),
                                               rng.randrange(1, 7)))
                change_count = 0 if phase == 0 else 1 if phase == 1 else 3
                for _ in range(change_count):
                    index = rng.randrange(job_count)
                    jobs[index] = (rng.choice(stops) if phase == 1 else rng.randrange(size),
                                   rng.choice((1, 2, 4)),
                                   rng.randrange(25, 91))
                operations.append(("dispatch", cutoff, tuple(range(vehicle_count)),
                                   tuple(jobs)))
            scenarios.append({"family": family, "node_count": size,
                              "edges": initial, "operations": tuple(operations)})
        return {"scenarios": tuple(scenarios)}

    solve = staticmethod(_reference.solve)


    def is_solution(self, problem, proposed):
        if (not plain_containers(proposed) or type(proposed) not in (tuple, list)
                or len(proposed) != len(problem["scenarios"])):
            return False
        if any(type(rounds) not in (tuple, list)
               or any(type(cost) is not int for cost in rounds)
               for rounds in proposed):
            return False
        return tuple(tuple(rounds) for rounds in proposed) == _checked_trace(problem)


TASK = LiveFleetDispatchTask()
